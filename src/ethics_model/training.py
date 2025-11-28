"""
Training functions for Ethics Model

This module provides training functions for the ethics model, including:
- Simple training function for basic use cases
- CUDA-optimized training with CUDA Graphs and CUDA Streams for efficient GPU utilization
"""

import logging
import os
import copy
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CUDAGraphTrainer:
    """
    CUDA Graph-based trainer for ethics model.
    
    This trainer uses CUDA Graphs to optimize the training process,
    capturing and replaying the computation graph for efficient execution.
    
    Args:
        model: Ethics model to train
        llm: Language model for embeddings
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device for training
        use_amp: Whether to use mixed precision training
        use_cuda_graphs: Whether to use CUDA Graphs
        grad_clip: Gradient clipping value
        checkpoint_path: Path to save best checkpoint
    """
    
    def __init__(
        self,
        model: nn.Module,
        llm: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: torch.device,
        use_amp: bool = True,
        use_cuda_graphs: bool = True,
        grad_clip: float = 1.0,
        checkpoint_path: Optional[str] = None
    ):
        self.model = model
        self.llm = llm
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.use_cuda_graphs = use_cuda_graphs and device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 7
        self.grad_clip = grad_clip
        self.checkpoint_path = checkpoint_path
        
        # Initialize AMP scaler if using mixed precision
        self.scaler = amp.GradScaler() if self.use_amp else None
        
        # Graph capture related variables
        self.graph = None
        self.static_inputs = None
        self.static_outputs = None
        
        # Check compatibility for CUDA Graphs
        if self.use_cuda_graphs and not hasattr(torch.cuda, 'graphs'):
            logger.warning("CUDA Graphs not available in this PyTorch version. Falling back to standard training.")
            self.use_cuda_graphs = False
        
        # Create streams for overlap
        if self.device.type == 'cuda':
            self.streams = {
                'llm': torch.cuda.Stream(),
                'ethics': torch.cuda.Stream(),
                'graph': torch.cuda.Stream() if self.use_cuda_graphs else None
            }
        else:
            self.streams = None
    
    def _prepare_static_batch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Prepare a static batch for CUDA Graph capture.
        
        Args:
            dataloader: DataLoader to get a batch from
            
        Returns:
            Dictionary containing static batch data
        """
        # Get a batch
        static_batch = next(iter(dataloader))
        
        # Move to device
        for k, v in static_batch.items():
            if isinstance(v, torch.Tensor):
                static_batch[k] = v.to(self.device)
        
        return static_batch
    
    def _capture_graph(self, static_batch: Dict[str, Any]) -> None:
        """
        Capture computation graph for static batch.
        
        Args:
            static_batch: Static batch for graph capture
        """
        # Ensure we're using CUDA
        if not self.use_cuda_graphs:
            return
        
        # Set static batch shapes
        self.static_inputs = {
            'input_ids': static_batch['input_ids'],
            'attention_mask': static_batch['attention_mask'],
            'ethics_label': static_batch['ethics_label'],
            'manipulation_label': static_batch['manipulation_label']
        }
        
        # Extract texts if available
        if 'texts' in static_batch:
            self.static_inputs['texts'] = static_batch['texts']
        
        # Prepare static outputs
        self.static_outputs = {
            'loss': torch.zeros(1, device=self.device),
            'ethics_score': torch.zeros(static_batch['ethics_label'].shape, device=self.device),
            'manipulation_score': torch.zeros(static_batch['manipulation_label'].shape, device=self.device)
        }
        
        # Capture the graph
        with torch.cuda.graph(torch.cuda.Graph()) as self.graph:
            # Forward LLM
            with torch.no_grad():
                llm_outputs = self.llm.model.transformer(self.static_inputs['input_ids']) if hasattr(self.llm, 'model') else self.llm.transformer(self.static_inputs['input_ids'])
                hidden_states = llm_outputs.last_hidden_state
            
            # Forward ethics model
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with amp.autocast():
                    outputs = self.model(
                        embeddings=hidden_states,
                        attention_mask=self.static_inputs['attention_mask'],
                        texts=self.static_inputs.get('texts')
                    )
                    
                    ethics_score = outputs['ethics_score']
                    manipulation_score = outputs['manipulation_score']
                    
                    loss_ethics = self.criterion(ethics_score, self.static_inputs['ethics_label'])
                    loss_manip = self.criterion(manipulation_score, self.static_inputs['manipulation_label'])
                    
                    loss = loss_ethics + 0.5 * loss_manip
                    
                    # Copy to static outputs
                    self.static_outputs['loss'].copy_(loss)
                    self.static_outputs['ethics_score'].copy_(ethics_score)
                    self.static_outputs['manipulation_score'].copy_(manipulation_score)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    embeddings=hidden_states,
                    attention_mask=self.static_inputs['attention_mask'],
                    texts=self.static_inputs.get('texts')
                )
                
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                
                loss_ethics = self.criterion(ethics_score, self.static_inputs['ethics_label'])
                loss_manip = self.criterion(manipulation_score, self.static_inputs['manipulation_label'])
                
                loss = loss_ethics + 0.5 * loss_manip
                
                # Copy to static outputs
                self.static_outputs['loss'].copy_(loss)
                self.static_outputs['ethics_score'].copy_(ethics_score)
                self.static_outputs['manipulation_score'].copy_(manipulation_score)
                
                # Backward pass
                loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
    
    def _replay_graph(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replay captured graph with new batch data.
        
        Args:
            batch: New batch data
            
        Returns:
            Dictionary containing outputs
        """
        # Copy batch data to static inputs
        for k, v in self.static_inputs.items():
            if k == 'texts':
                self.static_inputs[k] = batch[k]
            elif isinstance(v, torch.Tensor) and k in batch:
                self.static_inputs[k].copy_(batch[k])
        
        # Replay the graph
        self.graph.replay()
        
        # Return outputs
        return {
            'loss': self.static_outputs['loss'].item(),
            'ethics_score': self.static_outputs['ethics_score'].detach(),
            'manipulation_score': self.static_outputs['manipulation_score'].detach()
        }
    
    def _train_step_with_streams(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a training step using CUDA streams for overlapping computation.
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary of loss values
        """
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        # Process LLM in stream 1
        with torch.cuda.stream(self.streams['llm']):
            with torch.no_grad():
                llm_outputs = self.llm.model.transformer(batch['input_ids']) if hasattr(self.llm, 'model') else self.llm.transformer(batch['input_ids'])
                hidden_states = llm_outputs.last_hidden_state
        
        # Wait for LLM to finish
        self.streams['ethics'].wait_stream(self.streams['llm'])
        
        # Process ethics model in stream 2
        with torch.cuda.stream(self.streams['ethics']):
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with amp.autocast():
                    outputs = self.model(
                        embeddings=hidden_states,
                        attention_mask=batch['attention_mask'],
                        texts=batch.get('texts')
                    )
                    
                    ethics_score = outputs['ethics_score']
                    manipulation_score = outputs['manipulation_score']
                    
                    loss_ethics = self.criterion(ethics_score, batch['ethics_label'])
                    loss_manip = self.criterion(manipulation_score, batch['manipulation_label'])
                    
                    loss = loss_ethics + 0.5 * loss_manip
                
                # Backward pass with AMP
                self.scaler.scale(loss).backward()
                
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    embeddings=hidden_states,
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts')
                )
                
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                
                loss_ethics = self.criterion(ethics_score, batch['ethics_label'])
                loss_manip = self.criterion(manipulation_score, batch['manipulation_label'])
                
                loss = loss_ethics + 0.5 * loss_manip
                
                # Backward pass
                loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
        
        # Synchronize for metrics calculation
        torch.cuda.synchronize()
        
        return {
            'loss': loss.item(),
            'ethics_loss': loss_ethics.item(),
            'manipulation_loss': loss_manip.item()
        }
    
    def _train_step_standard(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a standard training step without CUDA streams.
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary of loss values
        """
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        # Forward LLM
        with torch.no_grad():
            llm_outputs = self.llm.model.transformer(batch['input_ids']) if hasattr(self.llm, 'model') else self.llm.transformer(batch['input_ids'])
            hidden_states = llm_outputs.last_hidden_state
        
        # Forward ethics model
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with amp.autocast():
                outputs = self.model(
                    embeddings=hidden_states,
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts')
                )
                
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                
                loss_ethics = self.criterion(ethics_score, batch['ethics_label'])
                loss_manip = self.criterion(manipulation_score, batch['manipulation_label'])
                
                loss = loss_ethics + 0.5 * loss_manip
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                embeddings=hidden_states,
                attention_mask=batch['attention_mask'],
                texts=batch.get('texts')
            )
            
            ethics_score = outputs['ethics_score']
            manipulation_score = outputs['manipulation_score']
            
            loss_ethics = self.criterion(ethics_score, batch['ethics_label'])
            loss_manip = self.criterion(manipulation_score, batch['manipulation_label'])
            
            loss = loss_ethics + 0.5 * loss_manip
            
            # Backward pass
            loss.backward()
            
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'ethics_loss': loss_ethics.item(),
            'manipulation_loss': loss_manip.item()
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary of average loss values
        """
        self.model.train()
        total_loss = 0.0
        total_ethics_loss = 0.0
        total_manipulation_loss = 0.0
        
        # Determine if we can use CUDA Graphs
        static_shapes = self.use_cuda_graphs
        
        # Check if all batches have the same shape
        if static_shapes and len(set([batch['input_ids'].shape[0] for batch in dataloader])) > 1:
            logger.warning("Variable batch sizes detected. CUDA Graphs require static shapes. Disabling CUDA Graphs.")
            static_shapes = False
            self.use_cuda_graphs = False
        
        # Capture graph if using CUDA Graphs
        if static_shapes:
            static_batch = self._prepare_static_batch(dataloader)
            self._capture_graph(static_batch)
        
        # Training loop
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Training step
            if static_shapes and self.graph is not None:
                # Use captured graph
                outputs = self._replay_graph(batch)
                metrics = {
                    'loss': outputs['loss'],
                    'ethics_loss': 0.0,  # Not available in graph replay
                    'manipulation_loss': 0.0  # Not available in graph replay
                }
            elif self.streams is not None:
                # Use CUDA streams
                metrics = self._train_step_with_streams(batch)
            else:
                # Standard training
                metrics = self._train_step_standard(batch)
            
            # Update totals
            total_loss += metrics['loss'] * batch['input_ids'].size(0)
            total_ethics_loss += metrics.get('ethics_loss', 0.0) * batch['input_ids'].size(0)
            total_manipulation_loss += metrics.get('manipulation_loss', 0.0) * batch['input_ids'].size(0)
            
            # Update progress bar
            pbar.set_postfix({"loss": metrics['loss']})
        
        # Calculate averages
        dataset_size = len(dataloader.dataset)
        avg_loss = total_loss / dataset_size
        avg_ethics_loss = total_ethics_loss / dataset_size
        avg_manipulation_loss = total_manipulation_loss / dataset_size
        
        return {
            'loss': avg_loss,
            'ethics_loss': avg_ethics_loss,
            'manipulation_loss': avg_manipulation_loss
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_ethics_loss = 0.0
        total_manipulation_loss = 0.0
        
        # Prediction metrics
        ethics_correct = 0
        manipulation_correct = 0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Forward LLM
                llm_outputs = self.llm.model.transformer(batch['input_ids']) if hasattr(self.llm, 'model') else self.llm.transformer(batch['input_ids'])
                hidden_states = llm_outputs.last_hidden_state
                
                # Forward ethics model
                outputs = self.model(
                    embeddings=hidden_states,
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts')
                )
                
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                
                # Calculate losses
                loss_ethics = self.criterion(ethics_score, batch['ethics_label'])
                loss_manip = self.criterion(manipulation_score, batch['manipulation_label'])
                
                loss = loss_ethics + 0.5 * loss_manip
                
                # Update totals
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_ethics_loss += loss_ethics.item() * batch['input_ids'].size(0)
                total_manipulation_loss += loss_manip.item() * batch['input_ids'].size(0)
                
                # Binary classification accuracy (threshold = 0.5)
                ethics_pred = (ethics_score >= 0.5).float()
                ethics_correct += (ethics_pred == (batch['ethics_label'] >= 0.5).float()).sum().item()
                
                manipulation_pred = (manipulation_score >= 0.5).float()
                manipulation_correct += (manipulation_pred == (batch['manipulation_label'] >= 0.5).float()).sum().item()
        
        # Calculate metrics
        dataset_size = len(dataloader.dataset)
        avg_loss = total_loss / dataset_size
        avg_ethics_loss = total_ethics_loss / dataset_size
        avg_manipulation_loss = total_manipulation_loss / dataset_size
        
        ethics_accuracy = ethics_correct / dataset_size
        manipulation_accuracy = manipulation_correct / dataset_size
        
        return {
            'loss': avg_loss,
            'ethics_loss': avg_ethics_loss,
            'manipulation_loss': avg_manipulation_loss,
            'ethics_accuracy': ethics_accuracy,
            'manipulation_accuracy': manipulation_accuracy
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 10,
        patience: int = 2,
        writer: Any = None
    ) -> nn.Module:
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            test_dataloader: Optional DataLoader for test data
            epochs: Maximum number of training epochs
            patience: Patience for early stopping
            writer: Optional TensorBoard writer
            
        Returns:
            Trained model
        """
        best_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation
            val_metrics = None
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                
                # Log metrics
                if writer is not None:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f'Validation/{k}', v, epoch+1)
            
            # Determine loss for early stopping
            current_loss = val_metrics['loss'] if val_metrics is not None else train_metrics['loss']
            
            # Log training metrics
            if writer is not None:
                for k, v in train_metrics.items():
                    writer.add_scalar(f'Training/{k}', v, epoch+1)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Training Loss: {train_metrics['loss']:.4f}")
            if val_metrics is not None:
                logger.info(f"  Validation Loss: {val_metrics['loss']:.4f}")
                logger.info(f"  Ethics Accuracy: {val_metrics['ethics_accuracy']:.4f}")
                logger.info(f"  Manipulation Accuracy: {val_metrics['manipulation_accuracy']:.4f}")
            
            # Early stopping & checkpointing
            if current_loss < best_loss:
                best_loss = current_loss
                best_model = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                
                # Save checkpoint
                if self.checkpoint_path is not None:
                    os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                        'metrics': val_metrics if val_metrics is not None else train_metrics
                    }, self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # Final evaluation on test set
        if test_dataloader is not None:
            test_metrics = self.evaluate(test_dataloader)
            logger.info("Test Results:")
            for k, v in test_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            
            if writer is not None:
                for k, v in test_metrics.items():
                    writer.add_scalar(f'Test/{k}', v, 0)
        
        return self.model


def train_ethics_model(
    model: nn.Module,
    llm: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    patience: int = 2,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    checkpoint_path: Optional[str] = None,
    writer: Any = None,
    use_amp: bool = True,
    use_cuda_graphs: bool = True
) -> nn.Module:
    """
    Train an ethics model with CUDA optimizations.
    
    Args:
        model: Ethics model to train
        llm: Language model for embeddings
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        test_dataloader: Optional DataLoader for test data
        optimizer: PyTorch optimizer (default: AdamW)
        criterion: Loss function (default: BCELoss)
        device: Device for training (default: GPU if available)
        epochs: Maximum number of training epochs
        patience: Patience for early stopping
        learning_rate: Learning rate if optimizer not provided
        grad_clip: Gradient clipping value
        checkpoint_path: Path to save best checkpoint
        writer: Optional TensorBoard writer
        use_amp: Whether to use mixed precision training
        use_cuda_graphs: Whether to use CUDA Graphs
        
    Returns:
        Trained model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move models to device
    model = model.to(device)
    llm = llm.to(device)
    
    # Set optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Set criterion if not provided
    if criterion is None:
        criterion = nn.BCELoss()
    
    # Create trainer
    trainer = CUDAGraphTrainer(
        model=model,
        llm=llm,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        use_cuda_graphs=use_cuda_graphs,
        grad_clip=grad_clip,
        checkpoint_path=checkpoint_path
    )
    
    # Train model
    model = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
        patience=patience,
        writer=writer
    )
    
    return model


def train(
    model: nn.Module,
    llm: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    writer: Any,
    device: torch.device,
    epochs: int = 10,
    patience: int = 2,
    grad_clip: float = 1.0,
    symbolic_constraints: Optional[Callable] = None,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Simple training function for ethics model.
    
    This is a simplified training function without CUDA optimizations.
    For optimized training, use train_ethics_model() instead.
    
    Args:
        model: Ethics model to train
        llm: Language model for embeddings
        dataloader: DataLoader for training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        writer: Optional TensorBoard writer
        device: Device for training
        epochs: Maximum number of training epochs
        patience: Patience for early stopping
        grad_clip: Gradient clipping value
        symbolic_constraints: Optional symbolic constraints function
        checkpoint_path: Path to save best checkpoint
        
    Returns:
        Trained model
    """
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ethics, total_manip, n_batches = 0.0, 0.0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            
            with torch.no_grad():
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                hidden_states = llm_outputs.last_hidden_state
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        embeddings=hidden_states,
                        attention_mask=attention_mask,
                        texts=batch.get('texts'),
                        symbolic_constraints=symbolic_constraints
                    )
                    ethics_score = outputs['ethics_score']
                    manipulation_score = outputs['manipulation_score']
                    loss_ethics = criterion(ethics_score, ethics_label)
                    loss_manip = criterion(manipulation_score, manipulation_label)
                    loss = loss_ethics + 0.5 * loss_manip
                
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    embeddings=hidden_states,
                    attention_mask=attention_mask,
                    texts=batch.get('texts'),
                    symbolic_constraints=symbolic_constraints
                )
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                loss_ethics = criterion(ethics_score, ethics_label)
                loss_manip = criterion(manipulation_score, manipulation_label)
                loss = loss_ethics + 0.5 * loss_manip
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            total_loss += loss.item() * input_ids.size(0)
            total_ethics += loss_ethics.item() * input_ids.size(0)
            total_manip += loss_manip.item() * input_ids.size(0)
            n_batches += input_ids.size(0)
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader.dataset)
        avg_ethics = total_ethics / len(dataloader.dataset)
        avg_manip = total_manip / len(dataloader.dataset)
        
        if writer is not None:
            writer.add_scalar('Loss/Total', avg_loss, epoch+1)
            writer.add_scalar('Loss/Ethics', avg_ethics, epoch+1)
            writer.add_scalar('Loss/Manipulation', avg_manip, epoch+1)
        
        # Early Stopping & Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(best_model, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model 
