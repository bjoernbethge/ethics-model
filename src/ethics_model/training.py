"""
Training Functions for Ethics Model

Enhanced training functions for the ethics model with support for
GraphBrain semantic hypergraphs.
"""

import copy
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Union, Callable


def train(
    model: torch.nn.Module,
    llm: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    writer: Optional[Any] = None,
    device: torch.device = torch.device('cpu'),
    epochs: int = 10,
    patience: int = 2,
    grad_clip: float = 1.0,
    symbolic_constraints: Optional[Callable] = None,
    checkpoint_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Train the ethics model using LLM embeddings.
    
    Args:
        model: Ethics model
        llm: Language model for embeddings
        dataloader: DataLoader with training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        writer: Optional TensorBoard writer
        device: Device for training
        epochs: Maximum number of training epochs
        patience: Patience for early stopping
        grad_clip: Gradient clipping value
        symbolic_constraints: Optional symbolic constraints
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
        total_ethics, total_manip = 0.0, 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            
            # Extract raw texts if available (for GraphBrain)
            texts = batch.get('texts', None)
            
            # Extract graph data if available
            graph_data = {}
            for k in batch.keys():
                if k.startswith('graph_'):
                    graph_data[k.replace('graph_', '')] = batch[k]
            
            # Get LLM embeddings
            with torch.no_grad():
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                hidden_states = llm_outputs.last_hidden_state
            
            optimizer.zero_grad()
            
            # Mixed precision training if available
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        embeddings=hidden_states,
                        attention_mask=attention_mask,
                        texts=texts,
                        graph_data=graph_data if graph_data else None,
                        symbolic_constraints=symbolic_constraints
                    )
                    
                    # Calculate losses
                    ethics_score = outputs['ethics_score']
                    manipulation_score = outputs['manipulation_score']
                    loss_ethics = criterion(ethics_score, ethics_label)
                    loss_manip = criterion(manipulation_score, manipulation_label)
                    
                    # Standard loss
                    loss = loss_ethics + 0.5 * loss_manip
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(
                    embeddings=hidden_states,
                    attention_mask=attention_mask,
                    texts=texts,
                    graph_data=graph_data if graph_data else None,
                    symbolic_constraints=symbolic_constraints
                )
                
                # Calculate losses
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                loss_ethics = criterion(ethics_score, ethics_label)
                loss_manip = criterion(manipulation_score, manipulation_label)
                
                # Standard loss
                loss = loss_ethics + 0.5 * loss_manip
                
                # Backward and optimize
                loss.backward()
                
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            # Track metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_ethics += loss_ethics.item() * batch_size
            total_manip += loss_manip.item() * batch_size
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate epoch metrics
        dataset_size = len(dataloader.dataset)
        avg_loss = total_loss / dataset_size
        avg_ethics = total_ethics / dataset_size
        avg_manip = total_manip / dataset_size
        
        # Log to TensorBoard if available
        if writer is not None:
            writer.add_scalar('Loss/Total', avg_loss, epoch+1)
            writer.add_scalar('Loss/Ethics', avg_ethics, epoch+1)
            writer.add_scalar('Loss/Manipulation', avg_manip, epoch+1)
            
            # Log additional metrics
            writer.add_scalar('Score/Ethics', ethics_score.mean().item(), epoch+1)
            writer.add_scalar('Score/Manipulation', manipulation_score.mean().item(), epoch+1)
        
        # Early Stopping & Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            if checkpoint_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model
