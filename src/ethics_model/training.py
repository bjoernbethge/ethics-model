"""
Training Functions for Ethics Model

Enhanced training functions for the ethics model with support for
NetworkX graphs and spaCy NLP processing.
"""

import copy
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Union, Callable
import time


def train(
    model: torch.nn.Module,
    llm: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    writer: Optional[Any] = None,
    device: Union[torch.device, str] = torch.device('cpu'),
    epochs: int = 10,
    patience: int = 2,
    grad_clip: float = 1.0,
    symbolic_constraints: Optional[Callable] = None,
    checkpoint_path: Optional[str] = None,
    use_amp: bool = True,
    log_interval: int = 100,
    validate_model: Optional[Callable] = None,
    validation_dataloader: Optional[torch.utils.data.DataLoader] = None
) -> torch.nn.Module:
    """
    Train the ethics model using LLM embeddings with enhanced graph support.
    
    Args:
        model: Ethics model to train
        llm: Language model for generating embeddings
        dataloader: DataLoader with training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        writer: Optional TensorBoard writer for logging
        device: Device for training (CPU/CUDA)
        epochs: Maximum number of training epochs
        patience: Patience for early stopping
        grad_clip: Gradient clipping value (0 to disable)
        symbolic_constraints: Optional symbolic constraints function
        checkpoint_path: Path to save best checkpoint
        use_amp: Whether to use automatic mixed precision
        log_interval: How often to log detailed metrics
        validate_model: Optional validation function
        validation_dataloader: Optional validation dataloader
        
    Returns:
        Trained model with best weights loaded
    """
    print(f"Starting training on device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Training for {epochs} epochs with patience {patience}")
    
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    # Initialize tracking variables
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    # Set up automatic mixed precision if requested and available
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Using automatic mixed precision training")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Track metrics
        total_loss = 0.0
        total_ethics_loss = 0.0
        total_manip_loss = 0.0
        num_samples = 0
        start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                if input_ids.dtype != torch.long:
                    input_ids = input_ids.long()
                attention_mask = batch['attention_mask'].to(device)
                if attention_mask.dtype != torch.float32:
                    attention_mask = attention_mask.float()
                
                ethics_label = batch['ethics_label'].to(device)
                manipulation_label = batch['manipulation_label'].to(device)
                
                # Extract texts for graph processing if available
                texts = batch.get('texts', None)
                
                # Get LLM embeddings (no gradients needed)
                with torch.no_grad():
                    try:
                        # Methode 1: Direkt mit Huggingface-API für Transformer-Modelle
                        if hasattr(llm, 'forward'):
                            outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
                            
                            # DistilBERT und ähnliche Modelle geben ein Dictionary/Objekt zurück
                            if hasattr(outputs, 'last_hidden_state'):
                                hidden_states = outputs.last_hidden_state
                            elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
                                hidden_states = outputs['last_hidden_state']
                            elif isinstance(outputs, tuple) and len(outputs) > 0:
                                hidden_states = outputs[0]  # Fallback für Tuple-Output
                            else:
                                print(f"Unbekannte LLM-Ausgabe vom Typ {type(outputs)}, versuche Fallback...")
                                hidden_states = outputs  # Direkter Fallback als letzter Versuch
                        else:
                            # Veraltete Methode für nicht-standardkonforme Modelle
                            print(f"LLM hat keine forward-Methode, verwende Fallback.")
                            if hasattr(llm, 'model') and hasattr(llm.model, 'transformer'):
                                hidden_states = llm.model.transformer(input_ids)
                            elif hasattr(llm, 'transformer'):
                                hidden_states = llm.transformer(input_ids)
                            else:
                                hidden_states = llm(input_ids)
                                
                            # Bei Bedarf letzte hidden states extrahieren
                            if hasattr(hidden_states, 'last_hidden_state'):
                                hidden_states = hidden_states.last_hidden_state
                            elif isinstance(hidden_states, tuple) and len(hidden_states) > 0:
                                hidden_states = hidden_states[0]
                    except Exception as e:
                        print(f"Fehler beim Extrahieren der LLM embeddings: {e}")
                        print(f"LLM Typ: {llm.__class__.__name__}")
                        print(f"Input shape: {input_ids.shape}")
                        raise
                
                # Debug-Check: NoneType und NaN
                assert hidden_states is not None, f"LLM hidden_states is None in batch {batch_idx} (input_ids shape: {input_ids.shape}, LLM Typ: {llm.__class__.__name__})"
                assert not torch.isnan(hidden_states).any(), f"LLM hidden_states contains NaN in batch {batch_idx} (input_ids shape: {input_ids.shape})"
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with optional mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            embeddings=hidden_states,
                            attention_mask=attention_mask,
                            texts=texts,
                            symbolic_constraints=symbolic_constraints
                        )
                        
                        # Calculate losses
                        ethics_score = outputs['ethics_score']
                        manipulation_score = outputs['manipulation_score']
                        
                        loss_ethics = criterion(ethics_score, ethics_label.view(-1, 1).float())
                        loss_manip = criterion(manipulation_score, manipulation_label.view(-1, 1).float())
                        
                        # Combined loss (weight manipulation loss lower)
                        loss = loss_ethics + 0.5 * loss_manip
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Standard precision forward pass
                    outputs = model(
                        embeddings=hidden_states,
                        attention_mask=attention_mask,
                        texts=texts,
                        symbolic_constraints=symbolic_constraints
                    )
                    
                    # Calculate losses
                    ethics_score = outputs['ethics_score']
                    manipulation_score = outputs['manipulation_score']
                    
                    loss_ethics = criterion(ethics_score, ethics_label.view(-1, 1).float())
                    loss_manip = criterion(manipulation_score, manipulation_label.view(-1, 1).float())
                    
                    # Combined loss
                    loss = loss_ethics + 0.5 * loss_manip
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    # Optimizer step
                    optimizer.step()
                
                # Track metrics
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_ethics_loss += loss_ethics.item() * batch_size
                total_manip_loss += loss_manip.item() * batch_size
                num_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ethics": f"{loss_ethics.item():.4f}",
                    "manip": f"{loss_manip.item():.4f}"
                })
                
                # Detailed logging
                if writer is not None and batch_idx % log_interval == 0:
                    step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar('Train/Batch_Loss', loss.item(), step)
                    writer.add_scalar('Train/Batch_Ethics_Loss', loss_ethics.item(), step)
                    writer.add_scalar('Train/Batch_Manip_Loss', loss_manip.item(), step)
                    writer.add_scalar('Train/Ethics_Score_Mean', ethics_score.mean().item(), step)
                    writer.add_scalar('Train/Manip_Score_Mean', manipulation_score.mean().item(), step)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        avg_ethics_loss = total_ethics_loss / num_samples if num_samples > 0 else float('inf')
        avg_manip_loss = total_manip_loss / num_samples if num_samples > 0 else float('inf')
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, "
              f"Ethics: {avg_ethics_loss:.4f}, Manip: {avg_manip_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Log epoch metrics
        if writer is not None:
            writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch + 1)
            writer.add_scalar('Train/Epoch_Ethics_Loss', avg_ethics_loss, epoch + 1)
            writer.add_scalar('Train/Epoch_Manip_Loss', avg_manip_loss, epoch + 1)
            writer.add_scalar('Train/Epoch_Time', epoch_time, epoch + 1)
        
        # Validation
        val_loss = None
        if validate_model is not None and validation_dataloader is not None:
            val_loss = validate_model(model, validation_dataloader, criterion, device)
            if writer is not None:
                writer.add_scalar('Val/Loss', val_loss, epoch + 1)
            print(f"Validation Loss: {val_loss:.4f}")
        
        # Use validation loss for early stopping if available, otherwise training loss
        current_loss = val_loss if val_loss is not None else avg_loss
        
        # Early stopping and checkpointing
        if current_loss < best_loss:
            best_loss = current_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save checkpoint
            if checkpoint_path is not None:
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': best_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'epochs': epochs,
                        'patience': patience,
                        'grad_clip': grad_clip,
                        'use_amp': use_amp
                    }
                }
                
                try:
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model weights
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Loaded best model with loss: {best_loss:.4f}")
    
    return model


def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: Union[torch.device, str],
    return_predictions: bool = False,
    llm: Optional[torch.nn.Module] = None  # Neuer optionaler Parameter für LLM
) -> Union[float, Dict[str, Any]]:
    """
    Validate the model on a validation dataset.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device for validation
        return_predictions: Whether to return predictions and labels
        llm: Optional language model for embedding extraction
        
    Returns:
        Validation loss or dictionary with detailed results
    """
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
        
    model.eval()
    
    total_loss = 0.0
    total_ethics_loss = 0.0
    total_manip_loss = 0.0
    num_samples = 0
    
    all_ethics_preds = []
    all_ethics_labels = []
    all_manip_preds = []
    all_manip_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            try:
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                if input_ids.dtype != torch.long:
                    input_ids = input_ids.long()
                attention_mask = batch['attention_mask'].to(device)
                if attention_mask.dtype != torch.float32:
                    attention_mask = attention_mask.float()
                
                ethics_label = batch['ethics_label'].to(device)
                manipulation_label = batch['manipulation_label'].to(device)
                
                texts = batch.get('texts', None)
                
                # LLM-Verarbeitung (falls vorhanden)
                if llm is not None:
                    try:
                        # Methode 1: Direkt mit Huggingface-API für Transformer-Modelle
                        if hasattr(llm, 'forward'):
                            outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
                            
                            # DistilBERT und ähnliche Modelle geben ein Dictionary/Objekt zurück
                            if hasattr(outputs, 'last_hidden_state'):
                                hidden_states = outputs.last_hidden_state
                            elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
                                hidden_states = outputs['last_hidden_state']
                            elif isinstance(outputs, tuple) and len(outputs) > 0:
                                hidden_states = outputs[0]  # Fallback für Tuple-Output
                            else:
                                print(f"Unbekannte LLM-Ausgabe vom Typ {type(outputs)}, versuche Fallback...")
                                hidden_states = outputs  # Direkter Fallback als letzter Versuch
                        else:
                            # Veraltete Methode für nicht-standardkonforme Modelle
                            print(f"LLM hat keine forward-Methode, verwende Fallback.")
                            if hasattr(llm, 'model') and hasattr(llm.model, 'transformer'):
                                hidden_states = llm.model.transformer(input_ids)
                            elif hasattr(llm, 'transformer'):
                                hidden_states = llm.transformer(input_ids)
                            else:
                                hidden_states = llm(input_ids)
                                
                            # Bei Bedarf letzte hidden states extrahieren
                            if hasattr(hidden_states, 'last_hidden_state'):
                                hidden_states = hidden_states.last_hidden_state
                            elif isinstance(hidden_states, tuple) and len(hidden_states) > 0:
                                hidden_states = hidden_states[0]
                        
                        # Forward pass mit LLM-Embeddings
                        outputs = model(
                            embeddings=hidden_states,
                            attention_mask=attention_mask,
                            texts=texts
                        )
                    except Exception as e:
                        print(f"Fehler beim Extrahieren der LLM embeddings in Validation: {e}")
                        print(f"LLM Typ: {llm.__class__.__name__}")
                        print(f"Input shape: {input_ids.shape}")
                        raise
                else:
                    # Standard Forward-Pass ohne LLM
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        texts=texts
                    )
                
                # Calculate losses
                ethics_score = outputs['ethics_score']
                manipulation_score = outputs['manipulation_score']
                
                loss_ethics = criterion(ethics_score, ethics_label.view(-1, 1).float())
                loss_manip = criterion(manipulation_score, manipulation_label.view(-1, 1).float())
                loss = loss_ethics + 0.5 * loss_manip
                
                # Track metrics
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_ethics_loss += loss_ethics.item() * batch_size
                total_manip_loss += loss_manip.item() * batch_size
                num_samples += batch_size
                
                # Store predictions if requested
                if return_predictions:
                    all_ethics_preds.extend(ethics_score.cpu().numpy())
                    all_ethics_labels.extend(ethics_label.cpu().numpy())
                    all_manip_preds.extend(manipulation_score.cpu().numpy())
                    all_manip_labels.extend(manipulation_label.cpu().numpy())
                    
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    # Calculate averages
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    avg_ethics_loss = total_ethics_loss / num_samples if num_samples > 0 else float('inf')
    avg_manip_loss = total_manip_loss / num_samples if num_samples > 0 else float('inf')
    
    if return_predictions:
        return {
            'loss': avg_loss,
            'ethics_loss': avg_ethics_loss,
            'manip_loss': avg_manip_loss,
            'ethics_predictions': all_ethics_preds,
            'ethics_labels': all_ethics_labels,
            'manip_predictions': all_manip_preds,
            'manip_labels': all_manip_labels
        }
    
    return avg_loss


def load_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str,
                   device: Union[torch.device, str]) -> Dict[str, Any]:
    """
    Load model and optimizer from checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint information
    """
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'config': checkpoint.get('config', {})
        }
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {}


def calculate_metrics(predictions: List[float], 
                     labels: List[float],
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate classification metrics for binary predictions.
    
    Args:
        predictions: Model predictions
        labels: True labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with calculated metrics
    """
    import numpy as np
    
    # Überprüfung auf leere Eingaben
    if len(predictions) == 0 or len(labels) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mse': 0.0,
            'mae': 0.0
        }
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Binary predictions
    binary_preds = (predictions > threshold).astype(int)
    binary_labels = (labels > threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((binary_preds == 1) & (binary_labels == 1))
    tn = np.sum((binary_preds == 0) & (binary_labels == 0))
    fp = np.sum((binary_preds == 1) & (binary_labels == 0))
    fn = np.sum((binary_preds == 0) & (binary_labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Vermeidung der Division durch Null bei F1
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Regression metrics mit Prüfung auf leere Arrays
    if len(predictions) > 0:
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
    else:
        mse = 0.0
        mae = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'mae': mae
    }
