"""
Training and visualization endpoints for the Ethics Model API.

This module adds routes for model training, visualization, and evaluation
to the main Ethics Model API.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..data import EthicsDataset
from ..modules.retriever import EthicsModel
from ..training import train
from .dependencies import get_llm, get_model, get_tokenizer
from .settings import Settings, get_settings

# Configure logging
logger = logging.getLogger("ethics_model.api.training")

# Create router
router = APIRouter(tags=["Training"])

# Training related models
class TrainingInput(BaseModel):
    """Input model for training requests."""
    train_texts: List[str] = Field(..., 
                                 description="Training texts", 
                                 min_items=10, 
                                 max_items=1000)
    ethics_labels: List[float] = Field(..., 
                                     description="Ethics scores for training texts")
    manipulation_labels: List[float] = Field(..., 
                                           description="Manipulation scores for training texts")
    validation_split: float = Field(0.2, 
                                   description="Validation split ratio",
                                   ge=0.0,
                                   le=0.5)
    epochs: int = Field(5, 
                       description="Number of training epochs",
                       ge=1,
                       le=100)
    batch_size: int = Field(16, 
                           description="Training batch size",
                           ge=1)
    learning_rate: float = Field(1e-4, 
                               description="Learning rate",
                               gt=0.0)
    augment: bool = Field(False,
                         description="Whether to use data augmentation")
    checkpoint_name: Optional[str] = Field(None,
                                         description="Name to save the checkpoint as")
    model_config: Optional[Dict[str, Any]] = Field(None,
                                                description="Optional model configuration parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "train_texts": ["Text 1", "Text 2", "..."],
                "ethics_labels": [0.8, 0.2, "..."],
                "manipulation_labels": [0.1, 0.9, "..."],
                "validation_split": 0.2,
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "checkpoint_name": "my_model_checkpoint"
            }
        }


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    task_id: str = Field(..., description="Training task ID")
    status: str = Field(..., description="Training status")
    progress: Optional[float] = Field(None, description="Training progress (0-1)")
    current_epoch: Optional[int] = Field(None, description="Current epoch")
    total_epochs: Optional[int] = Field(None, description="Total epochs")
    train_loss: Optional[float] = Field(None, description="Current training loss")
    val_loss: Optional[float] = Field(None, description="Current validation loss")
    eta_seconds: Optional[int] = Field(None, description="Estimated time remaining in seconds")


class TrainingResultResponse(BaseModel):
    """Response model for training results."""
    task_id: str = Field(..., description="Training task ID")
    status: str = Field(..., description="Training status")
    epochs_completed: int = Field(..., description="Number of epochs completed")
    train_loss_history: List[float] = Field(..., description="Training loss history")
    val_loss_history: Optional[List[float]] = Field(None, description="Validation loss history")
    train_ethics_accuracy: float = Field(..., description="Final training ethics accuracy")
    train_manipulation_accuracy: float = Field(..., description="Final training manipulation accuracy")
    val_ethics_accuracy: Optional[float] = Field(None, description="Final validation ethics accuracy")
    val_manipulation_accuracy: Optional[float] = Field(None, description="Final validation manipulation accuracy")
    checkpoint_path: Optional[str] = Field(None, description="Path to saved model checkpoint")
    training_duration_seconds: float = Field(..., description="Total training duration in seconds")


# In-memory training task store
training_tasks = {}


@router.post("/train", response_model=TrainingStatusResponse)
async def train_model(
    input_data: TrainingInput,
    background_tasks: BackgroundTasks,
    model: EthicsModel = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
    llm = Depends(get_llm),
    settings: Settings = Depends(get_settings)
):
    """
    Start asynchronous training of the ethics model.
    
    This endpoint initiates a training process for the model using the provided
    texts and labels. The training runs asynchronously, and the status can be
    checked using the returned task ID.
    
    Returns a task ID that can be used to check training status and retrieve results.
    """
    # Validation checks
    if len(input_data.train_texts) != len(input_data.ethics_labels) or len(input_data.train_texts) != len(input_data.manipulation_labels):
        raise HTTPException(
            status_code=400, 
            detail="Number of texts must match number of ethics and manipulation labels"
        )
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Create checkpoint path if needed
    checkpoint_path = None
    if input_data.checkpoint_name:
        checkpoint_dir = os.path.join(settings.cache_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{input_data.checkpoint_name}.pt")
    
    # Initialize task status
    training_tasks[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": input_data.epochs,
        "train_loss": None,
        "val_loss": None,
        "train_loss_history": [],
        "val_loss_history": [],
        "start_time": None,
        "end_time": None,
        "results": None
    }
    
    # Start training task
    background_tasks.add_task(
        _run_training_task,
        task_id=task_id,
        train_texts=input_data.train_texts,
        ethics_labels=input_data.ethics_labels,
        manipulation_labels=input_data.manipulation_labels,
        validation_split=input_data.validation_split,
        epochs=input_data.epochs,
        batch_size=input_data.batch_size,
        learning_rate=input_data.learning_rate,
        augment=input_data.augment,
        checkpoint_path=checkpoint_path,
        model_config=input_data.model_config,
        model=model,
        tokenizer=tokenizer,
        llm=llm,
        settings=settings
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": input_data.epochs,
        "train_loss": None,
        "val_loss": None,
        "eta_seconds": None
    }


@router.get("/train/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(task_id: str):
    """
    Check the status of an training task.
    
    Returns the current training status including progress, loss values,
    and estimated time remaining.
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    task = training_tasks[task_id]
    
    # Calculate ETA
    eta_seconds = None
    if task["status"] == "training" and task["start_time"] and task["progress"] > 0:
        elapsed = time.time() - task["start_time"]
        if task["progress"] < 1.0:
            eta_seconds = int((elapsed / task["progress"]) * (1.0 - task["progress"]))
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "current_epoch": task["current_epoch"],
        "total_epochs": task["total_epochs"],
        "train_loss": task["train_loss"],
        "val_loss": task["val_loss"],
        "eta_seconds": eta_seconds
    }


@router.get("/train/{task_id}/result", response_model=TrainingResultResponse)
async def get_training_result(task_id: str):
    """
    Get the results of a completed training task.
    
    Returns detailed training results including loss history, accuracy metrics,
    and path to the saved model checkpoint if available.
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    task = training_tasks[task_id]
    
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Training task is not completed (status: {task['status']})"
        )
    
    if task["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Training task failed: {task.get('error', 'Unknown error')}"
        )
    
    # Calculate training duration
    duration = None
    if task["start_time"] and task["end_time"]:
        duration = task["end_time"] - task["start_time"]
    
    results = task.get("results", {})
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "epochs_completed": task["current_epoch"],
        "train_loss_history": task["train_loss_history"],
        "val_loss_history": task["val_loss_history"],
        "train_ethics_accuracy": results.get("train_ethics_accuracy", 0.0),
        "train_manipulation_accuracy": results.get("train_manipulation_accuracy", 0.0),
        "val_ethics_accuracy": results.get("val_ethics_accuracy"),
        "val_manipulation_accuracy": results.get("val_manipulation_accuracy"),
        "checkpoint_path": results.get("checkpoint_path"),
        "training_duration_seconds": duration or 0.0
    }


# Helper functions
def _run_training_task(
    task_id: str,
    train_texts: List[str],
    ethics_labels: List[float],
    manipulation_labels: List[float],
    validation_split: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    augment: bool,
    checkpoint_path: Optional[str],
    model_config: Optional[Dict[str, Any]],
    model: EthicsModel,
    tokenizer,
    llm,
    settings: Settings
):
    """Background task for model training."""
    import torch.utils.data as data_utils
    from sklearn.model_selection import train_test_split
    from torch.utils.tensorboard import SummaryWriter

    from ..data import EthicsDataset

    try:
        # Update task status
        training_tasks[task_id]["status"] = "preparing"
        training_tasks[task_id]["start_time"] = time.time()
        
        # Initialize augmentation if needed
        synonym_augment_fn = None
        if augment:
            try:
                import nlpaug.augmenter.word as naw
                aug = naw.SynonymAug(aug_src='wordnet')
                
                def synonym_augment(text):
                    try:
                        return aug.augment(text)
                    except Exception:
                        return text
                
                synonym_augment_fn = synonym_augment
                logger.info("Data augmentation enabled with synonym replacement")
            except ImportError:
                logger.warning("nlpaug package not found. Data augmentation disabled.")
        
        # Create a copy of the model for training (or create a new one if model_config is provided)
        if model_config:
            logger.info(f"Creating new model with provided configuration")
            training_model = EthicsModel(
                input_dim=model_config.get('input_dim', llm.config.hidden_size if hasattr(llm, 'config') else 768),
                d_model=model_config.get('d_model', llm.config.hidden_size if hasattr(llm, 'config') else 768),
                n_layers=model_config.get('n_layers', 2),
                n_heads=model_config.get('n_heads', 8),
                vocab_size=model_config.get('vocab_size', tokenizer.vocab_size),
                max_seq_length=model_config.get('max_seq_length', settings.max_sequence_length),
                activation=model_config.get('activation', "gelu"),
                use_gnn=model_config.get('use_gnn', False)
            ).to(settings.device)
        else:
            training_model = type(model)(
                input_dim=model.embedding.embedding_dim,
                d_model=model.embedding.embedding_dim,
                n_layers=len(model.transformer_layers),
                n_heads=model.ethical_attention.n_heads,
                vocab_size=model.embedding.num_embeddings,
                max_seq_length=settings.max_sequence_length,
                activation=settings.model_config.get("activation", "gelu"),
                use_gnn=settings.model_config.get("use_gnn", False)
            ).to(settings.device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(training_model.parameters(), lr=learning_rate)
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Create TensorBoard writer
        log_dir = os.path.join(settings.cache_dir, "logs", f"training_{task_id}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        # Create dataset using EthicsDataset
        logger.info(f"Creating dataset with {len(train_texts)} texts")
        
        # Split data if validation split > 0
        if validation_split > 0:
            # Split indices
            train_indices, val_indices = train_test_split(
                range(len(train_texts)), test_size=validation_split, random_state=42
            )
            
            train_data_texts = [train_texts[i] for i in train_indices]
            train_data_ethics = [ethics_labels[i] for i in train_indices]
            train_data_manip = [manipulation_labels[i] for i in train_indices]
            
            val_data_texts = [train_texts[i] for i in val_indices]
            val_data_ethics = [ethics_labels[i] for i in val_indices]
            val_data_manip = [manipulation_labels[i] for i in val_indices]
            
            # Create datasets
            train_dataset = EthicsDataset(
                train_data_texts, 
                train_data_ethics, 
                train_data_manip, 
                tokenizer, 
                max_length=settings.max_sequence_length,
                augment=augment, 
                synonym_augment=synonym_augment_fn
            )
            
            val_dataset = EthicsDataset(
                val_data_texts, 
                val_data_ethics, 
                val_data_manip, 
                tokenizer, 
                max_length=settings.max_sequence_length,
                augment=False
            )
            
            # Create data loaders
            train_loader = data_utils.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = data_utils.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            # Use all data for training
            train_dataset = EthicsDataset(
                train_texts, 
                ethics_labels, 
                manipulation_labels, 
                tokenizer, 
                max_length=settings.max_sequence_length,
                augment=augment, 
                synonym_augment=synonym_augment_fn
            )
            
            train_loader = data_utils.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = None
        
        # Update task status
        training_tasks[task_id]["status"] = "training"
        
        # Import the training function
        from ...training import train as train_function
        
        # Define symbolic constraints - None for standard training
        symbolic_constraints = None
        
        # Train the model
        try:
            # Use the training function from ethics_model.training
            train_function(
                model=training_model,
                llm=llm,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                writer=writer,
                device=settings.device,
                epochs=epochs,
                patience=3,  # Early stopping patience
                grad_clip=1.0,  # Gradient clipping
                symbolic_constraints=symbolic_constraints,
                checkpoint_path=checkpoint_path
            )
            
            # Mark as best model (the train function should already save the best model)
            best_model_state = training_model.state_dict()
            
        except Exception as e:
            logger.exception(f"Error during training: {str(e)}")
            training_tasks[task_id]["status"] = "failed"
            training_tasks[task_id]["error"] = str(e)
            training_tasks[task_id]["end_time"] = time.time()
            writer.close()
            return
        
        # Custom training loop (adapted from training.py)
        for epoch in range(epochs):
            # Update task status
            training_tasks[task_id]["current_epoch"] = epoch + 1
            training_tasks[task_id]["progress"] = (epoch + 1) / epochs
            
            # Training step
            training_model.train()
            total_loss = 0.0
            total_samples = 0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(settings.device)
                attention_mask = batch["attention_mask"].to(settings.device)
                ethics_label = batch["ethics_label"].to(settings.device)
                manipulation_label = batch["manipulation_label"].to(settings.device)
                
                # Get LLM embeddings
                with torch.no_grad():
                    llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                    hidden_states = llm_outputs.last_hidden_state
                
                # Forward pass
                optimizer.zero_grad()
                outputs = training_model(embeddings=hidden_states, attention_mask=attention_mask)
                
                # Calculate loss
                ethics_score = outputs["ethics_score"]
                manipulation_score = outputs["manipulation_score"]
                loss_ethics = criterion(ethics_score, ethics_label)
                loss_manip = criterion(manipulation_score, manipulation_label)
                loss = loss_ethics + 0.5 * loss_manip
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
            
            # Calculate training loss
            train_loss = total_loss / total_samples
            training_tasks[task_id]["train_loss"] = train_loss
            training_tasks[task_id]["train_loss_history"].append(train_loss)
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            
            # Validation step (if validation data is available)
            val_loss = None
            if val_loader:
                training_model.eval()
                total_val_loss = 0.0
                total_val_samples = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(settings.device)
                        attention_mask = batch["attention_mask"].to(settings.device)
                        ethics_label = batch["ethics_label"].to(settings.device)
                        manipulation_label = batch["manipulation_label"].to(settings.device)
                        
                        # Get LLM embeddings
                        llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                        hidden_states = llm_outputs.last_hidden_state
                        
                        # Forward pass
                        outputs = training_model(embeddings=hidden_states, attention_mask=attention_mask)
                        
                        # Calculate loss
                        ethics_score = outputs["ethics_score"]
                        manipulation_score = outputs["manipulation_score"]
                        loss_ethics = criterion(ethics_score, ethics_label)
                        loss_manip = criterion(manipulation_score, manipulation_label)
                        loss = loss_ethics + 0.5 * loss_manip
                        
                        # Update statistics
                        total_val_loss += loss.item() * input_ids.size(0)
                        total_val_samples += input_ids.size(0)
                
                # Calculate validation loss
                val_loss = total_val_loss / total_val_samples
                training_tasks[task_id]["val_loss"] = val_loss
                training_tasks[task_id]["val_loss_history"].append(val_loss)
                writer.add_scalar("Loss/validation", val_loss, epoch + 1)
                
                # Check if this is the best model so far
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = training_model.state_dict()
            else:
                # If no validation data, use training loss
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model_state = training_model.state_dict()
        
        # Calculate final metrics
        training_model.eval()
        
        # Training metrics
        train_ethics_correct = 0
        train_manip_correct = 0
        train_total = 0
        
        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch["input_ids"].to(settings.device)
                attention_mask = batch["attention_mask"].to(settings.device)
                ethics_label = batch["ethics_label"].to(settings.device)
                manipulation_label = batch["manipulation_label"].to(settings.device)
                
                # Get LLM embeddings
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                hidden_states = llm_outputs.last_hidden_state
                
                # Forward pass
                outputs = training_model(embeddings=hidden_states, attention_mask=attention_mask)
                
                # Calculate accuracy (using a threshold of 0.5 for binary classification)
                ethics_pred = (outputs["ethics_score"] > 0.5).float()
                manipulation_pred = (outputs["manipulation_score"] > 0.5).float()
                
                ethics_correct = (ethics_pred == (ethics_label > 0.5).float()).sum().item()
                manip_correct = (manipulation_pred == (manipulation_label > 0.5).float()).sum().item()
                
                train_ethics_correct += ethics_correct
                train_manip_correct += manip_correct
                train_total += input_ids.size(0)
        
        train_ethics_accuracy = train_ethics_correct / train_total
        train_manip_accuracy = train_manip_correct / train_total
        
        # Validation metrics (if available)
        val_ethics_accuracy = None
        val_manip_accuracy = None
        
        if val_loader:
            val_ethics_correct = 0
            val_manip_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(settings.device)
                    attention_mask = batch["attention_mask"].to(settings.device)
                    ethics_label = batch["ethics_label"].to(settings.device)
                    manipulation_label = batch["manipulation_label"].to(settings.device)
                    
                    # Get LLM embeddings
                    llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                    hidden_states = llm_outputs.last_hidden_state
                    
                    # Forward pass
                    outputs = training_model(embeddings=hidden_states, attention_mask=attention_mask)
                    
                    # Calculate accuracy (using a threshold of 0.5 for binary classification)
                    ethics_pred = (outputs["ethics_score"] > 0.5).float()
                    manipulation_pred = (outputs["manipulation_score"] > 0.5).float()
                    
                    ethics_correct = (ethics_pred == (ethics_label > 0.5).float()).sum().item()
                    manip_correct = (manipulation_pred == (manipulation_label > 0.5).float()).sum().item()
                    
                    val_ethics_correct += ethics_correct
                    val_manip_correct += manip_correct
                    val_total += input_ids.size(0)
            
            val_ethics_accuracy = val_ethics_correct / val_total
            val_manip_accuracy = val_manip_correct / val_total
        
        # Save best model checkpoint
        if checkpoint_path and best_model_state:
            try:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_model_state, checkpoint_path)
                logger.info(f"Saved model checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving model checkpoint: {str(e)}")
                checkpoint_path = None
        
        # Update task status and results
        training_tasks[task_id]["status"] = "completed"
        training_tasks[task_id]["end_time"] = time.time()
        training_tasks[task_id]["progress"] = 1.0
        training_tasks[task_id]["results"] = {
            "train_ethics_accuracy": train_ethics_accuracy,
            "train_manipulation_accuracy": train_manip_accuracy,
            "val_ethics_accuracy": val_ethics_accuracy,
            "val_manipulation_accuracy": val_manip_accuracy,
            "checkpoint_path": checkpoint_path
        }
        
        # Close writer
        writer.close()
        
    except Exception as e:
        logger.exception(f"Error in training task: {str(e)}")
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["error"] = str(e)
        training_tasks[task_id]["end_time"] = time.time()
