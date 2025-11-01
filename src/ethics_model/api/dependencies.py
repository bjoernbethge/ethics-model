"""
Dependency injection utilities for the Ethics Model API.

This module provides dependencies for FastAPI routes, including
model loading, tokenizer initialization, and other shared resources.
"""

import logging
import os
from functools import lru_cache
from typing import Any

import torch
from fastapi import Depends
from transformers import AutoModel, AutoTokenizer

from ..modules.retriever import EthicsModel
from .settings import Settings, get_settings

logger = logging.getLogger("ethics_model.api.dependencies")


@lru_cache
def get_model(settings: Settings = Depends(get_settings)) -> Any:
    """
    Load and cache the Ethics Model.
    
    Args:
        settings: Application settings
        
    Returns:
        Loaded Ethics Model instance
    """
    logger.info(f"Loading Ethics Model (device: {settings.device})")
    
    # Create model (GRetriever-based)
    model = EthicsModel()
    
    # Load checkpoint if available
    if settings.checkpoint_path and os.path.exists(settings.checkpoint_path):
        logger.info(f"Loading model checkpoint from {settings.checkpoint_path}")
        try:
            state_dict = torch.load(settings.checkpoint_path, map_location=settings.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}")
            logger.warning("Proceeding with newly initialized model")
    
    # Move model to device
    model = model.to(settings.device)
    model.eval()  # Set to evaluation mode
    
    return model


@lru_cache
def get_tokenizer(settings: Settings = Depends(get_settings)):
    """
    Load and cache the tokenizer.
    
    Args:
        settings: Application settings
        
    Returns:
        Loaded tokenizer instance
    """
    logger.info(f"Loading tokenizer {settings.tokenizer_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        settings.tokenizer_name,
        cache_dir=settings.cache_dir
    )
    
    return tokenizer


@lru_cache
def get_llm(settings: Settings = Depends(get_settings)):
    """
    Load and cache the language model for embeddings.
    
    Args:
        settings: Application settings
        
    Returns:
        Loaded language model instance
    """
    logger.info(f"Loading LLM {settings.llm_name} (device: {settings.device})")
    
    llm = AutoModel.from_pretrained(
        settings.llm_name,
        cache_dir=settings.cache_dir
    )
    
    llm = llm.to(settings.device)
    llm.eval()  # Set to evaluation mode
    
    return llm
