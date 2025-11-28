"""
Settings management for the Ethics Model API.

This module provides configuration settings for the API server, model loading,
and inference parameters, with support for environment variables.
"""

import os
from functools import lru_cache
from typing import Optional

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model settings
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    tokenizer_name: str = "gpt2"
    llm_name: str = "gpt2"
    ethics_model_config: dict = {
        "input_dim": 512,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "vocab_size": 50257,  # Default for GPT-2
        "max_seq_length": 512,
        "activation": "gelu",
        "use_gnn": False
    }

    # Processing settings
    max_sequence_length: int = 512
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Storage settings
    cache_dir: str = "./cache"

    model_config = {
        "env_prefix": "ETHICS_API_",
        "env_file": ".env"
    }


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application settings object
    """
    return Settings()
