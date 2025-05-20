"""
Ethics Model - Comprehensive Ethical Text Analysis Framework

A modern, modular PyTorch framework for ethical text analysis, manipulation
detection, and narrative understanding using NetworkX and spaCy.
"""

from .model import EnhancedEthicsModel, EthicsModel, create_ethics_model
from .data import (
    MultiTaskDataset,
    GraphEthicsDataset,
    collate_ethics_batch,
    create_data_splits,
    load_from_json,
    save_to_json
)
from .training import train, validate, calculate_metrics
from .graph_reasoning import (
    EthicalRelationExtractor,
    EthicalGNN,
    extract_and_visualize
)

__version__ = "0.2.0"

__all__ = [
    # Models
    'EnhancedEthicsModel',
    'EthicsModel',
    'create_ethics_model',
    
    # Data processing
    'MultiTaskDataset',
    'GraphEthicsDataset',
    'collate_ethics_batch',
    'create_data_splits',
    'load_from_json',
    'save_to_json',
    
    # Training
    'train',
    'validate',
    'calculate_metrics',
    
    # Graph reasoning
    'EthicalRelationExtractor',
    'EthicalGNN',
    'extract_and_visualize',
]
