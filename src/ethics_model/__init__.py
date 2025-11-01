"""
Ethics Model PyTorch Extension

This package provides specialized layers and attention mechanisms for processing
ethical narratives, moral frameworks, and developing models that can detect
and analyze ethical manipulation in text.

Components:
- Moral Framework Embeddings (Deontological, Utilitarian, Virtue Ethics)
- Ethical Attention Mechanisms
- Narrative Manipulation Detection Layers
- Meta-cognitive Processing for Bias Detection
- FastAPI-based Inference API
"""

from .modules.activation import ReCA, get_activation
from .modules.attention import (
    DoubleProcessingAttention,
    EthicalAttention,
    GraphAttentionLayer,
    MoralIntuitionAttention,
    NarrativeFrameAttention,
)
from .modules.gnn import EthicsGNN, EthicsGNNConfig, create_ethics_gnn
from .modules.moral import (
    EthicalCrossDomainLayer,
    MoralFrameworkEmbedding,
    MultiFrameworkProcessor,
)
from .modules.narrative import (
    CognitiveDissonanceLayer,
    FramingDetector,
    NarrativeGraphLayer,
    NarrativeManipulationDetector,
    PropagandaDetector,
)
from .modules.retriever import EthicsModel

# Optional API imports
try:
    from .api import app as api_app
except ImportError:
    # FastAPI may not be installed
    api_app = None

__all__ = [
    # Core components
    'MoralFrameworkEmbedding',
    'MultiFrameworkProcessor',
    'EthicalCrossDomainLayer',
    'EthicalAttention',
    'MoralIntuitionAttention',
    'NarrativeFrameAttention',
    'NarrativeManipulationDetector',
    'FramingDetector',
    'CognitiveDissonanceLayer',
    'EthicsModel',
    'EthicsGNN',
    'EthicsGNNConfig',
    'create_ethics_gnn',
    'get_activation',
    'ReCA',
    'GraphAttentionLayer',
    'DoubleProcessingAttention',
    'PropagandaDetector',
    'NarrativeGraphLayer',
    
    # API components
    'api_app'
]

__version__ = '0.1.0'
