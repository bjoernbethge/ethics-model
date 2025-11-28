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

from .modules.activation import ReCA, RMSNorm, SwiGLU, get_activation
from .modules.attention import (
    DoubleProcessingAttention,
    EthicalAttention,
    GraphAttentionLayer,
    MoralIntuitionAttention,
    NarrativeFrameAttention,
)
from .modules.gqa import EthicalGQA, GroupedQueryAttention
from .modules.rope import RotaryPositionEmbedding, apply_rope
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

# API imports (optional, only if FastAPI is available)
try:
    from .api import app as api_app
except ImportError:
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

    # Activation and Normalization
    'get_activation',
    'ReCA',
    'RMSNorm',
    'SwiGLU',

    # Attention mechanisms
    'GraphAttentionLayer',
    'DoubleProcessingAttention',
    'GroupedQueryAttention',
    'EthicalGQA',

    # Position encoding
    'RotaryPositionEmbedding',
    'apply_rope',

    # Narrative components
    'PropagandaDetector',
    'NarrativeGraphLayer',

    # API components
    'api_app'
]

__version__ = '0.1.0'
