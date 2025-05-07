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
"""

from .moral import MoralFrameworkEmbedding, MultiFrameworkProcessor, EthicalCrossDomainLayer
from .attention import EthicalAttention, MoralIntuitionAttention, NarrativeFrameAttention, GraphAttentionLayer, DoubleProcessingAttention
from .narrative import NarrativeManipulationDetector, FramingDetector, CognitiveDissonanceLayer, PropagandaDetector, NarrativeGraphLayer
from .model import EthicsModel
from .activation import get_activation, ReCA

__all__ = [
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
    'get_activation',
    'ReCA',
    'GraphAttentionLayer',
    'DoubleProcessingAttention',
    'PropagandaDetector',
    'NarrativeGraphLayer'
]

__version__ = '0.1.0'
