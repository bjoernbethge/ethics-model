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

from .modules.moral import MoralFrameworkEmbedding, MultiFrameworkProcessor, EthicalCrossDomainLayer
from .modules.attention import EthicalAttention, MoralIntuitionAttention, NarrativeFrameAttention, GraphAttentionLayer, DoubleProcessingAttention
from .modules.narrative import NarrativeManipulationDetector, FramingDetector, CognitiveDissonanceLayer, PropagandaDetector, NarrativeGraphLayer
from .modules.activation import get_activation, ReCA
from .model import EthicsModel, create_ethics_model
from .common import collate_with_graphs, GraphBrainParserManager, process_text_to_hypergraph, prepare_graph_data_for_model

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
    'create_ethics_model',
    'get_activation',
    'ReCA',
    'GraphAttentionLayer',
    'DoubleProcessingAttention',
    'PropagandaDetector',
    'NarrativeGraphLayer',
    
    # Common utilities
    'collate_with_graphs',
    'GraphBrainParserManager',
    'process_text_to_hypergraph',
    'prepare_graph_data_for_model',
    
    # API components
    'api_app'
]

__version__ = '0.1.0'
