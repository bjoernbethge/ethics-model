"""
Ethics Model Modules

Modular components for ethical analysis including attention mechanisms,
moral frameworks, narrative analysis, and activation functions.
"""

from .activation import get_activation, ReCA
from .attention import (
    EthicalAttention,
    MoralIntuitionAttention,
    NarrativeFrameAttention,
    DoubleProcessingAttention,
    GraphAttentionLayer
)
from .moral import (
    MoralFrameworkEmbedding,
    EthicalCrossDomainLayer,
    MultiFrameworkProcessor,
    EthicalPrincipleEncoder,
    MoralFrameworkGraphLayer
)
from .narrative import (
    FramingDetector,
    CognitiveDissonanceLayer,
    NarrativeManipulationDetector,
    PropagandaDetector,
    NarrativeGraphLayer
)

__all__ = [
    # Activation functions
    'get_activation',
    'ReCA',
    
    # Attention mechanisms
    'EthicalAttention',
    'MoralIntuitionAttention',
    'NarrativeFrameAttention',
    'DoubleProcessingAttention',
    'GraphAttentionLayer',
    
    # Moral frameworks
    'MoralFrameworkEmbedding',
    'EthicalCrossDomainLayer',
    'MultiFrameworkProcessor',
    'EthicalPrincipleEncoder',
    'MoralFrameworkGraphLayer',
    
    # Narrative analysis
    'FramingDetector',
    'CognitiveDissonanceLayer',
    'NarrativeManipulationDetector',
    'PropagandaDetector',
    'NarrativeGraphLayer',
]
