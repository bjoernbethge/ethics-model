"""
Narrative Detection and Manipulation Analysis

Components for detecting manipulation in narratives, cognitive dissonance,
and framing techniques in text.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric import EdgeIndex
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing

from .activation import get_activation


class FramingDetector(MessagePassing):
    """
    Graph-native framing detector using GCN for frame consistency.
    """
    def __init__(
        self,
        d_model: int,
        activation: str = "silu",
    ):
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Frame type encoders
        self.frame_encoders = nn.ModuleDict({
            'loss_gain': nn.Linear(d_model, 2),
            'moral': nn.Linear(d_model, 5),
            'episodic_thematic': nn.Linear(d_model, 2),
            'problem_solution': nn.Linear(d_model, 2),
            'conflict_consensus': nn.Linear(d_model, 2),
            'urgency_deliberation': nn.Linear(d_model, 2)
        })
        
        # Frame consistency via GCN
        self.consistency_conv = GCNConv(d_model, d_model)
        
        # Framing strength detector
        self.strength_detector = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex,
        symbolic_constraints: Optional[Callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect framing techniques in graph.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            Dictionary with frame scores and consistency
        """
        # Detect frame types
        frame_scores = {}
        for frame_type, encoder in self.frame_encoders.items():
            scores = encoder(x)
            frame_scores[frame_type] = torch.softmax(scores, dim=-1)
        
        # Check consistency via graph convolution
        consistency_features = self.consistency_conv(x, edge_index)
        consistency_features = self.activation(consistency_features)
        
        # Consistency score (similarity between connected nodes)
        consistency_score = self.strength_detector(consistency_features)
        
        # Framing strength
        framing_strength = self.strength_detector(x)
        
        result = {
            'frame_scores': frame_scores,
            'framing_strength': framing_strength,
            'consistency_score': consistency_score
        }
        if symbolic_constraints is not None:
            symbolic_result = symbolic_constraints(result)
            if symbolic_result is not None:
                return symbolic_result
        return result


class CognitiveDissonanceLayer(MessagePassing):
    """
    Graph-native cognitive dissonance detector.
    Uses GCN to detect value conflicts between connected nodes.
    """
    def __init__(
        self,
        d_model: int,
        n_moral_values: int = 8,
        activation: str = "silu",
    ):
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Value encoder
        self.value_encoder = nn.Linear(d_model, n_moral_values)
        
        # Contradiction detector via GCN
        self.contradiction_conv = GCNConv(d_model, d_model)
        
        # Dissonance scorer
        self.dissonance_scorer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Resolution predictor
        self.resolution_predictor = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex
    ) -> Dict[str, torch.Tensor]:
        """
        Detect cognitive dissonance in graph.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            Dictionary with dissonance scores and resolution
        """
        # Encode moral values
        value_activations = torch.softmax(self.value_encoder(x), dim=-1)
        
        # Detect contradictions via graph convolution
        contradiction_features = self.contradiction_conv(x, edge_index)
        contradiction_features = self.activation(contradiction_features)
        
        # Dissonance score
        dissonance_score = self.dissonance_scorer(contradiction_features)
        
        # Resolution strategy
        resolution_strategy = self.resolution_predictor(contradiction_features)
        
        return {
            'dissonance_score': dissonance_score,
            'resolution_strategy': resolution_strategy,
            'value_activations': value_activations
        }


class NarrativeManipulationDetector(MessagePassing):
    """
    Comprehensive detector for various manipulation techniques in narratives
    including emotional appeals, logical fallacies, and framing biases.
    """

    def __init__(
        self,
        d_model: int,
        activation: str = "silu",
    ):
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Manipulation detection via GCN
        self.manipulation_conv = GCNConv(d_model, d_model)
        
        # Technique classifiers
        self.technique_classifiers = nn.ModuleDict({
            'emotional_appeal': nn.Linear(d_model, 1),
            'false_dichotomy': nn.Linear(d_model, 1),
            'appeal_to_authority': nn.Linear(d_model, 1),
            'bandwagon': nn.Linear(d_model, 1),
            'loaded_language': nn.Linear(d_model, 1),
            'cherry_picking': nn.Linear(d_model, 1),
            'straw_man': nn.Linear(d_model, 1),
            'slippery_slope': nn.Linear(d_model, 1)
        })
        
        # Aggregate score
        self.aggregator = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex
    ) -> Dict[str, torch.Tensor]:
        """
        Detect manipulation techniques in graph.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            Dictionary with manipulation scores
        """
        # Process via graph convolution
        manipulation_features = self.manipulation_conv(x, edge_index)
        manipulation_features = self.activation(manipulation_features)
        
        # Detect each technique
        technique_scores = {}
        for technique, classifier in self.technique_classifiers.items():
            score = torch.sigmoid(classifier(manipulation_features))
            technique_scores[technique] = score
        
        # Aggregate score
        aggregate_score = self.aggregator(manipulation_features)
        
        return {
            'technique_scores': technique_scores,
            'aggregate_score': aggregate_score,
            'manipulation_features': manipulation_features
        }


class PropagandaDetector(MessagePassing):
    """
    Graph-native propaganda detector using GATv2Conv.
    """
    def __init__(
        self,
        d_model: int,
        activation: str = "silu",
    ):
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Propaganda detection via GATv2
        self.propaganda_gat = GATv2Conv(d_model, d_model, heads=4, concat=False)
        
        # Intensity scorer
        self.intensity_scorer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Credibility estimator
        self.credibility_estimator = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex
    ) -> Dict[str, torch.Tensor]:
        """
        Detect propaganda in graph.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            Dictionary with propaganda scores
        """
        # Propaganda detection
        propaganda_features = self.propaganda_gat(x, edge_index)
        propaganda_features = self.activation(propaganda_features)
        
        # Intensity score
        intensity_score = self.intensity_scorer(propaganda_features)
        
        # Credibility score
        credibility_score = self.credibility_estimator(propaganda_features)
        
        return {
            'intensity_score': intensity_score,
            'credibility_score': credibility_score,
            'inverse_credibility': 1 - credibility_score
        }


class NarrativeGraphLayer(nn.Module):
    """
    Modern GNN layer for narrative graph structures using EdgeIndex optimization.
    Processes narrative framing, manipulation patterns, and discourse structure.
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = "silu"):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: EdgeIndex) -> torch.Tensor:
        """Forward pass requiring EdgeIndex (not COO tensor)."""
        x = self.gcn(x, edge_index)
        return self.activation(x)
