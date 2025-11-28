"""
Ethical Attention Mechanisms

Specialized attention mechanisms for processing ethical reasoning,
moral intuition, and narrative framing detection.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric import EdgeIndex
from torch_geometric.nn import GATv2Conv, MessagePassing

from .activation import RMSNorm, get_activation


class EthicalAttention(MessagePassing):
    """
    Graph-native ethical attention using GATv2Conv for moral relevance.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Use GATv2Conv for ethical attention
        self.ethical_gat = GATv2Conv(
            d_model, d_model, heads=n_heads,
            dropout=dropout, concat=False
        )
        
        # Salience scorer
        self.salience_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            get_activation(activation),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex,
        symbolic_constraints: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ethical attention on graph.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            output: Attended node features
            attention_weights: Attention distribution (salience scores)
        """
        # Ethical attention via GAT
        attended = self.ethical_gat(x, edge_index)
        attended = self.activation(attended)
        attended = self.dropout(attended)
        
        # Compute salience (attention weights)
        salience = self.salience_scorer(attended)
        attention_weights = salience.squeeze(-1)  # (num_nodes,)
        
        if symbolic_constraints is not None:
            symbolic_result = symbolic_constraints(attended, attention_weights)
            if symbolic_result is not None:
                attended, attention_weights = symbolic_result
                return attended, attention_weights
        
        return attended, attention_weights


class MoralIntuitionAttention(MessagePassing):
    """
    Graph-native moral intuition using GCN for fast System 1 judgments.
    """
    def __init__(
        self,
        d_model: int,
        n_moral_foundations: int = 6,
        activation: str = "silu",
    ):
        from torch_geometric.nn import GCNConv
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Moral foundation embeddings
        self.moral_foundation_embeddings = nn.Embedding(
            n_moral_foundations, d_model
        )
        
        # Quick intuition via GCN
        self.intuition_conv = GCNConv(d_model, d_model)
        
        # Foundation classifier
        self.foundation_classifier = nn.Sequential(
            nn.Linear(d_model, n_moral_foundations),
            nn.Softmax(dim=-1)
        )
        
        # Emotional intensity
        self.emotion_amplifier = nn.Sequential(
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
        Process graph through moral intuition pathway.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            Dictionary with intuitive responses and emotional intensity
        """
        # Quick intuition via graph convolution
        intuitive_output = self.intuition_conv(x, edge_index)
        intuitive_output = self.activation(intuitive_output)
        
        # Classify moral foundations
        foundation_activations = self.foundation_classifier(intuitive_output)
        
        # Compute emotional intensity
        emotional_intensity = self.emotion_amplifier(intuitive_output)
        
        return {
            'intuitive_response': intuitive_output,
            'emotional_intensity': emotional_intensity,
            'foundation_activations': foundation_activations
        }


class NarrativeFrameAttention(nn.Module):
    """
    Attention mechanism specialized for detecting narrative framing
    and manipulative communication patterns.
    """
    
    def __init__(self,
                 d_model: int,
                 n_frame_types: int = 5,
                 manipulation_threshold: float = 0.7,
                 activation: str = "silu"):
        super().__init__()
        
        self.activation = get_activation(activation)
        
        # Frame type classifiers
        self.frame_types = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                self.activation,
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for _ in range(n_frame_types)
        ])
        
        # Frame transition detector
        self.transition_detector = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Manipulation pattern recognition
        self.manipulation_detector = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            self.activation,
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.manipulation_threshold = manipulation_threshold
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect narrative frames and manipulation patterns.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - frame_scores: Scores for each frame type
                - manipulation_scores: Manipulation detection scores
                - frame_transitions: Frame transition predictions
        """
        # Detect frame types
        frame_scores = []
        for frame_classifier in self.frame_types:
            score = frame_classifier(x)
            frame_scores.append(score)
        
        frame_scores = torch.cat(frame_scores, dim=-1)
        
        # Detect frame transitions
        transition_outputs, _ = self.transition_detector(x)
        
        # Combine for manipulation detection
        combined_features = torch.cat([x, transition_outputs], dim=-1)
        manipulation_scores = self.manipulation_detector(combined_features)
        
        # Identify manipulative segments
        manipulative_segments = (manipulation_scores > self.manipulation_threshold).float()
        
        return {
            'frame_scores': frame_scores,
            'manipulation_scores': manipulation_scores,
            'manipulative_segments': manipulative_segments,
            'frame_transitions': transition_outputs
        }


class DoubleProcessingAttention(nn.Module):
    """
    Implements dual-process theory for ethical reasoning with both
    fast (System 1) and slow (System 2) processing pathways.
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 system1_latency: int = 2,
                 system2_latency: int = 8,
                 activation: str = "silu"):
        super().__init__()
        
        self.activation = get_activation(activation)
        
        # System 1: Fast, automatic processing
        self.system1 = nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, n_heads),
            'mlp': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                self.activation,
                nn.Linear(d_model // 2, d_model)
            )
        })
        
        # System 2: Slow, deliberative processing
        self.system2 = nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, n_heads),
            'mlp': nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                self.activation,
                nn.Linear(d_model * 2, d_model)
            ),
            'norm': RMSNorm(d_model)
        })
        
        # Conflict resolution
        self.conflict_resolver = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
        self.system1_latency = system1_latency
        self.system2_latency = system2_latency
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process input through dual-system architecture.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            final_output: Resolved output
            system_outputs: Outputs from each system
        """
        # System 1 processing (fast)
        system1_attention, _ = self.system1['attention'](x, x, x)
        system1_output = self.system1['mlp'](system1_attention)
        
        # System 2 processing (slow)
        system2_attention, _ = self.system2['attention'](x, x, x)
        system2_mlp = self.system2['mlp'](system2_attention)
        system2_output = self.system2['norm'](system2_mlp + system2_attention)
        
        # Resolve conflicts between systems
        combined = torch.cat([system1_output, system2_output], dim=-1)
        resolution_weights = self.conflict_resolver(combined)
        
        # Weight systems by resolution scores
        final_output = (
            system1_output * resolution_weights[..., 0:1] +
            system2_output * resolution_weights[..., 1:2]
        )
        
        system_outputs = {
            'system1_output': system1_output,
            'system2_output': system2_output,
            'resolution_weights': resolution_weights
        }
        
        return final_output, system_outputs


class GraphAttentionLayer(nn.Module):
    """
    Modern Graph Attention Layer using GATv2Conv with EdgeIndex optimization.
    GATv2Conv provides dynamic attention (better than static GAT).
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 activation: str = "silu", dropout: float = 0.1):
        super().__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads=heads,
                             concat=False, dropout=dropout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: EdgeIndex) -> torch.Tensor:
        """Forward pass requiring EdgeIndex (not COO tensor)."""
        x = self.gat(x, edge_index)
        return self.activation(x)
