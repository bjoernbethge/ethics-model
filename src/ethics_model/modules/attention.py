"""
Ethical Attention Mechanisms

Specialized attention mechanisms for processing ethical reasoning,
moral intuition, and narrative framing detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Callable
from .activation import get_activation, ReCA
from torch_geometric.nn import GATConv

class EthicalAttention(nn.Module):
    """
    Ethical attention mechanism that focuses on morally relevant aspects
    of the input while considering different ethical frameworks.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 moral_context_dim: int = 64,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.activation = get_activation(activation)
        
        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Moral context attention modifier
        self.moral_context_proj = nn.Linear(moral_context_dim, d_model)
        self.moral_attention_weight = nn.Parameter(torch.randn(1, n_heads, 1, 1))
        
        # Ethical salience scorer
        self.salience_scorer = nn.Sequential(
            nn.Linear(d_model, moral_context_dim),
            self.activation,
            nn.Linear(moral_context_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                moral_context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                symbolic_constraints: Optional[Callable] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ethical attention to the input.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            moral_context: Optional moral context vector
            mask: Optional attention mask
            symbolic_constraints: Optional callable for symbolic constraints
            
        Returns:
            output: Attended output
            attention_weights: Attention distribution
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add moral context if provided
        if moral_context is not None:
            # Projiziere den moralischen Kontext
            moral_bias = self.moral_context_proj(moral_context)  # [batch_size, d_model]
            
            # Teile den moralischen Kontext für Multi-Head-Attention auf
            moral_bias = moral_bias.view(batch_size, 1, self.n_heads, self.d_k)
            moral_bias = moral_bias.permute(0, 2, 1, 3)  # [batch_size, n_heads, 1, d_k]
            
            # Berechne den moralischen Bias-Term für die Scores
            # Füge eine zusätzliche Dimension für die Sequenzlänge hinzu
            bias_expand = moral_bias.expand(batch_size, self.n_heads, seq_len, self.d_k)
            bias_term = torch.sum(q * bias_expand, dim=-1, keepdim=True) * self.moral_attention_weight
            scores = scores + bias_term
        
        # Apply mask if provided
        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.out_proj(attended)
        
        if symbolic_constraints is not None:
            symbolic_result = symbolic_constraints(output, attention_weights.mean(dim=1))
            if symbolic_result is not None:
                output, attention_weights_out = symbolic_result
                return output, attention_weights_out
        return output, attention_weights.mean(dim=1)  # Average over heads for visualization


class MoralIntuitionAttention(nn.Module):
    """
    Simulates fast, intuitive moral judgments characteristic of
    System 1 thinking in dual-process theories.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_moral_foundations: int = 6,
                 temperature: float = 1.0,
                 activation: str = "gelu"):
        super().__init__()
        
        # Moral foundations: Care, Fairness, Loyalty, Authority, Purity, Liberty
        self.moral_foundation_embeddings = nn.Embedding(n_moral_foundations, d_model)
        self.n_moral_foundations = n_moral_foundations
        
        self.activation = get_activation(activation)
        
        # Quick intuitive response pathway
        self.intuition_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            self.activation,
            nn.Linear(d_model, n_moral_foundations),
            nn.Softmax(dim=-1)
        )
        
        # Emotional response amplifier
        self.emotion_amplifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.temperature = temperature
        
    def forward(self, 
                x: torch.Tensor,
                moral_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input through moral intuition pathway.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            moral_context: Optional moral context
            
        Returns:
            Dictionary containing:
                - intuitive_response: Quick moral judgment (batch_size, n_moral_foundations)
                - emotional_intensity: Emotional response strength (batch_size, seq_len, 1)
                - foundation_activations: Activation of each moral foundation (batch_size, n_moral_foundations)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Expand moral foundations to sequence length
        foundation_ids = torch.arange(
            0, 
            self.n_moral_foundations, 
            device=x.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        foundation_embeds = self.moral_foundation_embeddings(foundation_ids)  # [batch_size, n_moral_foundations, d_model]
        
        # Combine input with foundations
        input_expanded = x.unsqueeze(2).expand(-1, -1, foundation_embeds.size(1), -1)  # [batch_size, seq_len, n_foundations, d_model]
        foundation_expanded = foundation_embeds.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch_size, seq_len, n_foundations, d_model]
        
        combined = torch.cat([input_expanded, foundation_expanded], dim=-1)  # [batch_size, seq_len, n_foundations, 2*d_model]
        
        # Quick intuitive scoring
        intuition_scores = self.intuition_scorer(combined)  # [batch_size, seq_len, n_foundations, 1]
        
        # Compute emotional intensity
        emotional_intensity = self.emotion_amplifier(x)  # [batch_size, seq_len, 1]
        
        # Weight intuitions by emotional intensity
        weighted_intuitions = intuition_scores * emotional_intensity.unsqueeze(-2)  # [batch_size, seq_len, n_foundations]
        
        # Average over sequence to get overall intuitive response - korrigiere die Dimensionen
        intuitive_response = weighted_intuitions.mean(dim=1)  # [batch_size, n_foundations]
        
        # Stelle sicher, dass die Ausgabeform korrekt ist
        if len(intuitive_response.shape) == 3 and intuitive_response.shape[2] == self.n_moral_foundations:
            # Reduziere die überschüssige Dimension
            intuitive_response = intuitive_response.squeeze(-1)
        
        return {
            'intuitive_response': intuitive_response,  # (batch_size, n_moral_foundations)
            'emotional_intensity': emotional_intensity,  # (batch_size, seq_len, 1)
            'foundation_activations': intuitive_response  # (batch_size, n_moral_foundations)
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
                 activation: str = "gelu"):
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
                 activation: str = "gelu"):
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
            'norm': nn.LayerNorm(d_model)
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
    Graph Attention Layer auf Basis von torch_geometric GATConv.
    Kann als Baustein für hybride Attention-Modelle genutzt werden.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, activation: str = "gelu"):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False)
        self.activation = get_activation(activation)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat(x, edge_index)
        return self.activation(x)
