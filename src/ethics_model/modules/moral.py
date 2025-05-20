"""
Moral Framework Components

Implements specialized neural network layers for processing different ethical
frameworks including:
- Deontological ethics (duty-based)
- Utilitarian ethics (consequentialist)
- Virtue ethics (character-based)
- Narrative ethics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
from .activation import get_activation, ReCA
from torch_geometric.nn import GCNConv


class MoralFrameworkEmbedding(nn.Module):
    """
    Embedding layer that encodes different moral frameworks with learned 
    representations. Maps ethical concepts to multi-framework embeddings.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 framework_dim: int = 256,
                 n_frameworks: int = 5):
        super().__init__()
        
        # Each moral framework gets its own embedding subspace
        self.framework_embeddings = nn.ModuleDict({
            'deontological': nn.Linear(input_dim, framework_dim),
            'utilitarian': nn.Linear(input_dim, framework_dim),
            'virtue': nn.Linear(input_dim, framework_dim),
            'narrative': nn.Linear(input_dim, framework_dim),
            'care': nn.Linear(input_dim, framework_dim)
        })
        
        # Framework activation/selection
        self.framework_gate = nn.Linear(input_dim, len(self.framework_embeddings))
        
        # Combine frameworks
        self.framework_combiner = nn.Linear(
            framework_dim * len(self.framework_embeddings), 
            input_dim
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Projects input through different moral framework lenses.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            combined: Combined framework representation
            framework_outputs: Dictionary of individual framework outputs
        """
        # Calculate framework selection weights
        framework_weights = torch.softmax(self.framework_gate(x), dim=-1)
        
        framework_outputs = {}
        weighted_outputs = []
        
        for i, (framework_name, embedding_layer) in enumerate(self.framework_embeddings.items()):
            # Project through framework-specific layer
            framework_output = embedding_layer(x)
            framework_outputs[framework_name] = framework_output
            
            # Weight by framework selection
            weighted_output = framework_output * framework_weights[..., i:i+1]
            weighted_outputs.append(weighted_output)
        
        # Combine all weighted framework outputs
        combined = torch.cat(weighted_outputs, dim=-1)
        combined = self.framework_combiner(combined)
        
        return combined, framework_outputs


class EthicalCrossDomainLayer(nn.Module):
    """
    Cross-domain layer that connects ethical reasoning across different contexts
    (economic, political, social, personal).
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_domains: int = 4,
                 n_heads: int = 8,
                 activation: str = "gelu"):
        super().__init__()
        
        self.domain_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_domains)
        ])
        
        self.cross_domain_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.domain_fusion = nn.Sequential(
            nn.Linear(d_model * n_domains, d_model * 2),
            get_activation(activation),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through different ethical domains and fuse representations.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            domain_fused: Cross-domain ethical representation
        """
        domain_outputs = []
        for domain_proj in self.domain_projections:
            domain_outputs.append(domain_proj(x))  # (batch, seq_len, d_model)
        # Concatenate domains: (batch, seq_len, n_domains * d_model)
        domains_stacked = torch.cat(domain_outputs, dim=-1)  # (batch, seq_len, n_domains * d_model)
        # Für Attention: Split in n_domains, dann stacken und wieder zusammenführen
        domains_split = torch.stack(domain_outputs, dim=2)  # (batch, seq_len, n_domains, d_model)
        domains_reshaped = domains_split.view(-1, x.size(1), x.size(2))  # (batch * n_domains, seq_len, d_model)
        attended_domains, _ = self.cross_domain_attention(domains_reshaped, domains_reshaped, domains_reshaped)
        attended_domains = attended_domains.reshape(x.size(0), x.size(1), -1)  # (batch, seq_len, n_domains * d_model)
        domain_fused = self.domain_fusion(attended_domains)
        return domain_fused


class MultiFrameworkProcessor(nn.Module):
    """
    Processes text through multiple ethical frameworks simultaneously and
    detects conflicts or consensus across frameworks.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_frameworks: int = 5,
                 conflict_detection_dim: int = 128,
                 activation: str = "gelu"):
        super().__init__()
        
        self.framework_embedding = MoralFrameworkEmbedding(d_model, d_model // 2, n_frameworks)
        
        # Detect conflicts between frameworks
        self.conflict_detector = nn.Sequential(
            nn.Linear(d_model, conflict_detection_dim),
            get_activation(activation),
            nn.Linear(conflict_detection_dim, 1),
            nn.Sigmoid()
        )
        
        # Framework consensus layer
        self.consensus_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            activation=activation
        )
        
        self.framework_weights = nn.Parameter(torch.randn(n_frameworks))
        
    def forward(self, x: torch.Tensor, symbolic_constraints: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """
        Process input through multiple ethical frameworks.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - framework_outputs: Raw framework outputs
                - conflict_scores: Conflict detection scores
                - consensus_output: Consensus representation
        """
        # Get framework embeddings
        combined_output, framework_outputs = self.framework_embedding(x)
        
        # Detect framework conflicts
        conflict_scores = self.conflict_detector(combined_output)
        
        # Build consensus
        consensus_output = self.consensus_layer(combined_output)
        
        result = {
            'framework_outputs': framework_outputs,
            'conflict_scores': conflict_scores,
            'consensus_output': consensus_output,
            'combined_output': combined_output
        }
        if symbolic_constraints is not None:
            symbolic_result = symbolic_constraints(result)
            if symbolic_result is not None:
                return symbolic_result
        return result


class EthicalPrincipleEncoder(nn.Module):
    """
    Encodes fundamental ethical principles (autonomy, non-maleficence, 
    beneficence, justice) into neural representations.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_principles: int = 4):
        super().__init__()
        
        # Principle embeddings
        self.principle_embeddings = nn.Embedding(n_principles, d_model)
        
        # Principle interaction matrix - angepasst für die korrekte Größe
        self.principle_interaction = nn.Parameter(
            torch.randn(d_model, d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(n_principles * d_model, d_model)
        
    def forward(self, principle_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode ethical principles with interaction modeling.
        
        Args:
            principle_ids: Tensor of shape (batch_size, n_principles)
            
        Returns:
            encoded_principles: Encoded principles with interactions
        """
        batch_size, n_principles = principle_ids.shape
        d_model = self.principle_embeddings.embedding_dim
        
        # Get embeddings for each principle
        embeddings = self.principle_embeddings(principle_ids)  # (batch, n_principles, d_model)
        
        # Model principle interactions - vereinfacht und korrigiert
        # Anwenden der Interaktion auf jedes Embedding
        interactions = torch.matmul(embeddings, self.principle_interaction)
        
        # Combine original embeddings with interactions
        combined = embeddings + interactions
        
        # Reshape and project
        combined_flat = combined.reshape(batch_size, -1)
        encoded_principles = self.output_proj(combined_flat)
        
        return encoded_principles


class MoralFrameworkGraphLayer(nn.Module):
    """
    GNN-Layer für moralische Framework-Graphen (GCNConv).
    Kann als Baustein für hybride Framework-Modelle genutzt werden.
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu"):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.activation = get_activation(activation)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gcn(x, edge_index)
        return self.activation(x)
