"""
Moral Framework Components

Implements specialized neural network layers for processing different ethical
frameworks including:
- Deontological ethics (duty-based)
- Utilitarian ethics (consequentialist)
- Virtue ethics (character-based)
- Narrative ethics
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric import EdgeIndex
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.aggr import VariancePreservingAggregation

from .activation import RMSNorm, get_activation


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


class EthicalCrossDomainLayer(MessagePassing):
    """
    Graph-native cross-domain layer connecting ethical reasoning across contexts.
    Uses GATv2Conv for cross-domain attention.
    """
    def __init__(
        self,
        d_model: int,
        n_domains: int = 4,
        activation: str = "silu",
    ):
        from torch_geometric.nn import GATv2Conv
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Domain-specific projections
        self.domain_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_domains)
        ])
        
        # Cross-domain attention via GATv2Conv
        self.cross_domain_gat = GATv2Conv(
            d_model * n_domains, d_model, heads=4, concat=False
        )
        
        # Domain fusion
        self.domain_fusion = nn.Sequential(
            nn.Linear(d_model * n_domains, d_model),
            get_activation(activation),
            RMSNorm(d_model)
        )
        self.activation = get_activation(activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex
    ) -> torch.Tensor:
        """
        Process graph through different ethical domains.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            domain_fused: Cross-domain ethical representation
        """
        # Project through each domain
        domain_outputs = [proj(x) for proj in self.domain_projections]
        
        # Combine domains
        combined = torch.cat(domain_outputs, dim=-1)
        
        # Cross-domain attention via graph convolution
        attended = self.cross_domain_gat(combined, edge_index)
        attended = self.activation(attended)
        
        # Final fusion
        domain_fused = self.domain_fusion(combined) + attended
        return domain_fused


class MultiFrameworkProcessor(MessagePassing):
    """
    Graph-native processor for multiple ethical frameworks.
    Uses GNN layers to process frameworks across graph structure.
    """
    def __init__(
        self,
        d_model: int,
        n_frameworks: int = 5,
        activation: str = "silu",
    ):
        super().__init__(aggr='mean', flow='source_to_target')
        
        # Framework-specific GCN layers
        self.framework_convs = nn.ModuleDict({
            'deontological': GCNConv(d_model, d_model),
            'utilitarian': GCNConv(d_model, d_model),
            'virtue': GCNConv(d_model, d_model),
            'narrative': GCNConv(d_model, d_model),
            'care': GCNConv(d_model, d_model),
        })
        
        # Framework combination
        self.framework_combiner = nn.Sequential(
            nn.Linear(d_model * n_frameworks, d_model),
            get_activation(activation),
            RMSNorm(d_model)
        )
        
        # Consensus layer (GCN for consensus building)
        self.consensus_conv = GCNConv(d_model, d_model)
        self.activation = get_activation(activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex,
        symbolic_constraints: Optional[Callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process graph through multiple ethical frameworks.
        
        Args:
            x: Node features (num_nodes, d_model)
            edge_index: Graph edge indices
            
        Returns:
            Dictionary with framework outputs and consensus
        """
        framework_outputs = {}
        for name, conv in self.framework_convs.items():
            framework_outputs[name] = self.activation(conv(x, edge_index))
        
        # Combine frameworks
        combined = torch.cat(list(framework_outputs.values()), dim=-1)
        combined_output = self.framework_combiner(combined)
        
        # Build consensus via graph convolution
        consensus_output = self.consensus_conv(combined_output, edge_index)
        consensus_output = self.activation(consensus_output)
        
        result = {
            'framework_outputs': framework_outputs,
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

        # Principle interaction matrix - adjusted for correct size
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
    Modern GNN layer for moral framework graphs using variance-preserving aggregation.
    Prevents over-smoothing in deep moral reasoning hierarchies.
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = "silu"):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels,
                           aggr=VariancePreservingAggregation())
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: EdgeIndex) -> torch.Tensor:
        """Forward pass requiring EdgeIndex (not COO tensor)."""
        x = self.gcn(x, edge_index)
        return self.activation(x)
