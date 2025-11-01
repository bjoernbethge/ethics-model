"""
Ethics Model Main Architecture

Pure GNN-based architecture for ethical analysis and narrative manipulation detection.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.nn as nn
from torch_geometric import EdgeIndex
from torch_geometric.nn import GATv2Conv, MessagePassing

from .activation import RMSNorm, get_activation
from .attention import GraphAttentionLayer
from .moral import MoralFrameworkGraphLayer
from .narrative import NarrativeGraphLayer


@dataclass
class EthicsGNNConfig:
    """Configuration for EthicsGNN."""
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    activation: Literal["gelu", "swish", "mish", "reca", "silu"] = "gelu"
    use_ethics_modules: bool = True


class EthicsGNN(MessagePassing):
    """Graph-native ethics model using a GATv2 stack.

    Forward signature matches PyG GNNs: (x, edge_index, batch=None)
    Returns node embeddings, pooled graph embedding, and simple scores.
    """
    def __init__(self, config: EthicsGNNConfig):
        super().__init__(aggr='mean', flow='source_to_target')
        self.config = config

        self.layers = nn.ModuleList([
            GATv2Conv(
                in_channels=(
                    -1 if i == 0
                    else config.hidden_dim * config.num_heads
                ),
                out_channels=config.hidden_dim,
                heads=config.num_heads,
                dropout=config.dropout,
                concat=True,
            )
            for i in range(config.num_layers)
        ])

        self.activations = nn.ModuleList([
            get_activation(config.activation)
            for _ in range(config.num_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.dropout)
            for _ in range(config.num_layers)
        ])

        # Graph-native ethics modules (operate directly on graph structure)
        gnn_out_dim = config.hidden_dim * config.num_heads
        if config.use_ethics_modules:
            # Graph-native layers that use edge_index directly
            self.moral_framework_layer = MoralFrameworkGraphLayer(
                gnn_out_dim, gnn_out_dim, activation=config.activation
            )
            self.ethical_attention_layer = GraphAttentionLayer(
                gnn_out_dim, gnn_out_dim, heads=config.num_heads,
                activation=config.activation, dropout=config.dropout
            )
            self.narrative_layer = NarrativeGraphLayer(
                gnn_out_dim, gnn_out_dim, activation=config.activation
            )
            
            # Meta-cognitive layer for final processing
            self.meta_cognitive = nn.Sequential(
                nn.Linear(gnn_out_dim, gnn_out_dim),
                RMSNorm(gnn_out_dim),
                get_activation(config.activation),
            )
        
        # Scoring heads use pooled embedding
        final_dim = gnn_out_dim
        self.projection = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            get_activation(config.activation),
            RMSNorm(final_dim),
        )
        self.ethics_head = nn.Sequential(
            nn.Linear(final_dim, 1), nn.Sigmoid()
        )
        self.manipulation_head = nn.Sequential(
            nn.Linear(final_dim, 1), nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: EdgeIndex | torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        moral_context: Optional[torch.Tensor] = None,
        symbolic_constraints: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        # GATv2Conv layers for graph encoding
        for conv, act, drop in zip(self.layers, self.activations, self.dropouts):
            x = conv(x, edge_index)
            x = act(x)
            x = drop(x)

        node_embeddings = x

        # Apply ethics-specific processing if enabled
        # Work directly with node embeddings (num_nodes, hidden_dim) - graph-native!
        if self.config.use_ethics_modules:
            # Convert edge_index to EdgeIndex if needed
            if not isinstance(edge_index, EdgeIndex):
                if isinstance(edge_index, torch.Tensor):
                    num_nodes = x.size(0)
                    edge_index = EdgeIndex(
                        edge_index,
                        sparse_size=(num_nodes, num_nodes)
                    )
            
            # Graph-native ethics layers (work directly with edge_index)
            x = self.moral_framework_layer(x, edge_index)
            x = self.ethical_attention_layer(x, edge_index)
            x = self.narrative_layer(x, edge_index)
            
            # Meta-cognitive processing on graph-processed embeddings
            final_embedding = self.meta_cognitive(x)
            framework_outputs = {}
            intuition_outputs = {}
            system_outputs = {}
            narrative_outputs = {}
            framing_outputs = {}
            dissonance_outputs = {}
            manipulation_outputs = {}
            propaganda_outputs = {}
            attention_weights = None
        else:
            # Simple pooling without ethics modules
            final_embedding = node_embeddings
            framework_outputs = {}
            intuition_outputs = {}
            system_outputs = {}
            narrative_outputs = {}
            framing_outputs = {}
            dissonance_outputs = {}
            manipulation_outputs = {}
            propaganda_outputs = {}
            attention_weights = None

        # Pool to graph level (use batch if provided)
        if batch is not None:
            num_graphs = (
                int(batch.max().item()) + 1
                if batch.numel() > 0
                else 1
            )
            device = final_embedding.device
            graph_sums = torch.zeros(
                num_graphs, final_embedding.size(-1), device=device
            )
            graph_counts = torch.zeros(num_graphs, 1, device=device)
            graph_sums.index_add_(0, batch, final_embedding)
            graph_counts.index_add_(
                0, batch, torch.ones_like(final_embedding[:, :1])
            )
            graph_embedding = graph_sums / graph_counts.clamp_min(1.0)
        else:
            graph_embedding = final_embedding.mean(dim=0, keepdim=True)

        graph_embedding = self.projection(graph_embedding)
        ethics_score = self.ethics_head(graph_embedding)
        manipulation_score = self.manipulation_head(graph_embedding)

        return {
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "ethics_score": ethics_score,
            "manipulation_score": manipulation_score,
            "framework_analysis": framework_outputs if self.config.use_ethics_modules else {},
            "intuition_analysis": intuition_outputs if self.config.use_ethics_modules else {},
            "dual_process_analysis": system_outputs if self.config.use_ethics_modules else {},
            "narrative_analysis": narrative_outputs if self.config.use_ethics_modules else {},
            "framing_analysis": framing_outputs if self.config.use_ethics_modules else {},
            "dissonance_analysis": dissonance_outputs if self.config.use_ethics_modules else {},
            "manipulation_analysis": manipulation_outputs if self.config.use_ethics_modules else {},
            "propaganda_analysis": propaganda_outputs if self.config.use_ethics_modules else {},
            "attention_weights": attention_weights,
        }


def create_ethics_gnn(
    config: EthicsGNNConfig | Dict[str, Any]
) -> EthicsGNN:
    if isinstance(config, dict):
        config = EthicsGNNConfig(**config)
    return EthicsGNN(config)

