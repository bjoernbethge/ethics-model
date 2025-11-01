"""
Ethics Knowledge Graph Retrieval using PyG 2.7.0 GRetriever

Uses GRetriever + EthicsGNN for knowledge graph reasoning with Qwen3-3B.
"""

from torch_geometric.llm import LLM, GRetriever

from .gnn import EthicsGNN, EthicsGNNConfig


class EthicsModel(GRetriever):
    """GRetriever-based ethics retriever using EthicsGNN as the GNN."""

    def __init__(
        self,
        gnn_hidden_dim: int = 256,
        num_gnn_layers: int = 2,
        gnn_num_heads: int = 4,
        gnn_dropout: float = 0.1,
        use_lora: bool = True,
        mlp_out_tokens: int = 1,
        activation: str = "gelu"
    ) -> None:
        """
        Initialize EthicsModel with GRetriever.

        Args:
            gnn_hidden_dim: Hidden dimension for GNN layers
            num_gnn_layers: Number of GATv2Conv layers
            gnn_num_heads: Number of attention heads in GATv2Conv
            gnn_dropout: Dropout rate for GNN layers
            use_lora: Enable LoRA for efficient LLM fine-tuning
            mlp_out_tokens: Number of output tokens from MLP
            activation: Activation function name
        """
        # Build internal LLM
        llm = LLM(
            model_name="Qwen/Qwen3-3B-Instruct",
            num_params=3,
        )

        # Build EthicsGNN as the GNN backend
        gnn_config = EthicsGNNConfig(
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=gnn_num_heads,
            dropout=gnn_dropout,
            activation=activation,
        )
        gnn = EthicsGNN(gnn_config)

        # Initialize GRetriever super with our components
        super().__init__(
            llm=llm,
            gnn=gnn,
            use_lora=use_lora,
            mlp_out_tokens=mlp_out_tokens,
        )

    def _build_gnn(self, *_, **__):
        """Backward compatibility stub."""
        raise NotImplementedError(
            "EthicsModel uses EthicsGNN internally; "
            "no manual GNN stack here."
        )

    def forward(self, *args, **kwargs):
        # Defer to GRetriever forward
        return super().forward(*args, **kwargs)

    def get_framework_names(self) -> list[str]:
        """Returns names of the 5 moral frameworks in order."""
        return [
            'deontological',
            'utilitarian',
            'virtue',
            'care',
            'narrative'
        ]
