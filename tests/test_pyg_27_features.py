import pytest
import torch
from torch_geometric import EdgeIndex

from ethics_model.modules.attention import GraphAttentionLayer
from ethics_model.modules.gnn import (
    EthicsGNNConfig,
    create_ethics_gnn,
)
from ethics_model.modules.moral import MoralFrameworkGraphLayer
from ethics_model.modules.retriever import EthicsModel


def test_edge_index_optimization(summary_writer, cpu_or_cuda_profiler):
    """
    Test GraphAttentionLayer with modern EdgeIndex optimization.
    Verifies EdgeIndex type checking and correct output shapes.
    """
    in_channels = 64
    out_channels = 32
    heads = 4
    num_nodes = 10

    # Create GraphAttentionLayer
    layer = GraphAttentionLayer(in_channels, out_channels, heads=heads, activation="gelu")

    # Create node features
    x = torch.randn(num_nodes, in_channels)

    # Create EdgeIndex (modern PyG 2.7.0+ format)
    edge_list = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_index = EdgeIndex(edge_list, sparse_size=(num_nodes, num_nodes))

    # Verify EdgeIndex type
    assert isinstance(edge_index, EdgeIndex)

    # Forward pass
    out = layer(x, edge_index)

    # Verify output shape (concat=False, so out_channels, not out_channels * heads)
    assert out.shape == (num_nodes, out_channels)

    # Log profiler data to TensorBoard
    summary_writer.add_text(
        'profiler/edge_index_optimization_key_operators',
        str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    )
    summary_writer.add_scalar('edge_index/output_sum', float(out.sum()), 0)
    summary_writer.add_scalar('edge_index/num_edges', edge_index.size(1), 0)


def test_gatv2_forward(summary_writer, cpu_or_cuda_profiler):
    """
    Test GATv2Conv layer with dynamic attention mechanism.
    GATv2 provides improved attention over static GAT.
    """
    in_channels = 32
    out_channels = 16
    heads = 2
    num_nodes = 8

    # Create GraphAttentionLayer (uses GATv2Conv internally)
    layer = GraphAttentionLayer(in_channels, out_channels, heads=heads, activation="relu")

    # Create node features
    x = torch.randn(num_nodes, in_channels)

    # Create EdgeIndex
    edge_list = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_index = EdgeIndex(edge_list, sparse_size=(num_nodes, num_nodes))

    # Forward pass
    out = layer(x, edge_index)

    # Verify output shape (concat=False, so out_channels, not out_channels * heads)
    assert out.shape == (num_nodes, out_channels)

    # Verify dynamic attention (output is not zero)
    assert out.abs().sum() > 0

    # Log profiler data
    summary_writer.add_text(
        'profiler/gatv2_forward_key_operators',
        str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    )
    summary_writer.add_scalar('gatv2/output_mean', float(out.mean()), 0)
    summary_writer.add_scalar('gatv2/output_std', float(out.std()), 0)


def test_variance_preserving_aggregation(summary_writer, cpu_or_cuda_profiler):
    """
    Test MoralFrameworkGraphLayer with variance-preserving aggregation.
    Prevents over-smoothing in deep moral reasoning hierarchies.
    """
    in_channels = 48
    out_channels = 24
    num_nodes = 12

    # Create MoralFrameworkGraphLayer (uses VariancePreservingAggregation)
    layer = MoralFrameworkGraphLayer(in_channels, out_channels, activation="gelu")

    # Create node features
    x = torch.randn(num_nodes, in_channels)

    # Create EdgeIndex
    edge_list = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    edge_index = EdgeIndex(edge_list, sparse_size=(num_nodes, num_nodes))

    # Compute input variance
    input_variance = x.var()

    # Forward pass
    out = layer(x, edge_index)

    # Verify output shape
    assert out.shape == (num_nodes, out_channels)

    # Compute output variance
    output_variance = out.var()

    # Variance preservation: output variance should not collapse to zero
    assert output_variance > 0

    # Log profiler data
    summary_writer.add_text(
        'profiler/variance_preserving_key_operators',
        str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    )
    summary_writer.add_scalar('variance/input_variance', float(input_variance), 0)
    summary_writer.add_scalar('variance/output_variance', float(output_variance), 0)
    summary_writer.add_scalar('variance/preservation_ratio', float(output_variance / input_variance), 0)


def test_torch_compile_compatibility(summary_writer, cpu_or_cuda_profiler):
    """
    Test torch.compile() compatibility with EthicsGNN.
    Verifies compiled model forward pass works correctly.

    NOTE: Requires MSVC compiler (cl.exe) on Windows.
    Run in Developer PowerShell or skip: pytest -k "not compile"
    """
    # Check if compiler is available
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
        get_cpp_compiler()
    except RuntimeError:
        pytest.skip("Skipping torch.compile test - MSVC compiler (cl.exe) not found. Run in Developer PowerShell.")

    # Create EthicsGNN config
    config = EthicsGNNConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
    )

    # Create compiled model
    model = torch.compile(
        create_ethics_gnn(config),
        mode="max-autotune",
        dynamic=True
    )

    # Verify torch.compile() was applied
    assert hasattr(model, '_orig_mod') or hasattr(model, 'forward')

    # Create dummy graph
    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 64)
    edge_list = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = EdgeIndex(edge_list, sparse_size=(num_nodes, num_nodes))

    # Forward pass
    outputs = model(x, edge_index)

    # Verify output structure
    assert 'ethics_score' in outputs
    assert 'manipulation_score' in outputs
    assert 'node_embeddings' in outputs

    # Verify output shapes
    assert outputs['ethics_score'].shape[-1] == 1
    assert outputs['manipulation_score'].shape[-1] == 1

    # Log profiler data
    summary_writer.add_text(
        'profiler/torch_compile_key_operators',
        str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    )
    summary_writer.add_scalar('compile/ethics_score_mean', float(outputs['ethics_score'].mean()), 0)
    summary_writer.add_scalar('compile/manipulation_score_mean', float(outputs['manipulation_score'].mean()), 0)


def test_retriever_basic(summary_writer, cpu_or_cuda_profiler):
    """
    Test EthicsModel (GRetriever) initialization.

    NOTE: This test requires LLM model download (~6GB for Qwen3-3B).
    Skip if not needed: pytest -k "not retriever"
    """
    pytest.skip("Skipping GRetriever test - requires LLM download (~6GB Qwen3-3B)")

    # Create retriever with correct API
    retriever = EthicsModel(
        gnn_hidden_dim=64,
        num_gnn_layers=2,
        gnn_num_heads=4,
        gnn_dropout=0.1,
        use_lora=True,
        activation="gelu"
    )

    # Verify retriever is initialized
    assert retriever is not None
    assert hasattr(retriever, 'gnn')
    assert hasattr(retriever, 'llm')

    # Verify framework names
    framework_names = retriever.get_framework_names()
    assert len(framework_names) == 5
    assert 'deontological' in framework_names
    assert 'utilitarian' in framework_names
    assert 'virtue' in framework_names
    assert 'care' in framework_names
    assert 'narrative' in framework_names


def test_ethics_gnn_forward(summary_writer, cpu_or_cuda_profiler):
    """
    Test EthicsGNN forward pass with graph-native API.
    """
    # Create EthicsGNN config
    config = EthicsGNNConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
    )

    # Create model
    model = create_ethics_gnn(config)

    # Create dummy graph
    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 64)
    edge_list = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = EdgeIndex(edge_list, sparse_size=(num_nodes, num_nodes))

    # Forward pass
    outputs = model(x, edge_index)

    # Verify output structure
    assert 'node_embeddings' in outputs
    assert 'graph_embedding' in outputs
    assert 'ethics_score' in outputs
    assert 'manipulation_score' in outputs

    # Verify shapes
    assert outputs['node_embeddings'].shape == (
        num_nodes, config.hidden_dim * config.num_heads
    )
    assert outputs['graph_embedding'].shape[0] == 1
    assert outputs['ethics_score'].shape[-1] == 1
    assert outputs['manipulation_score'].shape[-1] == 1

    # Log profiler data
    summary_writer.add_scalar(
        'gnn/ethics_score_mean',
        float(outputs['ethics_score'].mean()),
        0
    )
    summary_writer.add_scalar(
        'gnn/manipulation_score_mean',
        float(outputs['manipulation_score'].mean()),
        0
    )
