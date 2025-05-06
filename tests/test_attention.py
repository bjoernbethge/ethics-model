import torch
from ethics_model.attention import EthicalAttention, GraphAttentionLayer

def test_ethical_attention_forward(summary_writer, cpu_or_cuda_profiler):
    model = EthicalAttention(d_model=16, n_heads=2)
    x = torch.randn(2, 5, 16)
    out, attn = model(x, x, x)
    summary_writer.add_text('profiler/key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/attn_sum', float(attn.sum()), 0)
    assert out.shape == (2, 5, 16)
    assert attn.shape == (2, 5, 5)

def test_graph_attention_layer_forward(summary_writer, cpu_or_cuda_profiler):
    model = GraphAttentionLayer(in_channels=16, out_channels=16, heads=2)
    x = torch.randn(6, 16)
    # Beispiel-Graph: 6 Knoten, 8 Kanten
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2], [1, 0, 3, 2, 5, 4, 2, 0]], dtype=torch.long)
    out = model(x, edge_index)
    summary_writer.add_text('profiler/key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/out_sum', float(out.sum()), 0)
    assert out.shape == (6, 32)  # heads=2 -> out_channels*heads 