import torch
from ethics_model.moral import MoralFrameworkEmbedding
from ethics_model.moral import MoralFrameworkGraphLayer

def test_moral_framework_embedding_forward(summary_writer, cpu_or_cuda_profiler):
    model = MoralFrameworkEmbedding(input_dim=8, framework_dim=4, n_frameworks=3)
    x = torch.randn(2, 5, 8)
    combined, outputs = model(x)
    summary_writer.add_text('profiler/key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/combined_sum', float(combined.sum()), 0)
    assert combined.shape == (2, 5, 8)
    assert isinstance(outputs, dict)
    assert all(v.shape == (2, 5, 4) for v in outputs.values())

def test_moral_framework_graph_layer_forward(summary_writer, cpu_or_cuda_profiler):
    model = MoralFrameworkGraphLayer(in_channels=8, out_channels=8)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    out = model(x, edge_index)
    summary_writer.add_text('profiler/key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/out_sum', float(out.sum()), 0)
    assert out.shape == (6, 8) 