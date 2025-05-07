import torch
from ethics_model.modules.narrative import NarrativeGraphLayer

def test_narrative_graph_layer_forward(summary_writer, cpu_or_cuda_profiler, symbolic_constraints):
    model = NarrativeGraphLayer(in_channels=8, out_channels=8)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    out = model(x, edge_index)
    summary_writer.add_text('profiler/key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/out_sum', float(out.sum()), 0)
    assert out.shape == (6, 8) 