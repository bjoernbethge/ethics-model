import torch
import pytest
from ethics_model.modules.activation import get_activation, ReCA

def test_get_activation_all(summary_writer, cpu_or_cuda_profiler, symbolic_constraints):
    names = ["relu", "leakyrelu", "prelu", "gelu", "swish", "mish", "reca"]
    for name in names:
        act = get_activation(name)
        x = torch.randn(4, 4)
        out = act(x)
        summary_writer.add_text(f'profiler/{name}_key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
        summary_writer.add_scalar(f'dummy/{name}_sum', float(out.sum()), 0)
        assert out.shape == x.shape

def test_get_activation_invalid():
    with pytest.raises(ValueError):
        get_activation("unknown")

def test_reca_forward(summary_writer, cpu_or_cuda_profiler):
    act = ReCA()
    x = torch.randn(3, 3)
    out = act(x)
    summary_writer.add_text('profiler/reca_key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/reca_sum', float(out.sum()), 0)
    assert out.shape == x.shape 