import pytest
import torch

from ethics_model.modules.activation import ReCA, RMSNorm, SwiGLU, get_activation


def test_get_activation_all(summary_writer, cpu_or_cuda_profiler, symbolic_constraints):
    names = ["relu", "leakyrelu", "prelu", "gelu", "swish", "mish", "reca", "silu"]
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

def test_rmsnorm_forward(summary_writer, cpu_or_cuda_profiler):
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    out = norm(x)
    summary_writer.add_text('profiler/rmsnorm_key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/rmsnorm_mean', float(out.mean()), 0)
    assert out.shape == x.shape

def test_swiglu_forward(summary_writer, cpu_or_cuda_profiler):
    act = SwiGLU(16)
    x = torch.randn(2, 4, 32)  # Input needs to be 2x dim for chunking
    out = act(x)
    summary_writer.add_text('profiler/swiglu_key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/swiglu_sum', float(out.sum()), 0)
    assert out.shape == (2, 4, 16) 