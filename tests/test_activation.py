"""
Test activation functions.
"""
import torch
import pytest

try:
    from ethics_model.modules.activation import get_activation, ReCA
except ImportError:
    pytest.skip("Activation modules not available", allow_module_level=True)


def test_get_activation_all():
    """Test all activation functions."""
    names = ["relu", "leakyrelu", "prelu", "gelu", "swish", "mish", "reca"]
    
    for name in names:
        act = get_activation(name)
        x = torch.randn(4, 4)
        out = act(x)
        assert out.shape == x.shape
        assert isinstance(out, torch.Tensor)


def test_get_activation_invalid():
    """Test invalid activation function name."""
    with pytest.raises(ValueError):
        get_activation("unknown_activation")


def test_reca_forward():
    """Test ReCA activation function."""
    act = ReCA()
    x = torch.randn(3, 3)
    out = act(x)
    assert out.shape == x.shape
    assert isinstance(out, torch.Tensor)


def test_reca_parameters():
    """Test ReCA with custom parameters."""
    act = ReCA(alpha=0.5, beta=0.3)
    assert abs(act.alpha.item() - 0.5) < 1e-5
    assert abs(act.beta.item() - 0.3) < 1e-5
    
    x = torch.randn(2, 2)
    out = act(x)
    assert out.shape == x.shape


def test_activation_gradients():
    """Test that activations preserve gradients."""
    x = torch.randn(2, 2, requires_grad=True)
    
    for name in ["relu", "gelu", "reca"]:
        act = get_activation(name)
        out = act(x)
        loss = out.sum()
        loss.backward(retain_graph=True)
        
        assert x.grad is not None
        x.grad.zero_()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
