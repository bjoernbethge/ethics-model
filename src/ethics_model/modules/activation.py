import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation - better than GELU for modern transformers."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ReCA(nn.Module):
    """Recurrent Competitive Activation - modern learnable activation."""
    def __init__(self, alpha: float = 0.25, beta: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + self.alpha * F.relu(-x) + self.beta * x


def get_activation(name: str = "swiglu") -> nn.Module:
    """Get activation function by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01)
    elif name == "prelu":
        return nn.PReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish" or name == "silu":
        return nn.SiLU()
    elif name == "mish":
        return nn.Mish()
    elif name == "reca":
        return ReCA()
    elif name == "swiglu":
        raise ValueError("SwiGLU requires dim parameter, use SwiGLU(dim) directly")
    else:
        raise ValueError(f"Unknown activation function: {name}") 