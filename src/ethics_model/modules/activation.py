import torch
import torch.nn as nn
from torch.nn.functional import relu

# Moderne Aktivierungsfunktion (2025):
class ReCA(nn.Module):
    def __init__(self, alpha: float = 0.25, beta: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu(x) + self.alpha * relu(-x) + self.beta * x

def get_activation(name: str = "gelu") -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01)
    elif name == "prelu":
        return nn.PReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish":
        return nn.SiLU()
    elif name == "mish":
        return nn.Mish()
    elif name == "reca":
        return ReCA()
    else:
        raise ValueError(f"Unbekannte Aktivierungsfunktion: {name}") 