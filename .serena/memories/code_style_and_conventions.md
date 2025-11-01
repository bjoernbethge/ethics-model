# Code Style and Conventions

## Import Style
- **Top-level imports only** - no try/except import blocks
- **Explicit imports** - import specific names, not wildcards
- **Direct imports** - no compatibility wrappers or version checks

Example:
```python
import torch
import torch.nn as nn
from torch.nn.functional import relu
```

## Type Hints
- **Full type annotations** for all functions and methods
- Use modern Python 3.12+ type syntax
- Example: `def forward(self, x: torch.Tensor) -> torch.Tensor:`

## Class Structure
- Inherit from `nn.Module` for neural network components
- Use `super().__init__()` in constructors
- Parameters should use `nn.Parameter` for learnable values

Example:
```python
class ReCA(nn.Module):
    def __init__(self, alpha: float = 0.25, beta: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu(x) + self.alpha * relu(-x) + self.beta * x
```

## Naming Conventions
- **Classes:** PascalCase (e.g., `EthicsModel`, `ReCA`)
- **Functions/Methods:** snake_case (e.g., `get_activation`, `forward`)
- **Variables:** snake_case (e.g., `input_ids`, `attention_mask`)
- **Constants:** UPPER_CASE (rare in this project)

## Documentation
- Code is self-documenting through clear naming
- Comments used sparingly (mostly German in existing code)
- No extensive docstrings observed, focus on type hints

## Error Handling
- Use `ValueError` for invalid inputs with descriptive messages
- Example: `raise ValueError(f"Unbekannte Aktivierungsfunktion: {name}")`

## Testing Conventions
- Tests use fixtures from `conftest.py`: `summary_writer`, `cpu_or_cuda_profiler`, `symbolic_constraints`
- All tests write profiler data to TensorBoard
- Use pytest for all testing
- Test files named `test_*.py`
