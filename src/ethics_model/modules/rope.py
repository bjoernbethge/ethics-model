"""
Rotary Position Embedding (RoPE)

Modern positional encoding that applies rotation to query and key vectors,
providing better length extrapolation than traditional positional embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for modern transformers.

    RoPE applies rotations to Q and K vectors based on their position,
    providing:
    - Better length extrapolation
    - Relative position encoding
    - Lower memory footprint than absolute position embeddings
    """

    def __init__(self, dim: int, max_seq_length: int = 2048, base: float = 10000.0):
        """
        Initialize RoPE.

        Args:
            dim: Dimension of the embeddings (must be even)
            max_seq_length: Maximum sequence length to precompute
            base: Base for the geometric progression
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos and sin values
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = 0

    def _compute_cos_sin(self, seq_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute or retrieve cached cos and sin values."""
        if seq_length > self._cached_seq_length:
            # Compute new cache
            t = torch.arange(seq_length, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
            self._cached_seq_length = seq_length

        return self._cached_cos[:seq_length], self._cached_sin[:seq_length]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor of shape (..., seq_len, dim)
            k: Key tensor of shape (..., seq_len, dim)
            seq_dim: Dimension index for sequence length (default: 1)

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_length = q.shape[seq_dim]

        # Get cos and sin
        cos, sin = self._compute_cos_sin(seq_length, q.device)

        # Reshape for broadcasting
        # cos, sin shape: (seq_len, dim)
        # Need to add dimensions to match q, k
        if seq_dim == 1:
            # q, k shape: (batch, seq_len, dim)
            cos = cos.unsqueeze(0)  # (1, seq_len, dim)
            sin = sin.unsqueeze(0)
        elif seq_dim == 2:
            # q, k shape: (batch, heads, seq_len, dim)
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
            sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionEmbedding,
    seq_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to apply RoPE to query and key tensors.

    Args:
        q: Query tensor
        k: Key tensor
        rope: RotaryPositionEmbedding instance
        seq_dim: Dimension index for sequence length

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    return rope(q, k, seq_dim=seq_dim)
