"""
Grouped Query Attention (GQA)

Modern attention mechanism that uses fewer key/value heads than query heads
for improved memory efficiency while maintaining model quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .activation import RMSNorm
from .rope import RotaryPositionEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - fewer K/V heads than Q heads.

    GQA is a middle ground between Multi-Head Attention (MHA) and
    Multi-Query Attention (MQA):
    - MHA: n_heads K/V heads for n_heads Q heads
    - GQA: n_kv_heads K/V heads for n_heads Q heads (n_kv_heads < n_heads)
    - MQA: 1 K/V head for n_heads Q heads

    Benefits:
    - Reduced memory usage (fewer K/V cache in inference)
    - Faster inference
    - Similar quality to MHA with proper tuning
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_length: int = 2048,
    ):
        """
        Initialize GQA.

        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key/value heads (should divide n_heads evenly)
            dropout: Dropout probability
            use_rope: Whether to use Rotary Position Embedding
            max_seq_length: Maximum sequence length for RoPE
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads per KV head

        # Q projection: full number of heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # K, V projections: reduced number of heads
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

        # Optional RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                self.head_dim, max_seq_length=max_seq_length
            )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat K/V heads to match number of Q heads.

        Args:
            x: Tensor of shape (batch, n_kv_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch, n_heads, seq_len, head_dim)
        """
        batch, n_kv_heads, seq_len, head_dim = x.shape

        if self.n_rep == 1:
            return x

        # Repeat each KV head n_rep times
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch, n_kv_heads * self.n_rep, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of GQA.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            - output: shape (batch, seq_len, d_model)
            - attention_weights: shape (batch, n_heads, seq_len, seq_len) if requested
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, seq_len, head_dim)

        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # (batch, n_kv_heads, seq_len, head_dim)

        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # (batch, n_kv_heads, seq_len, head_dim)

        # Apply RoPE if enabled
        if self.use_rope:
            # Apply RoPE to each head separately
            # Need to reshape: (batch, heads, seq_len, head_dim) -> (batch * heads, seq_len, head_dim)
            q_rope = q.reshape(batch_size * self.n_heads, seq_len, self.head_dim)
            k_rope = k.reshape(batch_size * self.n_kv_heads, seq_len, self.head_dim)

            q_rope, k_rope = self.rope(q_rope, k_rope, seq_dim=1)

            # Reshape back
            q = q_rope.reshape(batch_size, self.n_heads, seq_len, self.head_dim)
            k = k_rope.reshape(batch_size, self.n_kv_heads, seq_len, self.head_dim)

        # Repeat K, V to match Q heads
        k = self._repeat_kv(k)  # (batch, n_heads, seq_len, head_dim)
        v = self._repeat_kv(v)  # (batch, n_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # (batch, n_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # (batch, n_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Apply normalization
        output = self.norm(output)

        if return_attention:
            return output, attn_weights
        return output, None


class EthicalGQA(GroupedQueryAttention):
    """
    Grouped Query Attention specialized for ethical reasoning.

    Extends GQA with ethical salience scoring.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_length: int = 2048,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_length=max_seq_length,
        )

        # Ethical salience scorer
        self.salience_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_salience: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with ethical salience scoring.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            return_salience: Whether to return salience scores

        Returns:
            Tuple of (output, attention_weights, salience_scores)
        """
        # Run standard GQA
        output, attn_weights = super().forward(x, mask, return_attention)

        # Compute ethical salience
        salience_scores = None
        if return_salience:
            salience_scores = self.salience_scorer(output)

        return output, attn_weights, salience_scores
