"""
Tests for RoPE and GQA modules
"""

import pytest
import torch

from ethics_model.modules.rope import RotaryPositionEmbedding, apply_rope
from ethics_model.modules.gqa import GroupedQueryAttention, EthicalGQA


class TestRotaryPositionEmbedding:
    """Test RoPE implementation."""

    def test_rope_forward(self):
        """Test basic RoPE forward pass."""
        batch_size = 2
        seq_len = 10
        dim = 64

        rope = RotaryPositionEmbedding(dim=dim, max_seq_length=128)

        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)

        q_rot, k_rot = rope(q, k, seq_dim=1)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q, q_rot)  # Should be different
        assert not torch.allclose(k, k_rot)

    def test_rope_caching(self):
        """Test that RoPE caches cos/sin values."""
        dim = 64
        rope = RotaryPositionEmbedding(dim=dim, max_seq_length=128)

        # First call
        q1 = torch.randn(2, 10, dim)
        k1 = torch.randn(2, 10, dim)
        rope(q1, k1, seq_dim=1)

        assert rope._cached_seq_length == 10

        # Second call with longer sequence
        q2 = torch.randn(2, 20, dim)
        k2 = torch.randn(2, 20, dim)
        rope(q2, k2, seq_dim=1)

        assert rope._cached_seq_length == 20

    def test_rope_different_seq_dims(self):
        """Test RoPE with different sequence dimensions."""
        dim = 64
        rope = RotaryPositionEmbedding(dim=dim)

        # seq_dim=1: (batch, seq_len, dim)
        q1 = torch.randn(2, 10, dim)
        k1 = torch.randn(2, 10, dim)
        q_rot1, k_rot1 = rope(q1, k1, seq_dim=1)
        assert q_rot1.shape == (2, 10, dim)

        # seq_dim=2: (batch, heads, seq_len, dim)
        q2 = torch.randn(2, 4, 10, dim)
        k2 = torch.randn(2, 4, 10, dim)
        q_rot2, k_rot2 = rope(q2, k2, seq_dim=2)
        assert q_rot2.shape == (2, 4, 10, dim)

    def test_apply_rope_convenience(self):
        """Test apply_rope convenience function."""
        dim = 64
        rope = RotaryPositionEmbedding(dim=dim)

        q = torch.randn(2, 10, dim)
        k = torch.randn(2, 10, dim)

        q_rot, k_rot = apply_rope(q, k, rope, seq_dim=1)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestGroupedQueryAttention:
    """Test GQA implementation."""

    def test_gqa_forward(self):
        """Test basic GQA forward pass."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2
        batch_size = 2
        seq_len = 10

        gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=False,
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights = gqa(x, return_attention=False)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is None

    def test_gqa_with_rope(self):
        """Test GQA with RoPE enabled."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2
        batch_size = 2
        seq_len = 10

        gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=True,
            max_seq_length=128,
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output, _ = gqa(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_gqa_with_mask(self):
        """Test GQA with attention mask."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2
        batch_size = 2
        seq_len = 10

        gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=False,
        )

        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        output, _ = gqa(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_gqa_return_attention(self):
        """Test GQA with attention weight return."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2
        batch_size = 2
        seq_len = 10

        gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=False,
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights = gqa(x, return_attention=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_gqa_kv_repetition(self):
        """Test that KV heads are properly repeated."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2

        gqa = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=False,
        )

        # Test internal _repeat_kv method
        batch_size = 2
        seq_len = 10
        head_dim = d_model // n_heads

        kv = torch.randn(batch_size, n_kv_heads, seq_len, head_dim)
        kv_repeated = gqa._repeat_kv(kv)

        assert kv_repeated.shape == (batch_size, n_heads, seq_len, head_dim)


class TestEthicalGQA:
    """Test EthicalGQA implementation."""

    def test_ethical_gqa_forward(self):
        """Test basic EthicalGQA forward pass."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2
        batch_size = 2
        seq_len = 10

        ethical_gqa = EthicalGQA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=True,
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights, salience = ethical_gqa(
            x, return_attention=False, return_salience=False
        )

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is None
        assert salience is None

    def test_ethical_gqa_with_salience(self):
        """Test EthicalGQA with salience scoring."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2
        batch_size = 2
        seq_len = 10

        ethical_gqa = EthicalGQA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=False,
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights, salience = ethical_gqa(
            x, return_attention=True, return_salience=True
        )

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        assert salience is not None
        assert salience.shape == (batch_size, seq_len, 1)

        # Salience scores should be in [0, 1] range (sigmoid output)
        assert torch.all(salience >= 0) and torch.all(salience <= 1)

    def test_ethical_gqa_gradient_flow(self):
        """Test that gradients flow through EthicalGQA."""
        d_model = 256
        n_heads = 8
        n_kv_heads = 2

        ethical_gqa = EthicalGQA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=True,
        )

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output, _, salience = ethical_gqa(x, return_salience=True)

        # Compute loss and backprop
        loss = output.sum() + salience.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert ethical_gqa.q_proj.weight.grad is not None
        assert ethical_gqa.salience_scorer[0].weight.grad is not None
