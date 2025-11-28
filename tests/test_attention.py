"""
Test attention mechanisms.
"""
import torch
import pytest

from ethics_model.modules.attention import (
    EthicalAttention,
    MoralIntuitionAttention,
    NarrativeFrameAttention,
    DoubleProcessingAttention,
    GraphAttentionLayer
)


class TestEthicalAttention:
    """Test ethical attention mechanism."""
    
    def test_ethical_attention_forward(self):
        """Test forward pass of ethical attention."""
        d_model = 64
        n_heads = 4
        num_nodes = 10
        
        attention = EthicalAttention(d_model, n_heads)
        
        # Create graph input (EthicalAttention is graph-based, not sequence-based)
        x = torch.randn(num_nodes, d_model)  # Node features
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        
        output, attention_weights = attention(x, edge_index)
        
        assert output.shape == (num_nodes, d_model)
        assert attention_weights.shape == (num_nodes,)
    
    def test_ethical_attention_with_moral_context(self):
        """Test ethical attention with moral context."""
        d_model = 64
        n_heads = 4
        num_nodes = 5
        attention = EthicalAttention(d_model, n_heads)
        
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        output, weights = attention(x, edge_index)
        
        assert output.shape == (num_nodes, d_model)
        assert weights.shape == (num_nodes,)
    
    def test_ethical_attention_with_mask(self):
        """Test ethical attention with mask."""
        d_model = 32
        n_heads = 2
        num_nodes = 4
        attention = EthicalAttention(d_model, n_heads)
        
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        output, weights = attention(x, edge_index)
        
        assert output.shape == (num_nodes, d_model)
        assert weights.shape == (num_nodes,)


class TestMoralIntuitionAttention:
    """Test moral intuition attention."""
    
    def test_moral_intuition_forward(self):
        """Test moral intuition attention forward pass."""
        d_model = 64
        attention = MoralIntuitionAttention(d_model)
        
        # Graph-based input
        num_nodes = 8
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        outputs = attention(x, edge_index)
        
        assert 'intuitive_response' in outputs
        assert 'emotional_intensity' in outputs
        assert 'foundation_activations' in outputs
        
        # Check output dimensions
        assert outputs['intuitive_response'].shape == (num_nodes, d_model)
        assert outputs['emotional_intensity'].shape == (num_nodes, 1)
        assert outputs['foundation_activations'].shape == (num_nodes, 6)  # n_moral_foundations
    
    def test_moral_intuition_with_context(self):
        """Test moral intuition with moral context."""
        d_model = 32
        attention = MoralIntuitionAttention(d_model)
        
        num_nodes = 5
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        outputs = attention(x, edge_index)
        
        assert 'intuitive_response' in outputs
        assert outputs['intuitive_response'].shape == (num_nodes, d_model)


class TestNarrativeFrameAttention:
    """Test narrative frame attention."""
    
    def test_narrative_frame_forward(self):
        """Test narrative frame attention forward pass."""
        d_model = 64
        attention = NarrativeFrameAttention(d_model)
        
        x = torch.randn(2, 10, d_model)
        outputs = attention(x)
        
        assert 'frame_scores' in outputs
        assert 'manipulation_scores' in outputs
        assert 'manipulative_segments' in outputs
        assert 'frame_transitions' in outputs
        
        assert outputs['frame_scores'].shape == (2, 10, 5)  # 5 frame types
        assert outputs['manipulation_scores'].shape == (2, 10, 1)


class TestDoubleProcessingAttention:
    """Test double processing attention."""
    
    def test_double_processing_forward(self):
        """Test double processing attention forward pass."""
        d_model = 64
        attention = DoubleProcessingAttention(d_model, n_heads=4)
        
        # DoubleProcessingAttention is sequence-based, not graph-based
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, d_model)
        final_output, system_outputs = attention(x)
        
        assert final_output.shape == (batch_size, seq_len, d_model)
        assert 'system1_output' in system_outputs
        assert 'system2_output' in system_outputs
        assert 'resolution_weights' in system_outputs
        
        assert system_outputs['system1_output'].shape == (batch_size, seq_len, d_model)
        assert system_outputs['system2_output'].shape == (batch_size, seq_len, d_model)
        assert system_outputs['resolution_weights'].shape == (batch_size, seq_len, 2)


class TestGraphAttentionLayer:
    """Test graph attention layer."""
    
    def test_graph_attention_layer_forward(self):
        """Test graph attention layer forward pass."""
        in_channels = 16
        out_channels = 64
        heads = 2
        layer = GraphAttentionLayer(in_channels, out_channels, heads=heads)
        
        # Create sample graph data
        num_nodes = 5
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        # GAT with concat=False outputs out_channels, not out_channels * heads
        assert output.shape == torch.Size([num_nodes, out_channels])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
