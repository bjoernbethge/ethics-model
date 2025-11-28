"""
Test moral framework components.
"""
import torch
import pytest

from ethics_model.modules.moral import (
    MoralFrameworkEmbedding,
    EthicalCrossDomainLayer,
    MultiFrameworkProcessor,
    EthicalPrincipleEncoder,
    MoralFrameworkGraphLayer
)


class TestMoralFrameworkEmbedding:
    """Test moral framework embedding."""
    
    def test_moral_framework_embedding_forward(self):
        """Test moral framework embedding forward pass."""
        input_dim = 64
        framework_dim = 32
        embedding = MoralFrameworkEmbedding(input_dim, framework_dim)
        
        x = torch.randn(2, 10, input_dim)
        combined, framework_outputs = embedding(x)
        
        assert combined.shape == (2, 10, input_dim)
        assert isinstance(framework_outputs, dict)
        
        expected_frameworks = ['deontological', 'utilitarian', 'virtue', 'narrative', 'care']
        for framework in expected_frameworks:
            assert framework in framework_outputs
            assert framework_outputs[framework].shape == (2, 10, framework_dim)
    
    def test_moral_framework_different_dims(self):
        """Test with different dimensions."""
        input_dim = 128
        framework_dim = 64
        embedding = MoralFrameworkEmbedding(input_dim, framework_dim, n_frameworks=3)
        
        x = torch.randn(1, 5, input_dim)
        combined, framework_outputs = embedding(x)
        
        assert combined.shape == (1, 5, input_dim)
        assert len(framework_outputs) == 5  # Always 5 frameworks


class TestEthicalCrossDomainLayer:
    """Test ethical cross-domain layer."""
    
    def test_cross_domain_forward(self):
        """Test cross-domain layer forward pass."""
        d_model = 64
        layer = EthicalCrossDomainLayer(d_model, n_domains=4)
        
        num_nodes = 8
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, d_model)
    
    def test_cross_domain_different_domains(self):
        """Test with different number of domains."""
        d_model = 32
        layer = EthicalCrossDomainLayer(d_model, n_domains=6)
        
        num_nodes = 4
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, d_model)


class TestMultiFrameworkProcessor:
    """Test multi-framework processor."""
    
    def test_multi_framework_forward(self):
        """Test multi-framework processor forward pass."""
        d_model = 64
        processor = MultiFrameworkProcessor(d_model)
        
        num_nodes = 6
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        outputs = processor(x, edge_index)
        
        assert 'framework_outputs' in outputs
        assert 'consensus_output' in outputs
        assert 'combined_output' in outputs
        
        assert outputs['consensus_output'].shape == (num_nodes, d_model)
        assert outputs['combined_output'].shape == (num_nodes, d_model)
        assert len(outputs['framework_outputs']) == 5  # 5 frameworks
    
    def test_multi_framework_with_constraints(self):
        """Test with symbolic constraints."""
        d_model = 32
        processor = MultiFrameworkProcessor(d_model)
        
        def mock_constraint(result):
            # Mock constraint that modifies combined output
            result['combined_output'] = result['combined_output'] * 0.5
            return result
        
        num_nodes = 4
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        outputs = processor(x, edge_index, symbolic_constraints=mock_constraint)
        
        assert 'combined_output' in outputs
        # Constraint should have modified the output
        assert outputs['combined_output'].shape == (num_nodes, d_model)


class TestEthicalPrincipleEncoder:
    """Test ethical principle encoder."""
    
    def test_principle_encoder_forward(self):
        """Test principle encoder forward pass."""
        d_model = 64
        n_principles = 4
        encoder = EthicalPrincipleEncoder(d_model, n_principles)
        
        principle_ids = torch.randint(0, n_principles, (2, n_principles))
        output = encoder(principle_ids)
        
        assert output.shape == (2, d_model)
    
    def test_principle_encoder_different_principles(self):
        """Test with different number of principles."""
        d_model = 32
        n_principles = 6
        encoder = EthicalPrincipleEncoder(d_model, n_principles)
        
        principle_ids = torch.randint(0, n_principles, (1, n_principles))
        output = encoder(principle_ids)
        
        assert output.shape == (1, d_model)
    
    def test_principle_encoder_single_batch(self):
        """Test with single batch dimension."""
        d_model = 16
        n_principles = 3
        encoder = EthicalPrincipleEncoder(d_model, n_principles)
        
        # Test with the exact number of principles the encoder expects
        principle_ids = torch.randint(0, n_principles, (1, n_principles))
        output = encoder(principle_ids)
        
        assert output.shape == (1, d_model)


class TestMoralFrameworkGraphLayer:
    """Test moral framework graph layer."""
    
    def test_moral_framework_graph_layer_forward(self):
        """Test moral framework graph layer forward pass."""
        in_channels = 16
        out_channels = 32
        layer = MoralFrameworkGraphLayer(in_channels, out_channels)
        
        # Create sample graph data
        num_nodes = 4
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, out_channels)
    
    def test_moral_framework_graph_different_activation(self):
        """Test with different activation function."""
        layer = MoralFrameworkGraphLayer(8, 16, activation="relu")
        
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (3, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
