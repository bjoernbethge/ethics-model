"""
Test narrative analysis components.
"""
import torch
import pytest

from ethics_model.modules.narrative import (
    FramingDetector,
    CognitiveDissonanceLayer,
    NarrativeManipulationDetector,
    PropagandaDetector,
    NarrativeGraphLayer
)


class TestFramingDetector:
    """Test framing detector."""
    
    def test_framing_detector_forward(self):
        """Test framing detector forward pass."""
        d_model = 64
        detector = FramingDetector(d_model)
        
        num_nodes = 10
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        
        outputs = detector(x, edge_index)
        
        assert 'frame_scores' in outputs
        assert 'framing_strength' in outputs
        assert 'consistency_score' in outputs
        
        # Check frame scores for each type
        frame_scores = outputs['frame_scores']
        assert isinstance(frame_scores, dict)
        
        expected_frames = ['loss_gain', 'moral', 'episodic_thematic', 
                          'problem_solution', 'conflict_consensus', 'urgency_deliberation']
        for frame_type in expected_frames:
            assert frame_type in frame_scores
        
        assert outputs['framing_strength'].shape == (num_nodes, 1)
        assert outputs['consistency_score'].shape == (num_nodes, 1)
    
    def test_framing_detector_with_constraints(self):
        """Test framing detector with symbolic constraints."""
        d_model = 32
        detector = FramingDetector(d_model)
        
        def mock_constraint(result):
            result['framing_strength'] = result['framing_strength'] * 0.8
            return result
        
        num_nodes = 5
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        outputs = detector(x, edge_index, symbolic_constraints=mock_constraint)
        
        assert 'framing_strength' in outputs
        assert torch.all(outputs['framing_strength'] >= 0)


class TestCognitiveDissonanceLayer:
    """Test cognitive dissonance layer."""
    
    def test_cognitive_dissonance_forward(self):
        """Test cognitive dissonance layer forward pass."""
        d_model = 64
        layer = CognitiveDissonanceLayer(d_model)
        
        num_nodes = 8
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        outputs = layer(x, edge_index)
        
        assert 'dissonance_score' in outputs
        assert 'resolution_strategy' in outputs
        assert 'value_activations' in outputs
        
        assert outputs['dissonance_score'].shape == (num_nodes, 1)
        assert outputs['resolution_strategy'].shape == (num_nodes, 3)  # 3 strategies
        assert outputs['value_activations'].shape == (num_nodes, 8)  # n_moral_values
    
    def test_cognitive_dissonance_single_sequence(self):
        """Test with single sequence."""
        d_model = 32
        layer = CognitiveDissonanceLayer(d_model, n_moral_values=4)
        
        num_nodes = 3
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        outputs = layer(x, edge_index)
        
        assert outputs['dissonance_score'].shape == (num_nodes, 1)
        assert outputs['value_activations'].shape == (num_nodes, 4)


class TestNarrativeManipulationDetector:
    """Test narrative manipulation detector."""
    
    def test_manipulation_detector_forward(self):
        """Test manipulation detector forward pass."""
        d_model = 64
        detector = NarrativeManipulationDetector(d_model)
        
        num_nodes = 6
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        outputs = detector(x, edge_index)
        
        assert 'technique_scores' in outputs
        assert 'aggregate_score' in outputs
        assert 'manipulation_features' in outputs
        
        # Check technique scores
        technique_scores = outputs['technique_scores']
        assert isinstance(technique_scores, dict)
        
        expected_techniques = ['emotional_appeal', 'false_dichotomy', 'appeal_to_authority',
                              'bandwagon', 'loaded_language', 'cherry_picking', 
                              'straw_man', 'slippery_slope']
        for technique in expected_techniques:
            assert technique in technique_scores
            assert technique_scores[technique].shape == (num_nodes, 1)
        
        assert outputs['aggregate_score'].shape == (num_nodes, 1)
    
    def test_manipulation_detector_different_techniques(self):
        """Test with different number of manipulation types."""
        d_model = 32
        detector = NarrativeManipulationDetector(d_model)
        
        num_nodes = 4
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        outputs = detector(x, edge_index)
        
        assert 'technique_scores' in outputs
        assert len(outputs['technique_scores']) == 8  # Always 8 techniques


class TestPropagandaDetector:
    """Test propaganda detector."""
    
    def test_propaganda_detector_forward(self):
        """Test propaganda detector forward pass."""
        d_model = 64
        detector = PropagandaDetector(d_model)
        
        num_nodes = 8
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        
        outputs = detector(x, edge_index)
        
        assert 'intensity_score' in outputs
        assert 'credibility_score' in outputs
        assert 'inverse_credibility' in outputs
        
        assert outputs['intensity_score'].shape == (num_nodes, 1)
        assert outputs['credibility_score'].shape == (num_nodes, 1)
        assert outputs['inverse_credibility'].shape == (num_nodes, 1)
    
    def test_propaganda_detector_different_techniques(self):
        """Test with different number of propaganda techniques."""
        d_model = 32
        detector = PropagandaDetector(d_model)
        
        num_nodes = 5
        x = torch.randn(num_nodes, d_model)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        outputs = detector(x, edge_index)
        
        assert outputs['intensity_score'].shape == (num_nodes, 1)
        assert outputs['credibility_score'].shape == (num_nodes, 1)


class TestNarrativeGraphLayer:
    """Test narrative graph layer."""
    
    def test_narrative_graph_layer_forward(self):
        """Test narrative graph layer forward pass."""
        in_channels = 16
        out_channels = 32
        layer = NarrativeGraphLayer(in_channels, out_channels)
        
        # Create sample graph data
        num_nodes = 5
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, out_channels)
    
    def test_narrative_graph_empty_edges(self):
        """Test with empty edge set."""
        layer = NarrativeGraphLayer(8, 16)
        
        x = torch.randn(3, 8)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (3, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
