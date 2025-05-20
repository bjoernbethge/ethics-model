"""
Test narrative analysis components.
"""
import torch
import pytest

try:
    from ethics_model.modules.narrative import (
        FramingDetector,
        CognitiveDissonanceLayer,
        NarrativeManipulationDetector,
        PropagandaDetector,
        NarrativeGraphLayer
    )
except ImportError:
    pytest.skip("Narrative modules not available", allow_module_level=True)


class TestFramingDetector:
    """Test framing detector."""
    
    def test_framing_detector_forward(self):
        """Test framing detector forward pass."""
        d_model = 64
        detector = FramingDetector(d_model)
        
        x = torch.randn(2, 10, d_model)
        outputs = detector(x)
        
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
        
        assert outputs['framing_strength'].shape == (2, 10, 1)
        assert outputs['consistency_score'].shape == (2, 1)
    
    def test_framing_detector_with_constraints(self):
        """Test framing detector with symbolic constraints."""
        d_model = 32
        detector = FramingDetector(d_model)
        
        def mock_constraint(result):
            result['framing_strength'] = result['framing_strength'] * 0.8
            return result
        
        x = torch.randn(1, 5, d_model)
        outputs = detector(x, symbolic_constraints=mock_constraint)
        
        assert 'framing_strength' in outputs
        assert torch.all(outputs['framing_strength'] >= 0)


class TestCognitiveDissonanceLayer:
    """Test cognitive dissonance layer."""
    
    def test_cognitive_dissonance_forward(self):
        """Test cognitive dissonance layer forward pass."""
        d_model = 64
        layer = CognitiveDissonanceLayer(d_model)
        
        x = torch.randn(2, 8, d_model)
        outputs = layer(x)
        
        assert 'value_conflicts' in outputs
        assert 'dissonance_score' in outputs
        assert 'resolution_strategy' in outputs
        assert 'value_activations' in outputs
        
        assert outputs['value_conflicts'].shape == (2, 7, 1)  # seq_len - 1
        assert outputs['dissonance_score'].shape == (2, 1)
        assert outputs['resolution_strategy'].shape == (2, 8, 3)  # 3 strategies
        assert outputs['value_activations'].shape == (2, 8, 8)  # n_moral_values
    
    def test_cognitive_dissonance_single_sequence(self):
        """Test with single sequence."""
        d_model = 32
        layer = CognitiveDissonanceLayer(d_model, n_moral_values=4)
        
        x = torch.randn(1, 3, d_model)
        outputs = layer(x)
        
        assert outputs['value_conflicts'].shape == (1, 2, 1)
        assert outputs['value_activations'].shape == (1, 3, 4)


class TestNarrativeManipulationDetector:
    """Test narrative manipulation detector."""
    
    def test_manipulation_detector_forward(self):
        """Test manipulation detector forward pass."""
        d_model = 64
        detector = NarrativeManipulationDetector(d_model)
        
        x = torch.randn(2, 6, d_model)
        outputs = detector(x)
        
        assert 'technique_scores' in outputs
        assert 'aggregate_score' in outputs
        assert 'confidence' in outputs
        assert 'manipulation_map' in outputs
        
        # Check technique scores
        technique_scores = outputs['technique_scores']
        assert isinstance(technique_scores, dict)
        
        expected_techniques = ['emotional_appeal', 'false_dichotomy', 'appeal_to_authority',
                              'bandwagon', 'loaded_language', 'cherry_picking', 
                              'straw_man', 'slippery_slope']
        for technique in expected_techniques:
            assert technique in technique_scores
            assert technique_scores[technique].shape == (2, 6, 1)
        
        assert outputs['aggregate_score'].shape == (2, 6, 1)
        assert outputs['confidence'].shape == (2, 6, 1)
        assert outputs['manipulation_map'].shape == (2, 6, 1)
    
    def test_manipulation_detector_different_techniques(self):
        """Test with different number of manipulation types."""
        d_model = 32
        detector = NarrativeManipulationDetector(d_model, n_manipulation_types=4)
        
        x = torch.randn(1, 4, d_model)
        outputs = detector(x)
        
        assert 'technique_scores' in outputs
        assert len(outputs['technique_scores']) == 8  # Always 8 techniques


class TestPropagandaDetector:
    """Test propaganda detector."""
    
    def test_propaganda_detector_forward(self):
        """Test propaganda detector forward pass."""
        d_model = 64
        detector = PropagandaDetector(d_model)
        
        x = torch.randn(2, 8, d_model)
        outputs = detector(x)
        
        assert 'technique_matches' in outputs
        assert 'intensity_score' in outputs
        assert 'credibility_score' in outputs
        assert 'inverse_credibility' in outputs
        
        assert outputs['technique_matches'].shape == (2, 8, 14)  # 14 techniques
        assert outputs['intensity_score'].shape == (2, 8, 1)
        assert outputs['credibility_score'].shape == (2, 1, 1)  # Mean over sequence
        assert outputs['inverse_credibility'].shape == (2, 1, 1)
    
    def test_propaganda_detector_different_techniques(self):
        """Test with different number of propaganda techniques."""
        d_model = 32
        detector = PropagandaDetector(d_model, n_propaganda_techniques=8)
        
        x = torch.randn(1, 5, d_model)
        outputs = detector(x)
        
        assert outputs['technique_matches'].shape == (1, 5, 8)


class TestNarrativeGraphLayer:
    """Test narrative graph layer."""
    
    def test_narrative_graph_layer_forward(self):
        """Test narrative graph layer forward pass."""
        try:
            in_channels = 16
            out_channels = 32
            layer = NarrativeGraphLayer(in_channels, out_channels)
            
            # Create sample graph data
            num_nodes = 5
            x = torch.randn(num_nodes, in_channels)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
            
            output = layer(x, edge_index)
            
            assert output.shape == (num_nodes, out_channels)
            
        except ImportError:
            pytest.skip("PyTorch Geometric not available")
    
    def test_narrative_graph_empty_edges(self):
        """Test with empty edge set."""
        try:
            layer = NarrativeGraphLayer(8, 16)
            
            x = torch.randn(3, 8)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            output = layer(x, edge_index)
            assert output.shape == (3, 16)
            
        except ImportError:
            pytest.skip("PyTorch Geometric not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
