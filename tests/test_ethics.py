"""
Test main ethics model.
"""
import torch
import pytest
from unittest.mock import patch, MagicMock

try:
    from ethics_model.modules.graph_semantic import SemanticGraphProcessor
    from ethics_model.model import (
        EnhancedEthicsModel,
        EthicsModel,
        create_ethics_model
    )
except ImportError:
    pytest.skip("Ethics model not available", allow_module_level=True)


class TestEthicsModel:
    """Test the main ethics model."""
    
    def test_ethics_model_forward(self):
        """Test ethics model forward pass."""
        config = {
            'input_dim': 256,
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check required outputs
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
        assert 'framework_analysis' in outputs
        assert 'intuition_analysis' in outputs
        assert 'dual_process_analysis' in outputs
        assert 'narrative_analysis' in outputs
        assert 'framing_analysis' in outputs
        assert 'dissonance_analysis' in outputs
        assert 'manipulation_analysis' in outputs
        assert 'propaganda_analysis' in outputs
        assert 'attention_weights' in outputs
        assert 'hidden_states' in outputs
        assert 'meta_cognitive_features' in outputs
        
        # Check shapes
        assert outputs['ethics_score'].shape == (batch_size, 1)
        assert outputs['manipulation_score'].shape == (batch_size, 1)
        assert outputs['hidden_states'].shape == (batch_size, seq_len, 256)
    
    def test_ethics_model_with_embeddings(self):
        """Test model with pre-computed embeddings."""
        config = {
            'input_dim': 128,
            'd_model': 128,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 500,
            'max_seq_length': 16,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        batch_size = 1
        seq_len = 16
        embeddings = torch.randn(batch_size, seq_len, 128)
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(embeddings=embeddings, attention_mask=attention_mask)
        
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
        assert outputs['ethics_score'].shape == (batch_size, 1)
    
    def test_ethics_model_with_texts(self):
        """Test model with text inputs (requires spaCy)."""
        config = {
            'input_dim': 128,
            'd_model': 128,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 500,
            'max_seq_length': 16,
            'use_semantic_graphs': True,  # Enable semantic graphs
            'use_gnn': False
        }
        
        # Patche die nötigen imports
        try:
            import spacy
            from spacy.pipeline import Sentencizer
            
            # Echte Tests mit dem deutschen SpaCy-Modell
            # Überschreibe den Namen des Spacy-Modells im Config
            config['spacy_model'] = 'de_core_news_sm'
            
            model = create_ethics_model(config)
            
            batch_size = 2
            seq_len = 16
            input_ids = torch.randint(0, 500, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            texts = ["Das ist gut", "Das ist schlecht"]
                
            # Das Modell sollte fehlende spaCy-Funktionalität behandeln
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts
            )
            
            assert 'ethics_score' in outputs
            assert 'manipulation_score' in outputs
        
        except (ImportError, OSError):
            # Wenn semantische Graph-Module nicht verfügbar sind, überspringen
            pytest.skip("Semantic graph modules not available")
    
    def test_model_summary(self):
        """Test ethical summary generation."""
        config = {
            'input_dim': 64,
            'd_model': 64,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 100,
            'max_seq_length': 8,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones(1, 8)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        summary = model.get_ethical_summary(outputs)
        
        assert isinstance(summary, dict)
        assert 'overall_ethics_score' in summary
        assert 'manipulation_risk' in summary
        assert 'dominant_framework' in summary
        assert 'emotional_intensity' in summary
        assert 'system_conflict' in summary
        assert 'main_manipulation_techniques' in summary
        assert 'cognitive_dissonance_level' in summary
        assert 'framing_strength' in summary
        assert 'propaganda_risk' in summary
        
        # Check value ranges
        assert 0 <= summary['overall_ethics_score'] <= 1
        assert 0 <= summary['manipulation_risk'] <= 1
        assert isinstance(summary['dominant_framework'], str)
        assert isinstance(summary['main_manipulation_techniques'], list)


class TestModelVariants:
    """Test different model variants."""
    
    def test_legacy_model(self):
        """Test legacy model creation."""
        config = {
            'input_dim': 64,
            'd_model': 64,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 100,
            'max_seq_length': 8,
            'use_legacy': True
        }
        
        model = create_ethics_model(config)
        assert isinstance(model, EthicsModel)
        
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones(1, 8)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
    
    def test_enhanced_model(self):
        """Test enhanced model creation."""
        config = {
            'input_dim': 64,
            'd_model': 64,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 100,
            'max_seq_length': 8,
            'use_enhanced': True,
            'use_semantic_graphs': False
        }
        
        model = create_ethics_model(config)
        assert isinstance(model, EnhancedEthicsModel)
        
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones(1, 8)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
    
    def test_model_with_gnn(self):
        """Test model with graph neural network."""
        try:
            import torch_geometric
            from torch_geometric.data import Data, Batch
            
            model_config = {
                'input_dim': 64,
                'd_model': 64,
                'n_layers': 1,
                'n_heads': 2,
                'vocab_size': 100,
                'max_seq_length': 8,
                'use_gnn': True,
                'use_dynamic_graphs': True,
                'spacy_model': 'de_core_news_sm'  # Verwende deutsches Modell
            }
            
            model = create_ethics_model(model_config)
            
            batch_size = 2
            input_ids = torch.randint(0, 100, (batch_size, 8))
            attention_mask = torch.ones(batch_size, 8)
            
            # Erstelle eine korrekte Liste für graph_data
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            x = torch.randn(3, 64)
            graph_data = [{'edge_index': edge_index, 'x': x}]
            
            # Führe das Modell aus
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=graph_data
            )
            
            assert 'ethics_score' in outputs
            assert 'manipulation_score' in outputs
        
        except ImportError:
            pytest.skip("PyTorch Geometric not available")


class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_inputs(self):
        """Test with minimal inputs."""
        config = {
            'input_dim': 32,
            'd_model': 32,
            'n_layers': 1,
            'n_heads': 1,
            'vocab_size': 10,
            'max_seq_length': 4,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        # Minimal input
        input_ids = torch.randint(0, 10, (1, 4))
        attention_mask = torch.ones(1, 4)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
        assert not torch.isnan(outputs['ethics_score']).any()
        assert not torch.isnan(outputs['manipulation_score']).any()
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        config = {
            'input_dim': 32,
            'd_model': 32,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 50,
            'max_seq_length': 20,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        for seq_len in [1, 5, 10, 20]:
            input_ids = torch.randint(0, 50, (1, seq_len))
            attention_mask = torch.ones(1, seq_len)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            assert outputs['ethics_score'].shape == (1, 1)
            assert outputs['manipulation_score'].shape == (1, 1)
    
    def test_batch_sizes(self):
        """Test with different batch sizes."""
        config = {
            'input_dim': 32,
            'd_model': 32,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 50,
            'max_seq_length': 8,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, 50, (batch_size, 8))
            attention_mask = torch.ones(batch_size, 8)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            assert outputs['ethics_score'].shape == (batch_size, 1)
            assert outputs['manipulation_score'].shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
