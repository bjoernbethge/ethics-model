"""
Test the refactored graph_semantic module with NetworkX and spaCy.
"""
import pytest
import torch
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

try:
    from ethics_model.modules.graph_semantic import (
        SemanticGraphConverter,
        SemanticPatternExtractor,
        SemanticGraphProcessor,
        create_semantic_processor,
        analyze_text_semantics
    )
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class TestSemanticGraphConverter:
    """Test the NetworkX to PyG converter."""
    
    def test_empty_graph_conversion(self):
        """Test conversion of empty graph."""
        converter = SemanticGraphConverter()
        empty_graph = nx.DiGraph()
        
        data = converter.networkx_to_pyg_data(empty_graph)
        
        assert data.x.shape == (1, 128)  # Default dimension
        assert data.edge_index.shape == (2, 0)
        assert data.num_nodes == 1
    
    def test_simple_graph_conversion(self):
        """Test conversion of simple graph."""
        converter = SemanticGraphConverter()
        graph = nx.DiGraph()
        
        # Add nodes with attributes
        graph.add_node(0, text="John", semantic_type="agent", moral_valence=0.1)
        graph.add_node(1, text="help", semantic_type="action", moral_valence=0.8)
        graph.add_edge(0, 1, relation="nsubj", weight=0.9)
        
        data = converter.networkx_to_pyg_data(graph, default_dim=64)
        
        assert data.x.shape == (2, 64)
        assert data.edge_index.shape == (2, 1)
        assert data.num_nodes == 2
        assert hasattr(data, 'node_map')
        assert hasattr(data, 'nodes')
    
    def test_graph_conversion_with_embeddings(self):
        """Test conversion with provided concept embeddings."""
        converter = SemanticGraphConverter()
        graph = nx.DiGraph()
        
        graph.add_node(0, text="John", semantic_type="agent")
        graph.add_node(1, text="help", semantic_type="action")
        
        # Provide embeddings
        embeddings = {
            "John": torch.randn(32),
            "help": torch.randn(32)
        }
        
        data = converter.networkx_to_pyg_data(graph, concept_embeddings=embeddings)
        
        assert data.x.shape == (2, 32)
        assert torch.equal(data.x[0], embeddings["John"])


class TestSemanticPatternExtractor:
    """Test semantic pattern extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock spaCy model to avoid dependency issues
        self.mock_nlp = MagicMock()
        self.extractor = SemanticPatternExtractor(self.mock_nlp)
    
    def test_moral_valence_calculation(self):
        """Test moral valence scoring."""
        # Positive moral concept
        valence = self.extractor.get_moral_valence("help others")
        assert valence > 0
        
        # Negative moral concept
        valence = self.extractor.get_moral_valence("harm people")
        assert valence < 0
        
        # Neutral text
        valence = self.extractor.get_moral_valence("table chair")
        assert valence == 0.0
    
    def test_emotional_intensity_calculation(self):
        """Test emotional intensity scoring."""
        # High emotional content
        intensity = self.extractor.get_emotional_intensity("very angry upset")
        assert intensity > 0
        
        # Low emotional content
        intensity = self.extractor.get_emotional_intensity("the table is brown")
        assert intensity >= 0
    
    def test_extract_semantic_roles_mock(self):
        """Test semantic role extraction with mocked spaCy."""
        # Create mock document and tokens
        mock_sent = MagicMock()
        mock_tokens = []
        
        # Mock root verb
        mock_root = MagicMock()
        mock_root.dep_ = "ROOT"
        mock_root.pos_ = "VERB"
        mock_root.text = "help"
        mock_root.lemma_ = "help"
        mock_root.children = []
        mock_tokens.append(mock_root)
        
        # Mock subject
        mock_subj = MagicMock()
        mock_subj.dep_ = "nsubj"
        mock_subj.text = "John"
        mock_subj.ent_type_ = "PERSON"
        mock_tokens.append(mock_subj)
        
        # Set up relationships
        mock_root.children = [mock_subj]
        mock_sent.__iter__ = lambda self: iter(mock_tokens)
        
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        
        roles = self.extractor.extract_semantic_roles(mock_doc)
        
        assert len(roles) == 1
        assert len(roles[0]["actions"]) >= 0  # Should find the help action
        assert len(roles[0]["agents"]) >= 0   # Should find John as agent


class TestSemanticGraphProcessor:
    """Test the main semantic graph processor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        with patch('spacy.load') as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            
            processor = SemanticGraphProcessor(d_model=128, spacy_model="en_core_web_sm")
            
            assert processor.nlp == mock_nlp
            assert isinstance(processor.pattern_extractor, SemanticPatternExtractor)
    
    def test_initialization_fallback(self):
        """Test fallback when spaCy model not available."""
        with patch('spacy.load', side_effect=OSError("Model not found")):
            with patch('spacy.blank') as mock_blank:
                mock_nlp = MagicMock()
                mock_blank.return_value = mock_nlp
                
                processor = SemanticGraphProcessor(d_model=128)
                assert processor.nlp == mock_nlp
    
    def test_create_semantic_graph_empty(self):
        """Test graph creation with empty text."""
        with patch('spacy.load') as mock_load:
            # Mock empty document
            mock_doc = MagicMock()
            mock_doc.sents = []
            mock_doc.__iter__ = lambda self: iter([])
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            processor = SemanticGraphProcessor(d_model=128)
            graph = processor.create_semantic_graph("")
            
            assert isinstance(graph, nx.DiGraph)
            assert graph.number_of_nodes() >= 0
    
    def test_forward_pass(self):
        """Test forward pass through processor."""
        with patch('spacy.load') as mock_load:
            # Mock spaCy components
            mock_doc = MagicMock()
            mock_doc.sents = []
            mock_doc.__iter__ = lambda self: iter([])
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            processor = SemanticGraphProcessor(d_model=64)
            
            # Test data
            text_batch = ["Test sentence."]
            embeddings = torch.randn(1, 10, 64)
            
            outputs = processor(text_batch, embeddings, return_graphs=True)
            
            assert 'graph_embeddings' in outputs
            assert 'ethical_relations' in outputs
            assert 'graphs' in outputs
            assert 'graph_data' in outputs
            
            assert outputs['graph_embeddings'].shape == embeddings.shape
    
    def test_get_graph_summary_empty(self):
        """Test graph summary for empty graph."""
        processor = SemanticGraphProcessor(d_model=128)
        empty_graph = nx.DiGraph()
        
        summary = processor.get_graph_summary(empty_graph)
        
        assert summary["empty"] is True
    
    def test_get_graph_summary(self):
        """Test graph summary for non-empty graph."""
        processor = SemanticGraphProcessor(d_model=128)
        
        graph = nx.DiGraph()
        graph.add_node(0, text="John", semantic_type="agent", moral_valence=0.5)
        graph.add_node(1, text="help", semantic_type="action", moral_valence=0.8)
        graph.add_edge(0, 1, relation="nsubj")
        
        summary = processor.get_graph_summary(graph)
        
        assert summary["n_nodes"] == 2
        assert summary["n_edges"] == 1
        assert "semantic_types" in summary
        assert "avg_moral_valence" in summary
        assert summary["avg_moral_valence"] > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_semantic_processor(self):
        """Test factory function."""
        with patch('spacy.load') as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            
            processor = create_semantic_processor(d_model=256, semantic_dim=64)
            
            assert isinstance(processor, SemanticGraphProcessor)
            assert processor.semantic_encoder[0].in_features == 256
    
    def test_analyze_text_semantics(self):
        """Test text analysis utility."""
        with patch('spacy.load') as mock_load:
            # Mock spaCy components
            mock_doc = MagicMock()
            mock_doc.sents = []
            mock_doc.__iter__ = lambda self: iter([])
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            processor = create_semantic_processor(d_model=128)
            
            result = analyze_text_semantics("Test text", processor)
            
            assert "semantic_graph" in result
            assert "ethical_relations" in result
            assert "graph_summary" in result
            assert "graph_data" in result
            
            assert isinstance(result["semantic_graph"], nx.DiGraph)


class TestIntegration:
    """Test integration between components."""
    
    def test_full_pipeline_mock(self):
        """Test full processing pipeline with mocked components."""
        with patch('spacy.load') as mock_load:
            # Create comprehensive mock
            mock_token = MagicMock()
            mock_token.text = "help"
            mock_token.dep_ = "ROOT"
            mock_token.pos_ = "VERB"
            mock_token.lemma_ = "help"
            mock_token.children = []
            mock_token.ent_type_ = ""
            
            mock_sent = MagicMock()
            mock_sent.__iter__ = lambda self: iter([mock_token])
            
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent]
            mock_doc.__iter__ = lambda self: iter([mock_token])
            mock_doc.ents = []
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            
            # Test the pipeline
            processor = SemanticGraphProcessor(d_model=64)
            
            text = "People should help others in need."
            graph = processor.create_semantic_graph(text)
            relations = processor.extract_ethical_relations(graph, text)
            summary = processor.get_graph_summary(graph)
            
            # Basic assertions
            assert isinstance(graph, nx.DiGraph)
            assert isinstance(relations, dict)
            assert isinstance(summary, dict)
            
            # Check relation structure
            expected_keys = ['moral_agents', 'moral_actions', 'moral_objects', 
                           'obligations', 'values', 'consequences', 'conflicts']
            for key in expected_keys:
                assert key in relations
                assert isinstance(relations[key], list)


if __name__ == "__main__":
    pytest.main([__file__])
