"""
Test the refactored explainability module with NetworkX and spaCy.
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

try:
    from src.ethics_model.explainability import (
        AttentionVisualizer,
        GraphExplainer,
        EthicsExplainer,
        create_explainer,
        quick_explain
    )
    from transformers import AutoTokenizer
    import plotly.graph_objects as go
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def tokenize(self, text):
        return text.split()
    
    def convert_ids_to_tokens(self, ids):
        return [f"token_{i}" for i in ids]
    
    def __call__(self, text, **kwargs):
        tokens = text.split()
        return {
            'input_ids': list(range(len(tokens))),
            'attention_mask': [1] * len(tokens)
        }


class TestAttentionVisualizer:
    """Test attention visualization components."""
    
    def test_initialization(self):
        """Test attention visualizer initialization."""
        tokenizer = MockTokenizer()
        viz = AttentionVisualizer(tokenizer)
        assert viz.tokenizer == tokenizer
    
    def test_visualize_attention(self):
        """Test attention visualization."""
        tokenizer = MockTokenizer()
        viz = AttentionVisualizer(tokenizer)
        
        text = "This is a test"
        attention_weights = torch.randn(1, 4, 4)
        
        fig = viz.visualize_attention(text, attention_weights)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Ethics Model Attention Weights"
    
    def test_visualize_attention_with_contributions(self):
        """Test attention visualization with token contributions."""
        tokenizer = MockTokenizer()
        viz = AttentionVisualizer(tokenizer)
        
        text = "This is a test"
        attention_weights = torch.randn(1, 4, 4)
        token_contributions = torch.randn(1, 4)
        
        fig = viz.visualize_attention(text, attention_weights, token_contributions)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 1  # Should have attention heatmap + bar chart
    
    def test_visualize_token_attributions(self):
        """Test token attribution visualization."""
        tokenizer = MockTokenizer()
        viz = AttentionVisualizer(tokenizer)
        
        text = "This is a test"
        attributions = torch.randn(1, 4)
        ethics_score = 0.8
        manipulation_score = 0.2
        
        fig = viz.visualize_token_attributions(
            text, attributions, ethics_score, manipulation_score
        )
        
        assert isinstance(fig, go.Figure)
        # Check that scores are displayed as annotations
        assert len(fig.layout.annotations) >= 2
    
    def test_visualize_attention_head_view(self):
        """Test multi-head attention visualization."""
        tokenizer = MockTokenizer()
        viz = AttentionVisualizer(tokenizer)
        
        text = "This is a test"
        # Multi-head attention: (batch, heads, seq, seq)
        attention_weights = torch.randn(1, 2, 4, 4)
        
        fig = viz.visualize_attention_head_view(text, attention_weights)
        
        assert isinstance(fig, go.Figure)


class TestGraphExplainer:
    """Test graph-based explanation components."""
    
    def test_initialization(self):
        """Test graph explainer initialization."""
        with patch('spacy.load') as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            
            explainer = GraphExplainer("en_core_web_sm")
            assert explainer.nlp == mock_nlp
    
    def test_initialization_fallback(self):
        """Test fallback when spaCy model not available."""
        with patch('spacy.load', side_effect=OSError("Model not found")):
            with patch('spacy.blank') as mock_blank:
                mock_nlp = MagicMock()
                mock_blank.return_value = mock_nlp
                
                explainer = GraphExplainer("en_core_web_sm")
                assert explainer.nlp == mock_nlp
    
    def test_categorize_entity(self):
        """Test entity categorization."""
        explainer = GraphExplainer()
        
        # Test moral value
        category, relevance = explainer._categorize_entity("fairness")
        assert category == "moral_value"
        assert relevance > 0.5
        
        # Test positive action
        category, relevance = explainer._categorize_entity("help")
        assert category == "positive_action"
        assert relevance > 0.5
        
        # Test negative action
        category, relevance = explainer._categorize_entity("harm")
        assert category == "negative_action"
        assert relevance > 0.5
        
        # Test stakeholder
        category, relevance = explainer._categorize_entity("people")
        assert category == "stakeholder"
        assert relevance > 0.5
        
        # Test neutral entity
        category, relevance = explainer._categorize_entity("table", ent_type="NOUN")
        assert category in ["entity", "other"]
        assert relevance <= 0.6
    
    @patch('spacy.load')
    def test_build_ethical_graph(self, mock_load):
        """Test ethical graph construction."""
        # Mock spaCy document
        mock_token1 = MagicMock()
        mock_token1.text = "John"
        mock_token1.pos_ = "NOUN"
        mock_token1.is_stop = False
        mock_token1.is_punct = False
        mock_token1.head.text = "help"
        mock_token1.dep_ = "nsubj"
        
        mock_token2 = MagicMock()
        mock_token2.text = "help"
        mock_token2.pos_ = "VERB"
        mock_token2.is_stop = False
        mock_token2.is_punct = False
        mock_token2.head = mock_token2  # Self-reference for head
        mock_token2.dep_ = "ROOT"
        
        mock_ent = MagicMock()
        mock_ent.text = "John"
        mock_ent.label_ = "PERSON"
        
        mock_sent = MagicMock()
        mock_sent.__iter__ = lambda self: iter([mock_token1, mock_token2])
        
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_token1, mock_token2])
        mock_doc.ents = [mock_ent]
        mock_doc.sents = [mock_sent]
        
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        explainer = GraphExplainer()
        graph = explainer.build_ethical_graph("John helps people")
        
        assert graph.number_of_nodes() > 0
        # Should have nodes for John and help
        node_texts = [data.get('text', '') for _, data in graph.nodes(data=True)]
        assert "John" in node_texts
        assert "help" in node_texts
    
    @patch('spacy.load')
    def test_extract_ethical_entities(self, mock_load):
        """Test extraction of ethical entities."""
        # Mock spaCy processing
        mock_token1 = MagicMock()
        mock_token1.text = "fairness"
        mock_token1.is_stop = False
        mock_token1.is_punct = False
        mock_token1.pos_ = "NOUN"
        
        mock_token2 = MagicMock()
        mock_token2.text = "help"
        mock_token2.is_stop = False
        mock_token2.is_punct = False
        mock_token2.pos_ = "VERB"
        
        mock_ent = MagicMock()
        mock_ent.text = "John"
        mock_ent.label_ = "PERSON"
        
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_token1, mock_token2])
        mock_doc.ents = [mock_ent]
        
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        explainer = GraphExplainer()
        entities = explainer.extract_ethical_entities("John helps with fairness")
        
        assert "actors" in entities
        assert "actions" in entities
        assert "values" in entities
        assert "John" in entities["actors"]
        assert "help" in entities["actions"]
        assert "fairness" in entities["values"]
    
    @patch('spacy.load')
    def test_visualize_ethical_graph(self, mock_load):
        """Test graph visualization."""
        # Mock empty document to test empty graph case
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([])
        mock_doc.ents = []
        mock_doc.sents = []
        
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        explainer = GraphExplainer()
        fig = explainer.visualize_ethical_graph("Empty text")
        
        assert isinstance(fig, go.Figure)
        # Should show "No ethical relationships detected" message
        assert len(fig.layout.annotations) > 0
    
    @patch('spacy.load')
    def test_analyze_ethical_patterns(self, mock_load):
        """Test ethical pattern analysis."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([])
        mock_doc.ents = []
        mock_doc.sents = []
        
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        explainer = GraphExplainer()
        patterns = explainer.analyze_ethical_patterns("Empty text")
        
        assert "error" in patterns
        assert patterns["error"] == "No ethical entities detected"


class TestEthicsExplainer:
    """Test the comprehensive ethics explainer."""
    
    @patch('spacy.load')
    def test_initialization(self, mock_load):
        """Test explainer initialization."""
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        mock_model = MagicMock()
        mock_tokenizer = MockTokenizer()
        
        explainer = EthicsExplainer(mock_model, mock_tokenizer)
        
        assert explainer.model == mock_model
        assert explainer.tokenizer == mock_tokenizer
        assert isinstance(explainer.attention_viz, AttentionVisualizer)
        assert isinstance(explainer.graph_explainer, GraphExplainer)
    
    @patch('spacy.load')
    def test_compute_attributions(self, mock_load):
        """Test attribution computation."""
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        # Mock model outputs
        mock_outputs = {
            "ethics_score": torch.tensor([[0.8]]),
            "manipulation_score": torch.tensor([[0.2]])
        }
        
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        
        # Mock LLM
        mock_llm_outputs = MagicMock()
        mock_llm_outputs.last_hidden_state = torch.randn(1, 4, 512)
        
        mock_llm = MagicMock()
        mock_llm.transformer.return_value = mock_llm_outputs
        
        mock_tokenizer = MockTokenizer()
        
        explainer = EthicsExplainer(mock_model, mock_tokenizer)
        
        outputs, attributions = explainer.compute_attributions("Test text", mock_llm)
        
        assert "ethics_score" in outputs
        assert "manipulation_score" in outputs
        assert attributions.shape[0] == 1  # Batch size
    
    @patch('spacy.load')
    def test_explain_with_error_handling(self, mock_load):
        """Test explanation with error handling."""
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        # Mock model that raises exception
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Test error")
        
        mock_tokenizer = MockTokenizer()
        mock_llm = MagicMock()
        
        explainer = EthicsExplainer(mock_model, mock_tokenizer)
        explanation = explainer.explain("Test text", mock_llm)
        
        assert "error" in explanation
        assert explanation["text"] == "Test text"
        assert explanation["ethics_score"] == 0.5  # Default fallback


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('spacy.load')
    def test_create_explainer(self, mock_load):
        """Test explainer factory function."""
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        mock_model = MagicMock()
        mock_tokenizer = MockTokenizer()
        
        explainer = create_explainer(mock_model, mock_tokenizer)
        
        assert isinstance(explainer, EthicsExplainer)
        assert explainer.model == mock_model
        assert explainer.tokenizer == mock_tokenizer
    
    @patch('spacy.load')
    def test_quick_explain(self, mock_load):
        """Test quick explanation function."""
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        # Mock successful model outputs
        mock_outputs = {
            "ethics_score": torch.tensor([[0.8]]),
            "manipulation_score": torch.tensor([[0.2]]),
            "attention_weights": torch.randn(1, 4, 4)
        }
        
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        
        # Mock LLM
        mock_llm_outputs = MagicMock()
        mock_llm_outputs.last_hidden_state = torch.randn(1, 4, 512)
        
        mock_llm = MagicMock()
        mock_llm.transformer.return_value = mock_llm_outputs
        
        mock_tokenizer = MockTokenizer()
        
        explanation = quick_explain("Test text", mock_model, mock_llm, mock_tokenizer)
        
        assert "text" in explanation
        assert "ethics_score" in explanation
        assert explanation["text"] == "Test text"


class TestIntegration:
    """Test integration between components."""
    
    @patch('spacy.load')
    def test_full_pipeline(self, mock_load):
        """Test the full explanation pipeline."""
        # Mock spaCy
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([])
        mock_doc.ents = []
        mock_doc.sents = []
        
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        # Mock successful model
        mock_outputs = {
            "ethics_score": torch.tensor([[0.8]]),
            "manipulation_score": torch.tensor([[0.2]]),
            "attention_weights": torch.randn(1, 4, 4),
            "framework_analysis": {
                "framework_outputs": {
                    "deontological": torch.tensor([0.7]),
                    "utilitarian": torch.tensor([0.3])
                }
            },
            "framing_analysis": {
                "framing_strength": torch.tensor([0.5])
            },
            "manipulation_analysis": {
                "technique_scores": {
                    "emotional_appeal": torch.tensor([0.3]),
                    "false_dichotomy": torch.tensor([0.1])
                }
            }
        }
        
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        
        # Mock LLM
        mock_embeddings = torch.randn(1, 4, 512)
        mock_embeddings.requires_grad_(True)
        
        mock_llm_outputs = MagicMock()
        mock_llm_outputs.last_hidden_state = mock_embeddings
        
        mock_llm = MagicMock()
        mock_llm.transformer.return_value = mock_llm_outputs
        
        mock_tokenizer = MockTokenizer()
        
        explainer = EthicsExplainer(mock_model, mock_tokenizer)
        explanation = explainer.explain("This is a test of ethical analysis", mock_llm)
        
        # Check all components are present
        assert "text" in explanation
        assert "ethics_score" in explanation
        assert "manipulation_score" in explanation
        assert "attention_visualization" in explanation
        assert "token_attribution_visualization" in explanation
        assert "graph_visualization" in explanation
        assert "ethical_entities" in explanation
        assert "ethical_patterns" in explanation
        assert "framework_analysis" in explanation
        assert "narrative_analysis" in explanation
        
        # Check values
        assert explanation["ethics_score"] == 0.8
        assert explanation["manipulation_score"] == 0.2
        
        # Check framework analysis
        assert "dominant_framework" in explanation["framework_analysis"]
        assert "framework_scores" in explanation["framework_analysis"]
        
        # Check narrative analysis
        assert "framing_strength" in explanation["narrative_analysis"]
        assert "manipulation_techniques" in explanation["narrative_analysis"]


if __name__ == "__main__":
    pytest.main([__file__])
