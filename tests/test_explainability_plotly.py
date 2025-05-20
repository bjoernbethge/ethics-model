"""
Test the refactored explainability module with Plotly visualizations.
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

try:
    from src.ethics_model.explainability import (
        AttentionVisualizer,
        TokenContributionAnalyzer,
        EthicsExplainer,
        explain_text,
        create_explanation_dashboard
    )
    from transformers import AutoTokenizer
    import plotly.graph_objects as go
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytest.skip("Required modules not available", allow_module_level=True)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = ["hello", "world", "this", "is", "test"]
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock ethics model for testing."""
    model = MagicMock()
    model.embedding = MagicMock()
    model.embedding.return_value = torch.randn(1, 5, 128)
    
    # Mock model outputs
    mock_outputs = {
        'ethics_score': torch.tensor([[0.8]]),
        'manipulation_score': torch.tensor([[0.2]]),
        'attention_weights': torch.randn(1, 5, 5)
    }
    model.return_value = mock_outputs
    model.side_effect = None
    
    return model


class TestAttentionVisualizer:
    """Test the Plotly-based attention visualizer."""
    
    def test_initialization(self, mock_tokenizer):
        """Test visualizer initialization."""
        visualizer = AttentionVisualizer(mock_tokenizer)
        assert visualizer.tokenizer == mock_tokenizer
    
    def test_visualize_attention_basic(self, mock_tokenizer):
        """Test basic attention visualization."""
        visualizer = AttentionVisualizer(mock_tokenizer)
        
        attention_weights = torch.randn(1, 5, 5)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        fig = visualizer.visualize_attention(
            "hello world this is test",
            attention_weights,
            title="Test Attention"
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert "Test Attention" in fig.layout.title.text
    
    def test_visualize_attention_with_contributions(self, mock_tokenizer):
        """Test attention visualization with token contributions."""
        visualizer = AttentionVisualizer(mock_tokenizer)
        
        attention_weights = torch.randn(1, 5, 5)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        token_contributions = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2])
        
        fig = visualizer.visualize_attention(
            "hello world this is test",
            attention_weights,
            token_contributions=token_contributions
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        # Should have two subplots when contributions are provided
        assert len(fig.data) >= 2
    
    def test_visualize_head_comparison(self, mock_tokenizer):
        """Test multi-head attention comparison."""
        visualizer = AttentionVisualizer(mock_tokenizer)
        
        # Multi-head attention: (batch, heads, seq_len, seq_len)
        multi_head_attention = torch.randn(1, 4, 5, 5)
        multi_head_attention = torch.softmax(multi_head_attention, dim=-1)
        
        fig = visualizer.visualize_head_comparison(
            "hello world this is test",
            multi_head_attention,
            head_labels=["Head 1", "Head 2", "Head 3", "Head 4"]
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        # Should have 4 subplots for 4 heads
        assert len(fig.data) == 4


class TestTokenContributionAnalyzer:
    """Test token contribution analysis."""
    
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test analyzer initialization."""
        analyzer = TokenContributionAnalyzer(mock_model, mock_tokenizer)
        assert analyzer.model == mock_model
        assert analyzer.tokenizer == mock_tokenizer
    
    def test_compute_integrated_gradients(self, mock_model, mock_tokenizer):
        """Test integrated gradients computation."""
        analyzer = TokenContributionAnalyzer(mock_model, mock_tokenizer)
        
        # Mock the embedding layer
        mock_model.embedding.return_value = torch.randn(1, 5, 128, requires_grad=True)
        mock_model.return_value = {'ethics_score': torch.tensor([[0.8]])}
        
        # Run integrated gradients
        with patch('torch.autograd.grad') as mock_grad:
            mock_grad.return_value = [torch.randn(1, 5, 128)]
            
            attributions = analyzer.compute_integrated_gradients("test text")
            
            assert attributions is not None
            assert attributions.shape == (1, 5)  # batch_size, seq_len
    
    def test_analyze_token_importance(self, mock_model, mock_tokenizer):
        """Test token importance analysis."""
        analyzer = TokenContributionAnalyzer(mock_model, mock_tokenizer)
        
        # Mock the integrated gradients method
        with patch.object(analyzer, 'compute_integrated_gradients') as mock_ig:
            mock_ig.return_value = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.2]])
            
            analysis = analyzer.analyze_token_importance("test text")
            
            assert "tokens" in analysis
            assert "importance_scores" in analysis
            assert "token_importance" in analysis
            assert "method" in analysis
            assert analysis["method"] == "integrated_gradients"
            
            # Check that tokens are sorted by importance
            token_importance = analysis["token_importance"]
            assert len(token_importance) == 5
            assert all("token" in item and "importance" in item and "abs_importance" in item 
                      for item in token_importance)


class TestEthicsExplainer:
    """Test the comprehensive ethics explainer."""
    
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test explainer initialization."""
        with patch('spacy.load') as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.return_value = mock_nlp
            
            explainer = EthicsExplainer(mock_model, mock_tokenizer)
            
            assert explainer.model == mock_model
            assert explainer.tokenizer == mock_tokenizer
            assert explainer.nlp == mock_nlp
    
    def test_explain_prediction_basic(self, mock_model, mock_tokenizer):
        """Test basic prediction explanation."""
        with patch('spacy.load') as mock_spacy:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_doc.sents = [MagicMock()]
            mock_doc.__iter__ = lambda self: iter([])
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            explainer = EthicsExplainer(mock_model, mock_tokenizer)
            
            # Mock the token analyzer
            with patch.object(explainer.token_analyzer, 'analyze_token_importance') as mock_analyze:
                mock_analyze.return_value = {
                    "tokens": ["test"],
                    "importance_scores": [0.5],
                    "token_importance": [{"token": "test", "importance": 0.5, "abs_importance": 0.5}],
                    "method": "integrated_gradients"
                }
                
                explanation = explainer.explain_prediction("test text")
                
                assert "input_text" in explanation
                assert "model_prediction" in explanation
                assert "confidence" in explanation
                assert "explanations" in explanation
                assert "natural_language_explanation" in explanation
                
                assert explanation["input_text"] == "test text"
                assert explanation["model_prediction"] == "ethical"  # Because mock returns 0.8
    
    def test_create_token_importance_plot(self, mock_model, mock_tokenizer):
        """Test token importance plot creation."""
        with patch('spacy.load'):
            explainer = EthicsExplainer(mock_model, mock_tokenizer)
            
            token_analysis = {
                "tokens": ["hello", "world", "test"],
                "importance_scores": [0.1, -0.2, 0.3]
            }
            
            fig = explainer._create_token_importance_plot(token_analysis)
            
            assert fig is not None
            assert isinstance(fig, go.Figure)
            assert len(fig.data) == 1  # One bar chart
            assert fig.data[0].type == "bar"
    
    def test_analyze_linguistic_features(self, mock_model, mock_tokenizer):
        """Test linguistic feature analysis."""
        with patch('spacy.load') as mock_spacy:
            # Create mock spaCy objects
            mock_token = MagicMock()
            mock_token.text = "good"
            mock_token.pos_ = "ADJ"
            
            mock_ent = MagicMock()
            mock_ent.text = "John"
            mock_ent.label_ = "PERSON"
            
            mock_sent = MagicMock()
            
            mock_doc = MagicMock()
            mock_doc.ents = [mock_ent]
            mock_doc.sents = [mock_sent]
            mock_doc.__iter__ = lambda self: iter([mock_token])
            mock_doc.__len__ = lambda self: 1
            
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            explainer = EthicsExplainer(mock_model, mock_tokenizer)
            
            analysis = explainer._analyze_linguistic_features("This is good text")
            
            assert "entities" in analysis
            assert "sentiment_indicators" in analysis
            assert "moral_language" in analysis
            assert "complexity_metrics" in analysis
            
            # Check that the mock entity was captured
            assert len(analysis["entities"]) == 1
            assert analysis["entities"][0]["text"] == "John"
            assert analysis["entities"][0]["label"] == "PERSON"
    
    def test_generate_natural_explanation(self, mock_model, mock_tokenizer):
        """Test natural language explanation generation."""
        with patch('spacy.load'):
            explainer = EthicsExplainer(mock_model, mock_tokenizer)
            
            explanation_data = {
                "input_text": "test text",
                "model_prediction": "ethical",
                "confidence": 0.85,
                "manipulation_risk": 0.3,
                "explanations": {
                    "token_contributions": {
                        "token_importance": [
                            {"token": "good", "importance": 0.5},
                            {"token": "test", "importance": 0.3}
                        ]
                    },
                    "linguistic": {
                        "moral_language": [{"word": "good", "category": "positive"}],
                        "entities": [{"text": "John", "label": "PERSON"}]
                    }
                }
            }
            
            explanation = explainer._generate_natural_explanation(explanation_data)
            
            assert isinstance(explanation, str)
            assert "ethical" in explanation
            assert "85.0%" in explanation or "0.85" in explanation


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_explain_text(self, mock_model, mock_tokenizer):
        """Test the quick explain_text function."""
        with patch('spacy.load'):
            with patch('src.ethics_model.explainability.EthicsExplainer') as mock_explainer_class:
                mock_explainer = MagicMock()
                mock_explainer.explain_prediction.return_value = {"test": "result"}
                mock_explainer_class.return_value = mock_explainer
                
                result = explain_text("test text", mock_model, mock_tokenizer)
                
                assert result == {"test": "result"}
                mock_explainer.explain_prediction.assert_called_once_with("test text", save_path=None)
    
    def test_create_explanation_dashboard(self, mock_model, mock_tokenizer):
        """Test dashboard creation."""
        with patch('spacy.load'):
            with patch('src.ethics_model.explainability.EthicsExplainer') as mock_explainer_class:
                mock_explainer = MagicMock()
                mock_explainer.explain_prediction.return_value = {
                    "model_prediction": "ethical",
                    "confidence": 0.8
                }
                mock_explainer_class.return_value = mock_explainer
                
                texts = ["text1", "text2", "text3"]
                dashboard = create_explanation_dashboard(texts, mock_model, mock_tokenizer)
                
                assert dashboard is not None
                assert isinstance(dashboard, go.Figure)
                # Should have multiple subplots
                assert len(dashboard.data) >= 4


if __name__ == "__main__":
    pytest.main([__file__])
