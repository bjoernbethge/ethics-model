"""
Shared test configuration and fixtures for the ethics model test suite.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Tuple

try:
    import torch
except ImportError:
    # Installiere torch falls es nicht vorhanden ist
    import subprocess
    subprocess.check_call(["uv", "pip", "install", "torch"])


# ============================================================================
# Mock Classes
# ============================================================================

class MockTokenizer:
    """Mock tokenizer for consistent testing across all modules."""
    
    def __init__(self, vocab_size: int = 1000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        max_len = kwargs.get('max_length', self.max_length)
        
        # Simple word-based tokenization
        words = text.split()[:max_len]
        input_ids = [hash(word) % self.vocab_size for word in words]
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        while len(input_ids) < max_len:
            input_ids.append(0)
            attention_mask.append(0)
        
        # Truncate if necessary
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }


class MockLLM:
    """Mock language model for testing."""
    
    def __init__(self, d_model: int = 512, vocab_size: int = 1000):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Create mock transformer
        self.transformer = MagicMock()
        self.model = MagicMock()
        self.model.transformer = self.transformer
    
    def __call__(self, input_ids: torch.Tensor) -> MagicMock:
        batch_size, seq_len = input_ids.shape
        
        # Create mock output with last_hidden_state
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, self.d_model)
        
        self.transformer.return_value = mock_output
        return mock_output


class MockSpacyModel:
    """Mock spaCy model for testing."""
    
    def __init__(self):
        self.vocab = MagicMock()
        
    def __call__(self, text: str) -> MagicMock:
        mock_doc = MagicMock()
        mock_doc.text = text
        mock_doc.ents = []
        mock_doc.sents = [MagicMock(text=text)]
        mock_doc.has_annotation = lambda x: True
        return mock_doc