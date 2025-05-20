import torch
import pytest
import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ethics_model.data import MultiTaskDataset
from ethics_model.training import train
from ethics_model.model import EthicsModel
from unittest.mock import MagicMock


class TestLLMTraining(unittest.TestCase):
    
    def test_train_llm_short(self):
        """Test short LLM training with minimal setup."""
        # Mini-Datensatz
        texts = ["Hallo Welt", "Test", "Noch ein Text", "Letzter"]
        ethics_labels = [1.0, 0.0, 1.0, 0.0]
        manipulation_labels = [0.0, 1.0, 0.0, 1.0]
        
        try:
            # Verwende einen echten Tokenizer, aber immer noch einen Mock für das LLM
            try:
                # Versuche einen kleinen Tokenizer zu laden
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                # Fallback zur Mock-Version wenn das Laden fehlschlägt
                tokenizer = MagicMock()
                tokenizer.vocab_size = 1000
                tokenizer.pad_token = "[PAD]"
                tokenizer.eos_token = "[EOS]"
                tokenizer.__call__ = lambda text, **kwargs: {
                    "input_ids": torch.ones((1, min(len(text.split()), 16)), dtype=torch.long),
                    "attention_mask": torch.ones((1, min(len(text.split()), 16)))
                }
            
            # Mock LLM - hier weiterhin ein Mock, da echte LLMs zu groß sind
            llm = MagicMock()
            llm.config = MagicMock()
            llm.config.hidden_size = 32
            
            # Mock für forward mit besserer Dimensionalität
            def mock_forward(**kwargs):
                batch_size = kwargs.get('input_ids', torch.ones(1, 16)).shape[0]
                seq_len = kwargs.get('input_ids', torch.ones(1, 16)).shape[1]
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(batch_size, seq_len, 32)
                return mock_output
                
            llm.forward = mock_forward
            llm.__call__ = mock_forward
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
        
        # Maximale Sequenzlänge
        max_length = 16
        
        # Dataset mit echtem oder gemocktem Tokenizer
        dataset = MultiTaskDataset(texts, ethics_labels, manipulation_labels, tokenizer, max_length=max_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        model_config = {
            'input_dim': 32,  # Muss mit der hidden_size des LLM übereinstimmen
            'd_model': 32,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': getattr(tokenizer, 'vocab_size', 1000),
            'max_seq_length': max_length,
            'activation': 'gelu',
            'use_gnn': False,
            'spacy_model': 'de_core_news_sm'  # Verwende deutsches Modell
        }
        model = EthicsModel(**model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCELoss()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training (nur 1 Epoche, patience=0)
        try:
            trained_model = train(
                model, llm, dataloader, optimizer, criterion, 
                writer=None, device=device, epochs=1, patience=0, 
                symbolic_constraints=None, use_amp=False  # Disable AMP for testing
            )
            # Assert that training completed
            self.assertIsNotNone(trained_model)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")


if __name__ == '__main__':
    unittest.main()
