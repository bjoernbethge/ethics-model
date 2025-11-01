import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ethics_model.data import EthicsDataset
from ethics_model.modules.retriever import EthicsModel
from ethics_model.training import train


def test_train_llm_short():
    # Mini-Datensatz
    texts = ["Hallo Welt", "Test", "Noch ein Text", "Letzter"]
    ethics_labels = [1.0, 0.0, 1.0, 0.0]
    manipulation_labels = [0.0, 1.0, 0.0, 1.0]
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        quantization_config=bnb_config,
        device_map="cpu"
    )
    dataset = EthicsDataset(texts, ethics_labels, manipulation_labels, tokenizer, max_length=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    model_config = {
        'input_dim': llm.config.hidden_size,
        'd_model': llm.config.hidden_size,
        'n_layers': 1,
        'n_heads': 2,
        'vocab_size': tokenizer.vocab_size,
        'max_seq_length': 16,
        'activation': 'gelu',
        'use_gnn': False
    }
    model = EthicsModel(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    # Training (nur 1 Epoche, patience=0)
    train(model, llm, dataloader, optimizer, criterion, writer=None, device="cpu", epochs=1, patience=0, symbolic_constraints=None) 