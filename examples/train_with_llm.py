import os
# CUDA-Umgebungsvariablen für bitsandbytes und CUDA-Toolkit
os.environ["BNB_CUDA_VERSION"] = "121"   # für CUDA 12.1
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "D:/dev/nvidia")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # Ada (4070/4090) Compute Capability
# =====================================================

import copy
import random
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import nlpaug.augmenter.word as naw
from tqdm import tqdm

from ethics_model.model import EthicsModel
from ethics_model.data import MultiTaskDataset
from ethics_model.training import train

# =====================
# 1. Configuration & Logging
# =====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "checkpoints/best_ethics_model.pt"
TENSORBOARD_LOGDIR = "runs/ethics_llm_train"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# =====================
# 2. LLM & Tokenizer Setup
# =====================
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
llm = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    quantization_config=bnb_config,
    device_map="auto"
)
llm.eval()

# =====================
# 3. Data Augmentation
# =====================
aug = naw.SynonymAug(aug_src='wordnet')
def synonym_augment(text):
    try:
        return aug.augment(text)
    except Exception:
        return text

# =====================
# 4. Dataset Preparation
# =====================
ds = load_dataset("flozi00/Fineweb2-German-Eduscore-4andMore", split="train[:1000]")
texts = ds["text"]
ethics_labels = [float(x) for x in ds["eduscore"]]
manipulation_labels = [float(x) for x in ds["manipulation_score"]] if "manipulation_score" in ds.column_names else ethics_labels

# =====================
# 5. Model & Training Setup
# =====================
model_config = {
    'input_dim': llm.config.hidden_size,
    'd_model': llm.config.hidden_size,
    'n_layers': 2,
    'n_heads': 8,
    'vocab_size': tokenizer.vocab_size,
    'max_seq_length': 128,
    'activation': 'gelu',
    'use_gnn': False
}

# =====================
# 6. Training & Evaluation Functions
# =====================
def evaluate(model, llm, dataloader, tokenizer, writer, device, symbolic_constraints=None):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
            hidden_states = llm_outputs.last_hidden_state
            outputs = model(embeddings=hidden_states, attention_mask=attention_mask, symbolic_constraints=symbolic_constraints)
            ethics_score = outputs['ethics_score']
            manipulation_score = outputs['manipulation_score']
            logger.info(f"Text: {tokenizer.batch_decode(input_ids, skip_special_tokens=True)}")
            logger.info(f"Ethics Score: {ethics_score.squeeze(-1).cpu().numpy()} | Label: {ethics_label.squeeze(-1).cpu().numpy()}")
            logger.info(f"Manipulation Score: {manipulation_score.squeeze(-1).cpu().numpy()} | Label: {manipulation_label.squeeze(-1).cpu().numpy()}")
            writer.add_scalar('Eval/EthicsScore', ethics_score.mean().item(), 0)
            writer.add_scalar('Eval/ManipScore', manipulation_score.mean().item(), 0)

# =====================
# 7. Main
# =====================
def main():
    writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)
    dataset = MultiTaskDataset(texts, ethics_labels, manipulation_labels, tokenizer, augment=True, synonym_augment=synonym_augment)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = EthicsModel(**model_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss()
    model = train(model, llm, dataloader, optimizer, criterion, writer, DEVICE, symbolic_constraints=None)
    evaluate(model, llm, dataloader, tokenizer, writer, DEVICE, symbolic_constraints=None)
    writer.close()

if __name__ == '__main__':
    main() 