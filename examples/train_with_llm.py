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

from ethics_model.ethics import EthicsModel

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

class MultiTaskDataset(Dataset):
    def __init__(self, texts, ethics_labels, manipulation_labels, tokenizer, max_length=128, augment=False):
        self.texts = texts
        self.ethics_labels = ethics_labels
        self.manipulation_labels = manipulation_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment and random.random() < 0.3:
            text = synonym_augment(text)
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['ethics_label'] = torch.tensor([self.ethics_labels[idx]], dtype=torch.float32)
        item['manipulation_label'] = torch.tensor([self.manipulation_labels[idx]], dtype=torch.float32)
        return item

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
def train(model, llm, dataloader, optimizer, criterion, writer, device, epochs=10, patience=2):
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ethics, total_manip, n_batches = 0.0, 0.0, 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            with torch.no_grad():
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                hidden_states = llm_outputs.last_hidden_state
            outputs = model(embeddings=hidden_states, attention_mask=attention_mask)
            ethics_score = outputs['ethics_score']
            manipulation_score = outputs['manipulation_score']
            loss_ethics = criterion(ethics_score, ethics_label)
            loss_manip = criterion(manipulation_score, manipulation_label)
            loss = loss_ethics + 0.5 * loss_manip
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            total_ethics += loss_ethics.item() * input_ids.size(0)
            total_manip += loss_manip.item() * input_ids.size(0)
            n_batches += input_ids.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        avg_ethics = total_ethics / len(dataloader.dataset)
        avg_manip = total_manip / len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Ethics: {avg_ethics:.4f} | Manip: {avg_manip:.4f}")
        writer.add_scalar('Loss/Total', avg_loss, epoch+1)
        writer.add_scalar('Loss/Ethics', avg_ethics, epoch+1)
        writer.add_scalar('Loss/Manipulation', avg_manip, epoch+1)
        # Early Stopping & Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model, CHECKPOINT_PATH)
            logger.info(f"Checkpoint saved: {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    if best_model is not None:
        model.load_state_dict(best_model)
    return model

def evaluate(model, llm, dataloader, tokenizer, writer, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
            hidden_states = llm_outputs.last_hidden_state
            outputs = model(embeddings=hidden_states, attention_mask=attention_mask)
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
    dataset = MultiTaskDataset(texts, ethics_labels, manipulation_labels, tokenizer, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = EthicsModel(**model_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss()
    model = train(model, llm, dataloader, optimizer, criterion, writer, DEVICE)
    evaluate(model, llm, dataloader, tokenizer, writer, DEVICE)
    writer.close()

if __name__ == '__main__':
    main() 