import torch
from torch.utils.data import Dataset


class EthicsDataset(Dataset):
    def __init__(self, texts, ethics_labels, manipulation_labels, tokenizer, max_length=128, augment=False, synonym_augment=None):
        self.texts = texts
        self.ethics_labels = ethics_labels
        self.manipulation_labels = manipulation_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.synonym_augment = synonym_augment
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment and self.synonym_augment is not None:
            import random
            if random.random() < 0.3:
                text = self.synonym_augment(text)
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['ethics_label'] = torch.tensor([self.ethics_labels[idx]], dtype=torch.float32)
        item['manipulation_label'] = torch.tensor([self.manipulation_labels[idx]], dtype=torch.float32)
        return item 