"""
ETHICS Dataset Loader

This module provides utilities for loading and processing the ETHICS dataset
from Hendrycks et al. (2021) for training ethical analysis models.
Uses Polars for efficient data processing.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

try:
    import polars as pl
except ImportError:
    pl = None  # Optional dependency

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Ethical domains in the ETHICS dataset
DOMAINS = [
    "justice",
    "virtue",
    "deontology",
    "utilitarianism",
    "commonsense"
]


class EthicsDataset(Dataset):
    """
    Simple PyTorch Dataset for ethics text analysis.
    
    Args:
        texts: List of text strings
        ethics_labels: List of ethics scores (0.0-1.0)
        manipulation_labels: List of manipulation scores (0.0-1.0)
        tokenizer: HuggingFace tokenizer for text processing
        max_length: Maximum sequence length for tokenization
        augment: Whether to use data augmentation
        synonym_augment: Optional function for synonym-based augmentation
    """
    
    def __init__(
        self,
        texts: List[str],
        ethics_labels: List[float],
        manipulation_labels: List[float],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        augment: bool = False,
        synonym_augment: Optional[Callable[[str], str]] = None
    ):
        self.texts = texts
        self.ethics_labels = ethics_labels
        self.manipulation_labels = manipulation_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.synonym_augment = synonym_augment
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing tokenized text and labels
        """
        text = self.texts[idx]
        
        # Apply augmentation if enabled
        if self.augment and self.synonym_augment is not None:
            import random
            if random.random() < 0.3:
                text = self.synonym_augment(text)
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Create item dictionary
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['ethics_label'] = torch.tensor([self.ethics_labels[idx]], dtype=torch.float32)
        item['manipulation_label'] = torch.tensor([self.manipulation_labels[idx]], dtype=torch.float32)
        
        return item

class ETHICSDataset(Dataset):
    """
    PyTorch Dataset implementation for the ETHICS dataset by Hendrycks et al. (2021).
    Uses Polars for efficient data processing.
    
    Args:
        domain: Ethical domain (justice, virtue, deontology, utilitarianism, commonsense)
        split: Data split to use (train, test, test_hard, ambiguous)
        data_dir: Directory containing the ETHICS dataset
        tokenizer: HuggingFace tokenizer for text processing
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        domain: str,
        split: str,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        self.domain = domain
        self.split = split
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Handle 'ambiguous' split only for commonsense
        if split == "ambiguous" and domain != "commonsense":
            raise ValueError("'ambiguous' split only available for 'commonsense' domain")
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self):
        """
        Load and preprocess data from the ETHICS dataset.
        
        Returns:
            Polars DataFrame containing the dataset
        """
        if pl is None:
            raise ImportError(
                "polars is required for ETHICSDataset. "
                "Install it with: uv sync --extra train"
            )
        # Define file path
        if self.split == "ambiguous":
            file_path = os.path.join(self.data_dir, self.domain, f"{self.split}.csv")
        else:
            file_path = os.path.join(self.data_dir, self.domain, f"{self.split}.csv")
        
        # Load data with Polars
        df = pl.read_csv(file_path)
        
        # Process based on domain
        if self.domain == "commonsense":
            # For commonsense, we have 'label' and 'scenario'
            df = df.rename({"label": "ethics_label", "scenario": "text"})
            
            # Binary labels: 0 for unethical, 1 for ethical
            if "ethics_label" in df.columns:
                df = df.with_columns(pl.col("ethics_label").cast(pl.Float32))
            
            # Add manipulation label (not directly available in commonsense data)
            # Assume higher manipulation risk for unethical content (simplified)
            if "ethics_label" in df.columns:
                df = df.with_columns(
                    (1 - pl.col("ethics_label") * 0.8).alias("manipulation_label")
                )
        
        elif self.domain in ["justice", "virtue", "deontology", "utilitarianism"]:
            # Other domains have different structures 
            # Adapt as needed for each domain's specific columns
            if "label" in df.columns:
                df = df.rename({"label": "ethics_label"})
            
            # Find the text column - might be 'scenario', 'sentence', etc.
            text_column = None
            for col in ["scenario", "sentence", "text"]:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column:
                df = df.rename({text_column: "text"})
            else:
                # If no standard text column found, use the first column that's not 'ethics_label'
                for col in df.columns:
                    if col != "ethics_label":
                        df = df.rename({col: "text"})
                        break
            
            # Ensure ethics_label is float32
            if "ethics_label" in df.columns:
                df = df.with_columns(pl.col("ethics_label").cast(pl.Float32))
                
            # Add manipulation label (not directly available in these domains)
            # Simple heuristic: higher manipulation risk for unethical content
            if "ethics_label" in df.columns:
                df = df.with_columns(
                    (1 - pl.col("ethics_label") * 0.8).alias("manipulation_label")
                )
        
        # Ensure we have the required columns
        for col in ["text", "ethics_label", "manipulation_label"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        return df
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing tokenized text and labels
        """
        # Get row from DataFrame
        row = self.data.row(idx)
        
        # Extract text and labels
        text = row["text"]
        ethics_label = row["ethics_label"]
        manipulation_label = row["manipulation_label"]
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Create item dictionary
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['ethics_label'] = torch.tensor([ethics_label], dtype=torch.float32)
        item['manipulation_label'] = torch.tensor([manipulation_label], dtype=torch.float32)
        item['text'] = text
        
        return item


class ETHICSMultiDomainDataset(Dataset):
    """
    Combined dataset for multiple domains from the ETHICS dataset.
    
    Args:
        data_dir: Directory containing the ETHICS dataset
        domains: List of domains to include (default: all domains)
        split: Data split to use (train, test, test_hard)
        tokenizer: HuggingFace tokenizer for text processing
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        data_dir: str,
        domains: Optional[List[str]] = None,
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 128
    ):
        self.data_dir = data_dir
        self.domains = domains or DOMAINS
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Initialize domain datasets
        self.domain_datasets = {}
        self.domain_indices = {}
        
        # Total dataset size
        self.total_size = 0
        
        # Load data for each domain
        for domain in self.domains:
            try:
                dataset = ETHICSDataset(
                    domain=domain,
                    split=split,
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    max_length=max_length
                )
                
                # Store dataset and indices
                self.domain_datasets[domain] = dataset
                self.domain_indices[domain] = (self.total_size, self.total_size + len(dataset))
                self.total_size += len(dataset)
                
            except Exception as e:
                logger.error(f"Error loading {domain} dataset: {e}")
                continue
    
    def __len__(self) -> int:
        """Get the total number of examples across all domains."""
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example from the combined dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing example data and domain information
        """
        # Find the domain for this index
        domain = None
        domain_idx = None
        
        for d, (start, end) in self.domain_indices.items():
            if start <= idx < end:
                domain = d
                domain_idx = idx - start
                break
        
        if domain is None or domain_idx is None:
            raise IndexError(f"Index {idx} out of range")
        
        # Get item from the domain dataset
        item = self.domain_datasets[domain][domain_idx]
        
        # Add domain information
        item['domain'] = domain
        
        return item


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batches.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched tensors
    """
    texts = []
    domains = []
    standard_items = {}
    
    for item in batch:
        # Extract text and domain
        texts.append(item.pop('text'))
        if 'domain' in item:
            domains.append(item.pop('domain'))
        
        # Collect standard items
        for k, v in item.items():
            if k not in standard_items:
                standard_items[k] = []
            standard_items[k].append(v)
    
    # Batch standard items
    batched = {k: torch.stack(v) for k, v in standard_items.items()}
    batched['texts'] = texts
    
    if domains:
        batched['domains'] = domains
    
    return batched


def create_ethics_dataloaders(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    domains: Optional[List[str]] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoader objects for the ETHICS dataset.
    
    Args:
        data_dir: Directory containing the ETHICS dataset
        tokenizer: HuggingFace tokenizer for text processing
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length for tokenization
        domains: List of domains to include (default: all domains)
        num_workers: Number of workers for DataLoader
        
    Returns:
        Dictionary of DataLoader objects for train, test, and test_hard splits
    """
    # Create combined datasets for each split
    splits = ["train", "test", "test_hard"]
    datasets = {}
    
    for split in splits:
        datasets[split] = ETHICSMultiDomainDataset(
            data_dir=data_dir,
            domains=domains,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Add ambiguous dataset for commonsense domain if available
    try:
        ambiguous_dataset = ETHICSDataset(
            domain="commonsense",
            split="ambiguous",
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        dataloaders["ambiguous"] = DataLoader(
            ambiguous_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=True
        )
    except Exception as e:
        logger.warning(f"Ambiguous dataset not available: {e}")
    
    return dataloaders
