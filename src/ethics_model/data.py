"""
Data Processing for Ethics Model

Enhanced dataset classes for multi-task ethical analysis with support
for GraphBrain and Instructor integrations.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Union, Callable
from graphbrain import hgraph
import random
from tqdm import tqdm

from .common import (
    collate_with_graphs,
    GraphBrainParserManager,
    process_text_to_hypergraph,
    prepare_graph_data_for_model
)


class MultiTaskDataset(Dataset):
    """
    Basic multi-task dataset for ethics and manipulation detection.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 ethics_labels: List[float], 
                 manipulation_labels: List[float], 
                 tokenizer: Any, 
                 max_length: int = 128, 
                 augment: bool = False, 
                 synonym_augment: Optional[Callable] = None):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            ethics_labels: Ethics scores (0-1)
            manipulation_labels: Manipulation scores (0-1)
            tokenizer: Tokenizer (typically from Hugging Face)
            max_length: Maximum sequence length
            augment: Whether to use augmentation
            synonym_augment: Function for synonym replacement augmentation
        """
        self.texts = texts
        self.ethics_labels = ethics_labels
        self.manipulation_labels = manipulation_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.synonym_augment = synonym_augment
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Apply augmentation if enabled
        if self.augment and self.synonym_augment is not None:
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
        
        # Convert to PyTorch tensors
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['ethics_label'] = torch.tensor([self.ethics_labels[idx]], dtype=torch.float32)
        item['manipulation_label'] = torch.tensor([self.manipulation_labels[idx]], dtype=torch.float32)
        item['text'] = text  # Store raw text for additional processing
        
        return item


class EnhancedEthicsDataset(MultiTaskDataset):
    """
    Enhanced dataset with GraphBrain support.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 ethics_labels: List[float], 
                 manipulation_labels: List[float], 
                 tokenizer: Any, 
                 max_length: int = 128, 
                 augment: bool = False, 
                 synonym_augment: Optional[Callable] = None,
                 use_graphbrain: bool = True,  # Default to using GraphBrain
                 preprocess_graphs: bool = False,
                 parser_lang: str = "en",
                 cache_graphs: bool = True):
        """
        Initialize enhanced dataset.
        
        Args:
            texts: List of text strings
            ethics_labels: Ethics scores (0-1)
            manipulation_labels: Manipulation scores (0-1)
            tokenizer: Tokenizer (typically from Hugging Face)
            max_length: Maximum sequence length
            augment: Whether to use augmentation
            synonym_augment: Function for synonym replacement augmentation
            use_graphbrain: Whether to use GraphBrain
            use_instructor: Whether to use Instructor
            instructor_client: Instructor client for structured extraction
            preprocess_graphs: Whether to preprocess graphs (can be slow but saves time during training)
            parser_lang: Language for GraphBrain parser
            cache_graphs: Whether to cache parsed graphs
        """
        super().__init__(
            texts, ethics_labels, manipulation_labels, tokenizer, 
            max_length, augment, synonym_augment
        )
        
        self.use_graphbrain = use_graphbrain
        self.parser_lang = parser_lang
        self.cache_graphs = cache_graphs
        
        # Initialize GraphBrain parser if needed
        self.parser = None
        self.graph_cache: Dict[str, Any] = {}
        
        if use_graphbrain:
            self._init_parser()
            
            # Preprocess graphs if requested
            if preprocess_graphs:
                print("Preprocessing graphs (this may take a while)...")
                for i, text in enumerate(tqdm(texts)):
                    self._process_text_to_graph(text)
    
    def _init_parser(self):
        """Initialize GraphBrain parser using shared manager."""
        if self.parser is None and self.use_graphbrain:
            self.parser = GraphBrainParserManager.get_parser(self.parser_lang)
            if self.parser is None:
                self.use_graphbrain = False
    
    def _process_text_to_graph(self, text: str) -> Optional[hgraph]:
        """Process text to GraphBrain hypergraph."""
        if not self.use_graphbrain:
            return None
            
        # Use cache if available
        if self.cache_graphs and text in self.graph_cache:
            return self.graph_cache[text]
            
        self._init_parser()
        hg = process_text_to_hypergraph(text, self.parser, self.parser_lang)
        
        # Cache result
        if self.cache_graphs and hg is not None:
            self.graph_cache[text] = hg
                
        return hg
    
    def _prepare_graph_data(self, hg: Optional[hgraph]) -> Dict[str, Any]:
        """
        Prepare graph data for model input.
        
        Args:
            hg: GraphBrain hypergraph
            
        Returns:
            Dictionary with graph data
        """
        return prepare_graph_data_for_model(hg)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get base item from parent class
        item = super().__getitem__(idx)
        text = self.texts[idx]
        
        # Add GraphBrain data if enabled
        if self.use_graphbrain:
            graph = self._process_text_to_graph(text)
            graph_data = self._prepare_graph_data(graph)
            
            # Add graph data to item
            for k, v in graph_data.items():
                item[f'graph_{k}'] = v
        
        # We don't add Instructor data here since it requires
        # online API calls which we would do during training
        # rather than dataset loading for efficiency
        
        return item
