"""
Data Processing for Ethics Model

Enhanced dataset classes for multi-task ethical analysis with support
for GraphBrain and Instructor integrations.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Union, Callable
import graphbrain as gb
from graphbrain import hgraph
import random
from tqdm import tqdm


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
        self.graph_cache = {}
        
        if use_graphbrain:
            self._init_parser()
            
            # Preprocess graphs if requested
            if preprocess_graphs:
                print("Preprocessing graphs (this may take a while)...")
                for i, text in enumerate(tqdm(texts)):
                    self._process_text_to_graph(text)
    
    def _init_parser(self):
        """Initialize GraphBrain parser."""
        if self.parser is None and self.use_graphbrain:
            try:
                self.parser = gb.Parser(model=f"{self.parser_lang}_core_web_sm")
            except Exception as e:
                print(f"Error initializing GraphBrain parser: {e}")
                print("Make sure you've installed the required language model:")
                print(f"python -m spacy download {self.parser_lang}_core_web_sm")
                self.use_graphbrain = False
    
    def _process_text_to_graph(self, text: str) -> Optional[hgraph]:
        """Process text to GraphBrain hypergraph."""
        if not self.use_graphbrain:
            return None
            
        # Use cache if available
        if self.cache_graphs and text in self.graph_cache:
            return self.graph_cache[text]
            
        self._init_parser()
        if self.parser is None:
            return None
            
        try:
            # Create hypergraph
            hg = hgraph()
            
            # Parse text and add to hypergraph
            for sentence in text.split('.'):
                if sentence.strip():
                    parse = self.parser.parse(sentence)
                    hg.add(parse)
                    
            # Cache result
            if self.cache_graphs:
                self.graph_cache[text] = hg
                
            return hg
        except Exception as e:
            print(f"Error processing text to graph: {e}")
            return None
    
    def _prepare_graph_data(self, hg: hgraph) -> Dict[str, Any]:
        """
        Prepare graph data for model input.
        
        Args:
            hg: GraphBrain hypergraph
            
        Returns:
            Dictionary with graph data
        """
        if hg is None:
            return {'has_graph': False}
            
        # Extract nodes (atoms)
        nodes = list(hg.all_atoms())
        
        # Create node map for edge index creation
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Create edge index
        edge_index = []
        edge_types = []
        
        # Extract edges
        for edge in hg.all_edges():
            if len(edge) > 1:
                pred = edge[0]  # Predicate is typically first
                
                if gb.is_atom(pred):
                    pred_idx = node_map.get(pred)
                    
                    if pred_idx is not None:
                        # Connect predicate to all arguments
                        for i in range(1, len(edge)):
                            if gb.is_atom(edge[i]):
                                arg_idx = node_map.get(edge[i])
                                
                                if arg_idx is not None:
                                    # Add bidirectional connections
                                    edge_index.append([pred_idx, arg_idx])
                                    edge_index.append([arg_idx, pred_idx])
                                    
                                    # Add edge types (simplified)
                                    edge_type = 0  # Default
                                    edge_types.append(edge_type)
                                    edge_types.append(edge_type)
        
        # Convert to tensors
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)
        else:
            # Empty graph fallback
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_types_tensor = torch.zeros(0, dtype=torch.long)
        
        # Simple node features (one-hot encoding)
        node_features = torch.eye(len(nodes))
        
        return {
            'has_graph': True,
            'nodes': nodes,
            'node_features': node_features,
            'edge_index': edge_index_tensor,
            'edge_types': edge_types_tensor
        }
    
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


def collate_with_graphs(batch):
    """
    Custom collate function for batches with graph data.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched tensors and graph data
    """
    # Separate standard items and graph data
    standard_items = {}
    graph_data = {}
    texts = []
    
    for item in batch:
        # Extract text
        texts.append(item.pop('text'))
        
        # Separate graph data from standard items
        graph_keys = [k for k in item.keys() if k.startswith('graph_')]
        
        for k in item.keys():
            if k in graph_keys:
                if k not in graph_data:
                    graph_data[k] = []
                graph_data[k].append(item[k])
            else:
                if k not in standard_items:
                    standard_items[k] = []
                standard_items[k].append(item[k])
    
    # Batch standard items
    batched = {k: torch.stack(v) for k, v in standard_items.items()}
    batched['texts'] = texts
    
    # Add graph data
    for k, v in graph_data.items():
        batched[k] = v
    
    return batched
