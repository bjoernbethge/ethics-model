"""
Data Processing for Ethics Model

Enhanced dataset classes for multi-task ethical analysis with support
for NetworkX graphs and spaCy NLP processing.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Union, Callable, Tuple
import networkx as nx
import spacy
from spacy.tokens import Doc
import random
from tqdm import tqdm
import json
import pickle
import hashlib
from pathlib import Path


class MultiTaskDataset(Dataset):
    """
    Basic multi-task dataset for ethics and manipulation detection.
    
    This dataset handles the core functionality for ethics and manipulation
    detection tasks with support for data augmentation.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 ethics_labels: List[float], 
                 manipulation_labels: List[float], 
                 tokenizer: Any, 
                 max_length: int = 128, 
                 augment: bool = False, 
                 synonym_augment: Optional[Callable] = None,
                 include_raw_text: bool = True):
        """
        Initialize basic multi-task dataset.
        
        Args:
            texts: List of text strings
            ethics_labels: Ethics scores (0-1)
            manipulation_labels: Manipulation scores (0-1)
            tokenizer: Tokenizer (typically from Hugging Face)
            max_length: Maximum sequence length
            augment: Whether to use augmentation
            synonym_augment: Function for synonym replacement augmentation
            include_raw_text: Whether to include raw text in output
        """
        if len(texts) != len(ethics_labels) or len(texts) != len(manipulation_labels):
            raise ValueError("All input lists must have the same length")
        
        self.texts = texts
        self.ethics_labels = ethics_labels
        self.manipulation_labels = manipulation_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.synonym_augment = synonym_augment
        self.include_raw_text = include_raw_text
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def _apply_augmentation(self, text: str) -> str:
        """Apply text augmentation if enabled."""
        if not self.augment or self.synonym_augment is None:
            return text
            
        if random.random() < 0.3:  # 30% chance to augment
            try:
                return self.synonym_augment(text)
            except Exception as e:
                print(f"Warning: Augmentation failed: {e}")
                return text
        
        return text
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        if idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.texts)}")
        
        text = self.texts[idx]
        
        # Apply augmentation if enabled
        augmented_text = self._apply_augmentation(text)
                
        # Tokenize text
        try:
            inputs = self.tokenizer(
                augmented_text, 
                return_tensors='pt', 
                max_length=self.max_length, 
                truncation=True, 
                padding='max_length'
            )
        except Exception as e:
            print(f"Warning: Tokenization failed for text {idx}: {e}")
            # Fallback to empty sequence
            inputs = self.tokenizer(
                "", 
                return_tensors='pt', 
                max_length=self.max_length, 
                truncation=True, 
                padding='max_length'
            )
        
        # Convert to PyTorch tensors and squeeze batch dimension
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add labels
        item['ethics_label'] = torch.tensor([self.ethics_labels[idx]], dtype=torch.float32)
        item['manipulation_label'] = torch.tensor([self.manipulation_labels[idx]], dtype=torch.float32)
        
        # Add metadata
        item['sample_id'] = torch.tensor([idx], dtype=torch.long)
        
        # Include raw text if requested
        if self.include_raw_text:
            item['text'] = text  # Original text (not augmented)
            item['augmented_text'] = augmented_text  # Potentially augmented text
        
        return item


class GraphEthicsDataset(MultiTaskDataset):
    """
    Enhanced dataset with NetworkX graph support and spaCy NLP processing.
    
    This dataset extracts ethical relationships from text using spaCy and 
    represents them as NetworkX graphs for graph neural network processing.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 ethics_labels: List[float], 
                 manipulation_labels: List[float], 
                 tokenizer: Any, 
                 max_length: int = 128, 
                 augment: bool = False, 
                 synonym_augment: Optional[Callable] = None,
                 spacy_model: str = "en_core_web_sm",
                 preprocess_graphs: bool = False,
                 cache_graphs: bool = True,
                 cache_dir: Optional[str] = None,
                 include_raw_text: bool = True,
                 max_graph_nodes: int = 50):
        """
        Initialize enhanced dataset with graph capabilities.
        
        Args:
            texts: List of text strings
            ethics_labels: Ethics scores (0-1)
            manipulation_labels: Manipulation scores (0-1)
            tokenizer: Tokenizer (typically from Hugging Face)
            max_length: Maximum sequence length
            augment: Whether to use augmentation
            synonym_augment: Function for synonym replacement augmentation
            spacy_model: spaCy model name for NLP processing
            preprocess_graphs: Whether to preprocess graphs (slower but faster training)
            cache_graphs: Whether to cache processed graphs
            cache_dir: Directory for caching graphs
            include_raw_text: Whether to include raw text in output
            max_graph_nodes: Maximum number of nodes in graphs (for memory management)
        """
        super().__init__(
            texts, ethics_labels, manipulation_labels, tokenizer, 
            max_length, augment, synonym_augment, include_raw_text
        )
        
        self.spacy_model = spacy_model
        self.cache_graphs = cache_graphs
        self.max_graph_nodes = max_graph_nodes
        
        # Initialize spaCy model
        self.nlp = self._load_spacy_model()
        
        # Set up caching
        self.cache_dir = None
        self.graph_cache = {}
        if cache_graphs:
            self._setup_cache(cache_dir)
            
        # Preprocess graphs if requested
        if preprocess_graphs:
            print("Preprocessing graphs (this may take a while)...")
            self._preprocess_all_graphs()
    
    def _load_spacy_model(self) -> spacy.Language:
        """Load spaCy model with fallback."""
        try:
            return spacy.load(self.spacy_model)
        except OSError:
            print(f"Warning: Could not load spaCy model '{self.spacy_model}'. "
                  f"Please install with: python -m spacy download {self.spacy_model}")
            # Create blank model as fallback
            return spacy.blank("en")
    
    def _setup_cache(self, cache_dir: Optional[str] = None):
        """Set up graph caching."""
        if cache_dir is None:
            cache_dir = ".ethics_model_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load existing cache if available
        cache_file = self.cache_dir / f"graph_cache_{self.spacy_model}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.graph_cache = pickle.load(f)
                print(f"Loaded {len(self.graph_cache)} cached graphs")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.graph_cache = {}
    
    def _save_cache(self):
        """Save graph cache to disk."""
        if self.cache_dir is None or not self.graph_cache:
            return
        
        cache_file = self.cache_dir / f"graph_cache_{self.spacy_model}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph_cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _extract_graph_from_text(self, text: str) -> nx.DiGraph:
        """
        Extract ethical relationship graph from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            NetworkX directed graph
        """
        # Check cache first
        text_hash = self._get_text_hash(text)
        if self.cache_graphs and text_hash in self.graph_cache:
            return self.graph_cache[text_hash]
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Extract entities and their moral significance
        moral_entities = self._extract_moral_entities(doc)
        
        # Add nodes
        node_id = 0
        node_mapping = {}
        
        for entity_text, entity_data in moral_entities.items():
            if len(graph.nodes) >= self.max_graph_nodes:
                break
                
            node_mapping[entity_text] = node_id
            graph.add_node(node_id, 
                          text=entity_text,
                          **entity_data)
            node_id += 1
        
        # Add edges from dependency relationships
        self._add_dependency_edges(doc, graph, node_mapping)
        
        # Add co-occurrence edges
        self._add_cooccurrence_edges(doc, graph, node_mapping)
        
        # Cache result
        if self.cache_graphs:
            self.graph_cache[text_hash] = graph
        
        return graph
    
    def _extract_moral_entities(self, doc: Doc) -> Dict[str, Dict[str, Any]]:
        """Extract morally relevant entities from spaCy doc."""
        moral_categories = {
            "positive_actions": ["help", "assist", "support", "protect", "save", "benefit", 
                               "contribute", "donate", "volunteer", "share", "cooperate"],
            "negative_actions": ["harm", "hurt", "damage", "destroy", "kill", "injure", 
                               "steal", "cheat", "lie", "deceive", "manipulate", "exploit"],
            "moral_values": ["fairness", "justice", "care", "loyalty", "authority", "purity", 
                           "freedom", "rights", "equality", "respect"],
            "emotions": ["happy", "sad", "angry", "fear", "love", "hate", "hope", "despair"]
        }
        
        entities = {}
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"]:
                entities[ent.text] = {
                    "type": "entity",
                    "subtype": ent.label_,
                    "moral_weight": 0.5,
                    "sentiment": 0.0
                }
        
        # Extract morally significant words
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                text_lower = token.text.lower()
                
                for category, words in moral_categories.items():
                    if any(word in text_lower for word in words):
                        moral_weight = 0.8 if "action" in category else 0.6
                        sentiment = 1.0 if "positive" in category else (-1.0 if "negative" in category else 0.0)
                        
                        entities[token.text] = {
                            "type": category,
                            "subtype": category,
                            "moral_weight": moral_weight,
                            "sentiment": sentiment
                        }
                        break
        
        return entities
    
    def _add_dependency_edges(self, doc: Doc, graph: nx.DiGraph, node_mapping: Dict[str, int]):
        """Add edges based on dependency relationships."""
        for token in doc:
            if token.text in node_mapping and token.head.text in node_mapping:
                source_id = node_mapping[token.head.text]
                target_id = node_mapping[token.text]
                
                # Calculate edge weight based on dependency type
                weight = self._get_dependency_weight(token.dep_)
                
                graph.add_edge(source_id, target_id, 
                              relation=token.dep_,
                              weight=weight,
                              edge_type="dependency")
    
    def _add_cooccurrence_edges(self, doc: Doc, graph: nx.DiGraph, node_mapping: Dict[str, int]):
        """Add edges for entities that co-occur in the same sentence."""
        for sent in doc.sents:
            sent_entities = []
            for token in sent:
                if token.text in node_mapping:
                    sent_entities.append(node_mapping[token.text])
            
            # Connect entities that appear together
            for i, entity1 in enumerate(sent_entities):
                for entity2 in sent_entities[i+1:]:
                    if not graph.has_edge(entity1, entity2):
                        graph.add_edge(entity1, entity2, 
                                      relation="co_occurrence",
                                      weight=0.3,
                                      edge_type="co_occurrence")
    
    def _get_dependency_weight(self, dep: str) -> float:
        """Calculate weight for dependency relationship."""
        weight_map = {
            "nsubj": 0.9, "nsubjpass": 0.9,  # Subject relationships
            "dobj": 0.8, "iobj": 0.7,        # Object relationships
            "amod": 0.6, "advmod": 0.5,       # Modifiers
            "compound": 0.4, "prep": 0.3      # Other relationships
        }
        return weight_map.get(dep, 0.2)
    
    def _preprocess_all_graphs(self):
        """Preprocess graphs for all texts."""
        for i, text in enumerate(tqdm(self.texts, desc="Processing graphs")):
            self._extract_graph_from_text(text)
        
        # Save cache after preprocessing
        if self.cache_graphs:
            self._save_cache()
    
    def _graph_to_tensors(self, graph: nx.DiGraph) -> Dict[str, torch.Tensor]:
        """Convert NetworkX graph to PyTorch tensors."""
        if graph.number_of_nodes() == 0:
            # Empty graph fallback
            return {
                "node_features": torch.zeros((1, 6), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, 3), dtype=torch.float32),
                "num_nodes": 1,
                "has_graph": False
            }
        
        # Extract node features
        nodes = list(graph.nodes())
        node_features = []
        
        for node in nodes:
            node_data = graph.nodes[node]
            
            # Create feature vector: [type_encoding(4), moral_weight(1), sentiment(1)]
            feature = [0, 0, 0, 0]  # One-hot for type
            
            node_type = node_data.get("type", "other")
            if "action" in node_type:
                feature[0] = 1
            elif node_type == "entity":
                feature[1] = 1
            elif "value" in node_type:
                feature[2] = 1
            else:
                feature[3] = 1
            
            feature.append(node_data.get("moral_weight", 0.0))
            feature.append(node_data.get("sentiment", 0.0))
            
            node_features.append(feature)
        
        # Extract edges
        edge_indices = []
        edge_attributes = []
        
        for source, target, edge_data in graph.edges(data=True):
            source_idx = nodes.index(source)
            target_idx = nodes.index(target)
            
            # Add both directions for undirected representation
            edge_indices.extend([[source_idx, target_idx], [target_idx, source_idx]])
            
            # Edge attributes: [weight, relation_type_encoded, is_dependency]
            weight = edge_data.get("weight", 0.5)
            relation = edge_data.get("relation", "unknown")
            
            # Encode relation type
            relation_encoding = {
                "nsubj": 1.0, "dobj": 0.9, "amod": 0.7, "co_occurrence": 0.3
            }.get(relation, 0.5)
            
            is_dependency = 1.0 if edge_data.get("edge_type") == "dependency" else 0.0
            
            edge_attr = [weight, relation_encoding, is_dependency]
            edge_attributes.extend([edge_attr, edge_attr])
        
        # Convert to tensors
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        if edge_indices:
            edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)
        else:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.zeros((0, 3), dtype=torch.float32)
        
        return {
            "node_features": node_features_tensor,
            "edge_index": edge_index_tensor,
            "edge_attr": edge_attr_tensor,
            "num_nodes": len(nodes),
            "has_graph": True
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item with graph data."""
        # Get base item from parent class
        item = super().__getitem__(idx)
        
        if not self.include_raw_text:
            return item
        
        # Extract graph from text
        text = item.get('text', self.texts[idx])
        
        try:
            graph = self._extract_graph_from_text(text)
            graph_tensors = self._graph_to_tensors(graph)
            
            # Add graph data to item
            for key, value in graph_tensors.items():
                item[f'graph_{key}'] = value
                
        except Exception as e:
            print(f"Warning: Graph extraction failed for sample {idx}: {e}")
            # Add empty graph data
            empty_graph = {
                "node_features": torch.zeros((1, 6), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, 3), dtype=torch.float32),
                "num_nodes": 1,
                "has_graph": False
            }
            for key, value in empty_graph.items():
                item[f'graph_{key}'] = value
        
        return item
    
    def __del__(self):
        """Cleanup: save cache when dataset is destroyed."""
        if hasattr(self, 'cache_graphs') and self.cache_graphs:
            self._save_cache()


def collate_ethics_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching ethics dataset items.
    
    Handles both standard text data and graph data, creating proper
    batches for PyTorch training.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched data dictionary
    """
    if not batch:
        return {}
    
    # Separate different types of data
    standard_items = {}
    graph_items = {}
    texts = []
    augmented_texts = []
    
    for item in batch:
        # Extract texts if available
        if 'text' in item:
            texts.append(item['text'])
        if 'augmented_text' in item:
            augmented_texts.append(item['augmented_text'])
        
        # Separate graph data from standard items
        for key, value in item.items():
            if key.startswith('graph_'):
                if key not in graph_items:
                    graph_items[key] = []
                graph_items[key].append(value)
            elif key not in ['text', 'augmented_text']:  # Skip text fields
                if key not in standard_items:
                    standard_items[key] = []
                standard_items[key].append(value)
    
    # Batch standard items
    batched = {}
    for key, values in standard_items.items():
        try:
            if isinstance(values[0], torch.Tensor):
                if key == 'input_ids':
                    batched[key] = torch.stack([v.long() for v in values])
                else:
                    batched[key] = torch.stack(values)
            else:
                batched[key] = values
        except Exception as e:
            print(f"Warning: Could not batch key '{key}': {e}")
            batched[key] = values
    
    # Add text data
    if texts:
        batched['texts'] = texts
    if augmented_texts:
        batched['augmented_texts'] = augmented_texts
    
    # Add graph data (keep as lists for graph processing)
    for key, values in graph_items.items():
        batched[key] = values
    
    # Add batch metadata
    batched['batch_size'] = len(batch)
    
    return batched


def create_data_splits(texts: List[str], 
                      ethics_labels: List[float], 
                      manipulation_labels: List[float],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      seed: int = 42) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        texts: List of text strings
        ethics_labels: Ethics scores
        manipulation_labels: Manipulation scores
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Create indices and shuffle
    n_samples = len(texts)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    # Calculate split points
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create data splits
    def create_split(split_indices):
        return {
            'texts': [texts[i] for i in split_indices],
            'ethics_labels': [ethics_labels[i] for i in split_indices],
            'manipulation_labels': [manipulation_labels[i] for i in split_indices]
        }
    
    train_data = create_split(train_indices)
    val_data = create_split(val_indices)
    test_data = create_split(test_indices)
    
    return train_data, val_data, test_data


def load_from_json(file_path: str) -> Tuple[List[str], List[float], List[float]]:
    """
    Load dataset from JSON file.
    
    Expected format:
    [
        {
            "text": "Sample text",
            "ethics_score": 0.8,
            "manipulation_score": 0.2
        },
        ...
    ]
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Tuple of (texts, ethics_labels, manipulation_labels)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    ethics_labels = []
    manipulation_labels = []
    
    for item in data:
        texts.append(item['text'])
        ethics_labels.append(float(item['ethics_score']))
        manipulation_labels.append(float(item['manipulation_score']))
    
    return texts, ethics_labels, manipulation_labels


def save_to_json(texts: List[str], 
                ethics_labels: List[float], 
                manipulation_labels: List[float],
                file_path: str):
    """
    Save dataset to JSON file.
    
    Args:
        texts: List of text strings
        ethics_labels: Ethics scores
        manipulation_labels: Manipulation scores
        file_path: Path to save JSON file
    """
    data = []
    for text, ethics, manipulation in zip(texts, ethics_labels, manipulation_labels):
        data.append({
            "text": text,
            "ethics_score": ethics,
            "manipulation_score": manipulation
        })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Example usage and utilities
if __name__ == "__main__":
    # Example dataset creation
    sample_texts = [
        "John helped Mary when she was struggling with her work.",
        "The politician manipulated the statistics to mislead voters.",
        "She showed compassion and fairness in her decision making.",
        "The company exploited workers for maximum profit."
    ]
    
    sample_ethics = [0.9, 0.2, 0.95, 0.1]
    sample_manipulation = [0.1, 0.9, 0.05, 0.85]
    
    print("Creating sample dataset...")
    
    # Mock tokenizer for demonstration
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            words = text.split()[:10]  # Limit to 10 words
            return {
                'input_ids': torch.tensor([list(range(len(words)))]),
                'attention_mask': torch.tensor([([1] * len(words))])
            }
    
    tokenizer = MockTokenizer()
    
    # Create enhanced dataset
    dataset = GraphEthicsDataset(
        texts=sample_texts,
        ethics_labels=sample_ethics,
        manipulation_labels=sample_manipulation,
        tokenizer=tokenizer,
        max_length=128,
        preprocess_graphs=True
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test data loading
    sample_item = dataset[0]
    print(f"Sample item keys: {list(sample_item.keys())}")
    
    # Test collate function
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    batched = collate_ethics_batch(batch)
    print(f"Batched data keys: {list(batched.keys())}")
