"""
ETHICS Dataset Loader

This module provides utilities for loading and processing the ETHICS dataset
from Hendrycks et al. (2021) for training ethical analysis models.
Uses Polars for efficient data processing.
"""

import os
import polars as pl
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

# Ethical domains in the ETHICS dataset
DOMAINS = [
    "justice",
    "virtue",
    "deontology",
    "utilitarianism",
    "commonsense"
]

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
        use_graphbrain: Whether to use GraphBrain for semantic hypergraph processing
        parser_lang: Language for GraphBrain parser
        cache_graphs: Whether to cache processed graphs
    """
    
    def __init__(
        self,
        domain: str,
        split: str,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        use_graphbrain: bool = True,
        parser_lang: str = "en",
        cache_graphs: bool = True
    ):
        self.domain = domain
        self.split = split
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_graphbrain = use_graphbrain
        self.parser_lang = parser_lang
        self.cache_graphs = cache_graphs
        
        # Handle 'ambiguous' split only for commonsense
        if split == "ambiguous" and domain != "commonsense":
            raise ValueError("'ambiguous' split only available for 'commonsense' domain")
        
        # Load data
        self.data = self._load_data()
        
        # Initialize GraphBrain parser if needed
        self.parser = None
        if self.use_graphbrain:
            try:
                import graphbrain as gb
                self.parser = gb.Parser(model=f"{self.parser_lang}_core_web_sm")
                self.graph_cache = {}
            except ImportError:
                print("GraphBrain not available. Install with 'pip install graphbrain'")
                self.use_graphbrain = False
            except Exception as e:
                print(f"Error initializing GraphBrain parser: {e}")
                print(f"Make sure you have the {self.parser_lang}_core_web_sm model installed:")
                print(f"python -m spacy download {self.parser_lang}_core_web_sm")
                self.use_graphbrain = False
    
    def _load_data(self) -> pl.DataFrame:
        """
        Load and preprocess data from the ETHICS dataset.
        
        Returns:
            Polars DataFrame containing the dataset
        """
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
    
    def _process_text_to_graph(self, text: str) -> Optional[Any]:
        """
        Process text to create a GraphBrain hypergraph.
        
        Args:
            text: Text to process
            
        Returns:
            GraphBrain hypergraph or None if processing failed
        """
        if not self.use_graphbrain or self.parser is None:
            return None
            
        # Use cache if available
        cache_key = hash(text)
        if self.cache_graphs and cache_key in self.graph_cache:
            return self.graph_cache[cache_key]
            
        try:
            import graphbrain as gb
            from graphbrain import hgraph
            
            # Create hypergraph
            hg = hgraph()
            
            # Parse text and add to hypergraph
            for sentence in text.split('.'):
                if sentence.strip():
                    parse = self.parser.parse(sentence)
                    hg.add(parse)
                    
            # Cache result
            if self.cache_graphs:
                self.graph_cache[cache_key] = hg
                
            return hg
        except Exception as e:
            print(f"Error processing text to graph: {str(e)[:100]}")
            return None
    
    def _prepare_graph_data(self, text: str) -> Dict[str, Any]:
        """
        Prepare graph data for model input.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary with graph data
        """
        # Process text to graph
        hg = self._process_text_to_graph(text)
        
        if hg is None:
            return {'has_graph': False}
        
        try:
            import graphbrain as gb
            
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
        except Exception as e:
            print(f"Error preparing graph data: {str(e)[:100]}")
            return {'has_graph': False}
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing tokenized text, labels, and graph data
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
        
        # Add GraphBrain data if enabled
        if self.use_graphbrain:
            graph_data = self._prepare_graph_data(text)
            for k, v in graph_data.items():
                item[f'graph_{k}'] = v
        
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
        use_graphbrain: Whether to use GraphBrain for semantic hypergraph processing
        parser_lang: Language for GraphBrain parser
        cache_graphs: Whether to cache processed graphs
    """
    
    def __init__(
        self,
        data_dir: str,
        domains: Optional[List[str]] = None,
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 128,
        use_graphbrain: bool = True,
        parser_lang: str = "en",
        cache_graphs: bool = True
    ):
        self.data_dir = data_dir
        self.domains = domains or DOMAINS
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_graphbrain = use_graphbrain
        self.parser_lang = parser_lang
        self.cache_graphs = cache_graphs
        
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
                    max_length=max_length,
                    use_graphbrain=use_graphbrain,
                    parser_lang=parser_lang,
                    cache_graphs=cache_graphs
                )
                
                # Store dataset and indices
                self.domain_datasets[domain] = dataset
                self.domain_indices[domain] = (self.total_size, self.total_size + len(dataset))
                self.total_size += len(dataset)
                
            except Exception as e:
                print(f"Error loading {domain} dataset: {e}")
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


def collate_with_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    domains = []
    
    for item in batch:
        # Extract text and domain
        texts.append(item.pop('text'))
        if 'domain' in item:
            domains.append(item.pop('domain'))
        
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
    
    if domains:
        batched['domains'] = domains
    
    # Add graph data
    for k, v in graph_data.items():
        batched[k] = v
    
    return batched


def create_ethics_dataloaders(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    domains: Optional[List[str]] = None,
    use_graphbrain: bool = True,
    parser_lang: str = "en",
    cache_graphs: bool = True,
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
        use_graphbrain: Whether to use GraphBrain
        parser_lang: Language for GraphBrain parser
        cache_graphs: Whether to cache processed graphs
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
            max_length=max_length,
            use_graphbrain=use_graphbrain,
            parser_lang=parser_lang,
            cache_graphs=cache_graphs
        )
    
    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_with_graphs,
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
            max_length=max_length,
            use_graphbrain=use_graphbrain,
            parser_lang=parser_lang,
            cache_graphs=cache_graphs
        )
        
        dataloaders["ambiguous"] = DataLoader(
            ambiguous_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_with_graphs,
            num_workers=num_workers,
            pin_memory=True
        )
    except Exception as e:
        print(f"Ambiguous dataset not available: {e}")
    
    return dataloaders
