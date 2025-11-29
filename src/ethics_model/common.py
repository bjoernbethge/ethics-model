"""
Common Utilities for Ethics Model

This module provides shared utilities used across different components
of the ethics model to avoid code duplication.
"""

import torch
from typing import Dict, List, Any, Optional
import graphbrain as gb
from graphbrain import hgraph


def collate_with_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batches with graph data.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched tensors and graph data
    """
    # Separate standard items and graph data
    standard_items: Dict[str, List[Any]] = {}
    graph_data: Dict[str, List[Any]] = {}
    texts: List[str] = []
    domains: List[str] = []
    
    for item in batch:
        # Extract text
        if 'text' in item:
            texts.append(item.pop('text'))
        
        # Extract domain if present
        if 'domain' in item:
            domains.append(item.pop('domain'))
        
        # Separate graph data from standard items
        graph_keys = [k for k in item.keys() if k.startswith('graph_')]
        
        for k in list(item.keys()):
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


class GraphBrainParserManager:
    """
    Manages GraphBrain parser instances to avoid duplicate initialization code.
    """
    
    _parsers: Dict[str, Any] = {}
    
    @classmethod
    def get_parser(cls, parser_lang: str = "en") -> Optional[Any]:
        """
        Get or create a GraphBrain parser for the specified language.
        
        Args:
            parser_lang: Language for the parser (e.g., "en")
            
        Returns:
            GraphBrain parser instance or None if initialization fails
        """
        if parser_lang not in cls._parsers:
            try:
                cls._parsers[parser_lang] = gb.Parser(model=f"{parser_lang}_core_web_sm")
            except Exception as e:
                print(f"Error initializing GraphBrain parser: {e}")
                print(f"Make sure you have the {parser_lang}_core_web_sm model installed:")
                print(f"python -m spacy download {parser_lang}_core_web_sm")
                return None
        return cls._parsers[parser_lang]


def process_text_to_hypergraph(
    text: str,
    parser: Optional[Any] = None,
    parser_lang: str = "en"
) -> Optional[hgraph]:
    """
    Process text to create a GraphBrain hypergraph.
    
    Args:
        text: Text to process
        parser: Optional pre-initialized parser (if None, uses shared parser)
        parser_lang: Language for the parser if parser is None
        
    Returns:
        GraphBrain hypergraph or None if processing failed
    """
    if parser is None:
        parser = GraphBrainParserManager.get_parser(parser_lang)
    
    if parser is None:
        return None
        
    try:
        hg_instance = hgraph()
        
        # Parse text and add to hypergraph
        for sentence in text.split('.'):
            if sentence.strip():
                parse = parser.parse(sentence)
                hg_instance.add(parse)
                
        return hg_instance
    except Exception as e:
        print(f"Error processing text to graph: {e}")
        return None


def prepare_graph_data_for_model(hg_instance: Optional[hgraph]) -> Dict[str, Any]:
    """
    Prepare graph data from a hypergraph for model input.
    
    Args:
        hg_instance: GraphBrain hypergraph
        
    Returns:
        Dictionary with graph data ready for model consumption
    """
    if hg_instance is None:
        return {'has_graph': False}
        
    try:
        # Extract nodes (atoms)
        nodes = list(hg_instance.all_atoms())
        
        # Create node map for edge index creation
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Create edge index
        edge_index = []
        edge_types = []
        
        # Extract edges
        for edge in hg_instance.all_edges():
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
        # When nodes is empty, create a tensor with 0 rows but consistent feature dimension
        if nodes:
            node_features = torch.eye(len(nodes))
        else:
            node_features = torch.zeros((0, 1), dtype=torch.float)
        
        return {
            'has_graph': True,
            'nodes': nodes,
            'node_features': node_features,
            'edge_index': edge_index_tensor,
            'edge_types': edge_types_tensor
        }
    except Exception as e:
        print(f"Error preparing graph data: {e}")
        return {'has_graph': False}
