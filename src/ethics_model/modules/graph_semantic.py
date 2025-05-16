"""
Semantic Graph Processing with GraphBrain

This module provides integration with GraphBrain to create semantic hypergraphs
from text for enhanced ethical and narrative analysis.
"""

import torch
import torch.nn as nn
import graphbrain as gb
from graphbrain import hgraph
from typing import Dict, List, Optional, Tuple, Any, Union
from torch_geometric.data import Data

class HypergraphConverter:
    """
    Converts GraphBrain hypergraphs to formats compatible with PyTorch Geometric.
    """
    
    @staticmethod
    def hypergraph_to_pyg_data(hg: hgraph, concept_embeddings: Optional[Dict[str, torch.Tensor]] = None) -> Data:
        """
        Converts a GraphBrain hypergraph to a PyTorch Geometric Data object.
        
        Args:
            hg: GraphBrain hypergraph
            concept_embeddings: Optional dictionary mapping concept strings to embeddings
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract nodes and edges
        nodes = list(hg.all_atoms())
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Create edge index
        edge_index = []
        
        # Process hyperedges
        for edge in hg.all_edges():
            # For each hyperedge, create binary edges between the primary node and others
            pred = edge[0]  # Predicate is typically the first atom in a hyperedge
            
            if gb.is_atom(pred) and len(edge) > 1:
                pred_idx = node_map[pred]
                
                # Connect predicate to all arguments
                for i in range(1, len(edge)):
                    if gb.is_atom(edge[i]):
                        arg_idx = node_map[edge[i]]
                        # Add bidirectional connections
                        edge_index.append([pred_idx, arg_idx])
                        edge_index.append([arg_idx, pred_idx])
        
        # Create node features
        if concept_embeddings is not None:
            node_features = []
            default_dim = next(iter(concept_embeddings.values())).size(0) if concept_embeddings else 300
            default_embedding = torch.zeros(default_dim)
            
            for node in nodes:
                node_str = str(node)
                if node_str in concept_embeddings:
                    node_features.append(concept_embeddings[node_str])
                else:
                    node_features.append(default_embedding)
                    
            node_features = torch.stack(node_features)
        else:
            # Simple one-hot encoding if no embeddings provided
            node_features = torch.eye(len(nodes))
        
        # Create PyG data object
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        data = Data(
            x=node_features,
            edge_index=edge_index_tensor
        )
        
        return data


class SemanticGraphProcessor(nn.Module):
    """
    Neural module that processes text using GraphBrain to create and analyze
    semantic hypergraphs for ethical reasoning.
    """
    
    def __init__(self, 
                 d_model: int,
                 parser_lang: str = "en",
                 edge_type_embedding_dim: int = 32):
        super().__init__()
        
        # Initialize GraphBrain parser
        self.parser_lang = parser_lang
        self.parser = None  # Lazy initialization in forward to avoid loading at init time
        
        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(50, edge_type_embedding_dim)  # 50 common edge types
        
        # Projection layers
        self.node_projection = nn.Linear(d_model, d_model)
        self.graph_projection = nn.Linear(d_model, d_model)
        
        # GNN layers to be defined in derived classes
        self.converter = HypergraphConverter()
        
    def _init_parser(self):
        """Lazy initialize the parser when first needed"""
        if self.parser is None:
            self.parser = gb.Parser(model=f"{self.parser_lang}_core_web_sm")
    
    def process_text(self, text: str) -> hgraph:
        """
        Process text to a GraphBrain hypergraph.
        
        Args:
            text: Text to process
            
        Returns:
            GraphBrain hypergraph
        """
        self._init_parser()
        hg = hgraph()
        
        # Parse text and add to hypergraph
        for sentence in text.split('.'):
            if sentence.strip():
                parse = self.parser.parse(sentence)
                hg.add(parse)
                
        return hg
    
    def extract_ethical_relations(self, hg: hgraph) -> Dict[str, List[Any]]:
        """
        Extract ethically relevant relations from hypergraph.
        
        Args:
            hg: GraphBrain hypergraph
            
        Returns:
            Dictionary of ethical relations
        """
        ethical_relations = {
            'actions': [],
            'consequences': [],
            'actors': [],
            'values': [],
            'obligations': []
        }
        
        # Find agent-action patterns (who did what)
        for edge in hg.edges():
            if edge[0].type() == 'P':  # Predicate
                # Extract agents (subjects)
                agents = []
                actions = []
                
                # Simple agent-action extraction
                for i in range(1, len(edge)):
                    if gb.is_atom(edge[i]) and edge[i].type() == 'C':  # Concept
                        if i == 1:  # First argument often the agent
                            agents.append(edge[i])
                        else:
                            actions.append(edge[i])
                
                if agents:
                    ethical_relations['actors'].extend(agents)
                if actions:
                    ethical_relations['actions'].extend(actions)
                
        return ethical_relations
    
    def forward(self, 
                text_batch: List[str], 
                embeddings: torch.Tensor,
                return_graphs: bool = False) -> Dict[str, Any]:
        """
        Process a batch of texts using GraphBrain.
        
        Args:
            text_batch: Batch of text strings
            embeddings: Text embeddings (batch_size, seq_len, d_model)
            return_graphs: Whether to return the raw graphs
            
        Returns:
            Dictionary containing:
                - graph_embeddings: Graph-enhanced embeddings
                - ethical_relations: Extracted ethical relations
                - graphs: Raw GraphBrain graphs (if return_graphs=True)
        """
        batch_size = len(text_batch)
        self._init_parser()
        
        # Process each text in batch
        graphs = []
        ethical_relations_batch = []
        graph_data_batch = []
        
        for i, text in enumerate(text_batch):
            # Create hypergraph
            hg = self.process_text(text)
            graphs.append(hg)
            
            # Extract ethical relations
            relations = self.extract_ethical_relations(hg)
            ethical_relations_batch.append(relations)
            
            # Convert to PyG format using text embeddings
            # Use mean pooled embeddings as node features
            mean_embedding = embeddings[i].mean(dim=0)
            concept_embeddings = {}
            
            # Create simple concept embeddings for now - would be more sophisticated in practice
            for atom in hg.all_atoms():
                concept_embeddings[str(atom)] = mean_embedding
                
            graph_data = self.converter.hypergraph_to_pyg_data(
                hg,
                concept_embeddings=concept_embeddings
            )
            graph_data_batch.append(graph_data)
        
        # Process graph embeddings - this implementation is a placeholder
        # In a full implementation, you'd run actual GNN layers here
        graph_embeddings = embeddings  # Placeholder - replace with actual GNN processing
        
        result = {
            'graph_embeddings': graph_embeddings,
            'ethical_relations': ethical_relations_batch,
        }
        
        if return_graphs:
            result['graphs'] = graphs
            result['graph_data'] = graph_data_batch
            
        return result
