"""
Advanced Graph-Based Ethical Reasoning

This module provides advanced graph reasoning capabilities for ethical analysis,
integrating GraphBrain semantic hypergraphs with PyTorch Geometric GNNs
for improved understanding of ethical relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import graphbrain as gb
from graphbrain import hgraph
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import plotly.graph_objects as go
import plotly.express as px


class EthicalRelationExtractor:
    """
    Extracts ethical relations from text using GraphBrain.
    
    This class analyzes text to identify ethical concepts, actors, actions,
    and their relationships in a structured format suitable for graph processing.
    
    Args:
        parser_lang: Language for GraphBrain parser
    """
    
    def __init__(self, parser_lang: str = "en"):
        # Initialize GraphBrain parser
        try:
            self.parser = gb.Parser(model=f"{parser_lang}_core_web_sm")
        except Exception as e:
            print(f"Error initializing GraphBrain parser: {e}")
            self.parser = None
        
        # Ethical concept dictionaries
        self.moral_actions = {
            "positive": ["help", "assist", "support", "protect", "save", "benefit", 
                         "contribute", "donate", "volunteer", "share", "cooperate"],
            "negative": ["harm", "hurt", "damage", "destroy", "kill", "injure", "steal", 
                         "cheat", "lie", "deceive", "manipulate", "exploit"]
        }
        
        self.moral_values = {
            "care": ["care", "compassion", "kindness", "empathy", "sympathy", "help"],
            "fairness": ["fair", "just", "equal", "equitable", "rights", "deserve"],
            "loyalty": ["loyal", "faithful", "committed", "patriotic", "solidarity"],
            "authority": ["respect", "obedient", "tradition", "honor", "duty"],
            "purity": ["pure", "sacred", "natural", "clean", "innocent"],
            "liberty": ["freedom", "liberty", "autonomy", "choice", "independence"]
        }
        
        self.consequences = {
            "positive": ["benefit", "advantage", "gain", "profit", "reward", "improvement"],
            "negative": ["harm", "damage", "loss", "cost", "penalty", "deterioration"]
        }
    
    def _get_graph(self, text: str) -> Optional[hgraph]:
        """Parse text to GraphBrain hypergraph."""
        if self.parser is None:
            return None
            
        try:
            hg = hgraph()
            
            for sentence in text.split('.'):
                if sentence.strip():
                    parse = self.parser.parse(sentence)
                    hg.add(parse)
                    
            return hg
        except Exception as e:
            print(f"Error parsing text: {e}")
            return None
    
    def is_moral_word(self, word: str) -> Tuple[bool, str]:
        """Check if a word has moral connotations."""
        word_lower = word.lower()
        
        # Check actions
        for category, words in self.moral_actions.items():
            if any(moral_word in word_lower for moral_word in words):
                return True, f"action_{category}"
        
        # Check values
        for value, words in self.moral_values.items():
            if any(moral_word in word_lower for moral_word in words):
                return True, f"value_{value}"
        
        # Check consequences
        for category, words in self.consequences.items():
            if any(consequence in word_lower for consequence in words):
                return True, f"consequence_{category}"
        
        return False, ""
    
    def extract_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract ethical relations from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of ethical relations
        """
        hg = self._get_graph(text)
        if hg is None:
            return {"error": "Failed to parse text"}
        
        # Extract entities and relations
        entities = {
            "actors": set(),
            "actions": set(),
            "values": set(),
            "consequences": set()
        }
        
        relations = []
        
        # Process all edges in the graph
        for edge in hg.edges():
            if len(edge) <= 1:
                continue
                
            # Extract predicate (action)
            if gb.is_atom(edge[0]):
                predicate = str(edge[0])
                
                # Check if the predicate has moral connotations
                is_moral, moral_type = self.is_moral_word(predicate)
                
                if is_moral:
                    if moral_type.startswith("action_"):
                        entities["actions"].add((predicate, moral_type.split("_")[1]))
                    elif moral_type.startswith("value_"):
                        entities["values"].add((predicate, moral_type.split("_")[1]))
                    elif moral_type.startswith("consequence_"):
                        entities["consequences"].add((predicate, moral_type.split("_")[1]))
                else:
                    entities["actions"].add((predicate, "neutral"))
                
                # Extract subject (actor)
                if len(edge) > 1 and gb.is_atom(edge[1]):
                    subject = str(edge[1])
                    entities["actors"].add(subject)
                    
                    # Add relation between actor and action
                    relations.append({
                        "source": subject,
                        "relation": predicate,
                        "target": "",
                        "type": "actor_action"
                    })
                
                # Extract object
                if len(edge) > 2:
                    for i in range(2, len(edge)):
                        if gb.is_atom(edge[i]):
                            obj = str(edge[i])
                            
                            # Check if object has moral connotations
                            is_moral_obj, moral_type_obj = self.is_moral_word(obj)
                            
                            if is_moral_obj:
                                if moral_type_obj.startswith("value_"):
                                    entities["values"].add((obj, moral_type_obj.split("_")[1]))
                                elif moral_type_obj.startswith("consequence_"):
                                    entities["consequences"].add((obj, moral_type_obj.split("_")[1]))
                            
                            # Add relation between action and object
                            relations.append({
                                "source": predicate,
                                "relation": "affects",
                                "target": obj,
                                "type": "action_object"
                            })
        
        # Convert sets to lists
        result = {
            "actors": list(entities["actors"]),
            "actions": list(entities["actions"]),
            "values": list(entities["values"]),
            "consequences": list(entities["consequences"]),
            "relations": relations
        }
        
        return result
    
    def to_pyg_data(self, relations: Dict[str, Any]) -> Data:
        """
        Convert extracted relations to PyTorch Geometric Data object.
        
        Args:
            relations: Dictionary of ethical relations
            
        Returns:
            PyTorch Geometric Data object
        """
        # Build node dictionary
        nodes = []
        node_types = []
        node_features = []
        
        # Add actors
        for actor in relations["actors"]:
            nodes.append(actor)
            node_types.append(0)  # Type 0: Actor
            node_features.append([1, 0, 0, 0])  # One-hot encoded type
        
        # Add actions
        for action, sentiment in relations["actions"]:
            nodes.append(action)
            node_types.append(1)  # Type 1: Action
            
            # Feature: [is_actor, is_action, is_value, is_consequence]
            feature = [0, 1, 0, 0]
            
            # Add sentiment information
            if sentiment == "positive":
                feature.append(1)
            elif sentiment == "negative":
                feature.append(-1)
            else:
                feature.append(0)
                
            node_features.append(feature)
        
        # Add values
        for value, category in relations["values"]:
            nodes.append(value)
            node_types.append(2)  # Type 2: Value
            
            # One-hot encoded type with additional dimension for sentiment
            feature = [0, 0, 1, 0, 0]  # Neutral sentiment by default
            node_features.append(feature)
        
        # Add consequences
        for consequence, sentiment in relations["consequences"]:
            nodes.append(consequence)
            node_types.append(3)  # Type 3: Consequence
            
            # Base feature
            feature = [0, 0, 0, 1]
            
            # Add sentiment information
            if sentiment == "positive":
                feature.append(1)
            elif sentiment == "negative":
                feature.append(-1)
            else:
                feature.append(0)
                
            node_features.append(feature)
        
        # Create node index mapping
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Create edges
        edge_indices = []
        edge_types = []
        
        for relation in relations["relations"]:
            source = relation["source"]
            target = relation["target"]
            
            # Skip if nodes not in map
            if source not in node_map or (target and target not in node_map):
                continue
            
            source_idx = node_map[source]
            
            if target:
                target_idx = node_map[target]
                
                # Add bidirectional edges
                edge_indices.append((source_idx, target_idx))
                edge_indices.append((target_idx, source_idx))
                
                # Edge types
                if relation["type"] == "actor_action":
                    edge_types.extend([0, 0])  # Type 0: Actor-Action
                elif relation["type"] == "action_object":
                    edge_types.extend([1, 1])  # Type 1: Action-Object
                else:
                    edge_types.extend([2, 2])  # Type 2: Other
        
        # Convert to PyTorch tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            # Empty graph fallback
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        
        # Node features and types
        x = torch.tensor(node_features, dtype=torch.float)
        node_type = torch.tensor(node_types, dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            num_nodes=len(nodes)
        )
        
        # Add original nodes for reference
        data.nodes = nodes
        
        return data


class EthicalGNN(nn.Module):
    """
    Graph Neural Network for ethical reasoning.
    
    This model processes ethical relationship graphs to derive insights
    about ethical implications, using heterogeneous graph convolutions.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
        num_layers: Number of graph convolutional layers
        conv_type: Type of graph convolution to use
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 3,
        conv_type: str = "gat",
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node embedding layer
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Edge embedding
        self.edge_embedding = nn.Embedding(3, hidden_channels)  # 3 edge types
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Choose convolution type
        self.conv_type = conv_type
        
        for i in range(num_layers):
            in_dim = hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            if conv_type == "gcn":
                conv = GCNConv(in_dim, out_dim)
            elif conv_type == "gat":
                conv = GATConv(in_dim, out_dim, heads=4, concat=False)
            elif conv_type == "gin":
                nn_layer = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_dim)
                )
                conv = GINConv(nn_layer)
            elif conv_type == "sage":
                conv = SAGEConv(in_dim, out_dim)
            elif conv_type == "transformer":
                conv = TransformerConv(in_dim, out_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unknown convolution type: {conv_type}")
                
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        # Ethical assessment layers
        self.ethics_scorer = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )
        
        self.manipulation_scorer = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Sentiment analysis for actions/consequences
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 3)  # 3 classes: negative, neutral, positive
        )
        
        # Moral foundation classifier
        self.moral_foundation_classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 6)  # 6 moral foundations
        )
    
    def forward(self, data: Union[Data, Batch]) -> Dict[str, Any]:
        """
        Forward pass through the ethical GNN.
        
        Args:
            data: PyTorch Geometric Data or Batch
            
        Returns:
            Dictionary of model outputs
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial node encoding
        x = self.node_encoder(x)
        
        # Apply graph convolutions
        for i in range(self.num_layers):
            # Apply convolution
            x = self.convs[i](x, edge_index)
            
            # Apply batch normalization
            x = self.batch_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if hasattr(data, 'batch'):
            # If batched, use global pooling
            node_embeddings = x
            
            # Different pooling strategies
            graph_embedding_mean = global_mean_pool(x, data.batch)
            graph_embedding_max = global_max_pool(x, data.batch)
            graph_embedding_sum = global_add_pool(x, data.batch)
            
            # Combine pooling methods
            graph_embedding = (graph_embedding_mean + graph_embedding_max + graph_embedding_sum) / 3
        else:
            # If single graph, use mean pooling
            node_embeddings = x
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Compute ethics and manipulation scores
        ethics_score = self.ethics_scorer(graph_embedding)
        manipulation_score = self.manipulation_scorer(graph_embedding)
        
        # Node-level analysis
        if hasattr(data, 'node_type'):
            # Get node types
            node_types = data.node_type
            
            # Group nodes by type
            actors_mask = node_types == 0
            actions_mask = node_types == 1
            values_mask = node_types == 2
            consequences_mask = node_types == 3
            
            # Analysis by node type
            actor_embeddings = node_embeddings[actors_mask] if torch.any(actors_mask) else None
            action_embeddings = node_embeddings[actions_mask] if torch.any(actions_mask) else None
            value_embeddings = node_embeddings[values_mask] if torch.any(values_mask) else None
            consequence_embeddings = node_embeddings[consequences_mask] if torch.any(consequences_mask) else None
            
            # Sentiment analysis for actions
            action_sentiments = None
            if action_embeddings is not None and len(action_embeddings) > 0:
                action_sentiments = self.sentiment_classifier(action_embeddings)
            
            # Moral foundation analysis for values
            moral_foundations = None
            if value_embeddings is not None and len(value_embeddings) > 0:
                moral_foundations = self.moral_foundation_classifier(value_embeddings)
        
        # Compile results
        results = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'ethics_score': ethics_score,
            'manipulation_score': manipulation_score
        }
        
        # Add node-level analysis if available
        if hasattr(data, 'node_type'):
            results.update({
                'actor_embeddings': actor_embeddings,
                'action_embeddings': action_embeddings,
                'value_embeddings': value_embeddings,
                'consequence_embeddings': consequence_embeddings,
                'action_sentiments': action_sentiments,
                'moral_foundations': moral_foundations
            })
        
        return results


class EthicalRelationReasoning(nn.Module):
    """
    Advanced ethical relation reasoning using combined text and graph representations.
    
    This module performs graph-based reasoning over ethical relationships
    extracted from text, providing deeper insights into ethical implications.
    
    Args:
        d_model: Input embedding dimension
        gnn_hidden_dim: Hidden dimension for GNN
        gnn_output_dim: Output dimension for GNN
        gnn_num_layers: Number of GNN layers
        gnn_conv_type: Type of GNN convolution
        dropout: Dropout rate
        parser_lang: Language for GraphBrain parser
    """
    
    def __init__(
        self,
        d_model: int = 512,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 32,
        gnn_num_layers: int = 3,
        gnn_conv_type: str = "gat",
        dropout: float = 0.3,
        parser_lang: str = "en"
    ):
        super().__init__()
        
        # Relation extractor
        self.relation_extractor = EthicalRelationExtractor(parser_lang)
        
        # Ethical GNN
        self.gnn = EthicalGNN(
            in_channels=5,  # Feature dimension from relation_extractor
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_output_dim,
            num_layers=gnn_num_layers,
            conv_type=gnn_conv_type,
            dropout=dropout
        )
        
        # Integration layers
        self.graph_to_text = nn.Linear(gnn_output_dim, d_model)
        self.integration = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        text_batch: List[str],
        text_embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Process a batch of texts with graph reasoning.
        
        Args:
            text_batch: Batch of input texts
            text_embeddings: Text embeddings from language model
            
        Returns:
            Dictionary of model outputs
        """
        batch_size = len(text_batch)
        
        # Process each text to extract relations
        graph_data_list = []
        for text in text_batch:
            # Extract relations
            relations = self.relation_extractor.extract_relations(text)
            
            # Convert to PyG data
            graph_data = self.relation_extractor.to_pyg_data(relations)
            
            graph_data_list.append(graph_data)
        
        # Process graphs with GNN
        if graph_data_list:
            # Create batch
            batched_data = Batch.from_data_list(graph_data_list)
            
            # Run GNN
            gnn_outputs = self.gnn(batched_data)
            
            # Get graph embeddings
            graph_embeddings = gnn_outputs['graph_embedding']
            
            # Project to text embedding space
            projected_graph_embeddings = self.graph_to_text(graph_embeddings)
            
            # Reshape for integration with text
            projected_graph_embeddings = projected_graph_embeddings.view(batch_size, 1, -1)
            projected_graph_embeddings = projected_graph_embeddings.expand(-1, text_embeddings.size(1), -1)
            
            # Integrate with text embeddings
            combined_embeddings = torch.cat([text_embeddings, projected_graph_embeddings], dim=-1)
            integrated_embeddings = self.integration(combined_embeddings)
            
            return {
                'integrated_embeddings': integrated_embeddings,
                'graph_embeddings': graph_embeddings,
                'gnn_outputs': gnn_outputs
            }
        else:
            # If no graphs, return original embeddings
            return {
                'integrated_embeddings': text_embeddings,
                'graph_embeddings': None,
                'gnn_outputs': None
            }


class GraphReasoningEthicsModel(nn.Module):
    """
    Enhanced ethics model with advanced graph-based reasoning.
    
    This model incorporates GraphBrain-based ethical relation extraction
    and PyTorch Geometric graph neural networks for deeper ethical analysis.
    
    Args:
        base_model: Base ethics model
        d_model: Embedding dimension
        gnn_hidden_dim: Hidden dimension for GNN
        gnn_output_dim: Output dimension for GNN
        gnn_num_layers: Number of GNN layers
        gnn_conv_type: Type of GNN convolution
        parser_lang: Language for GraphBrain parser
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        d_model: int = 512,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 32,
        gnn_num_layers: int = 3,
        gnn_conv_type: str = "gat",
        parser_lang: str = "en"
    ):
        super().__init__()
        
        self.base_model = base_model
        
        # Ethical relation reasoning module
        self.relation_reasoning = EthicalRelationReasoning(
            d_model=d_model,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_conv_type=gnn_conv_type,
            parser_lang=parser_lang
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with graph-based reasoning.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            embeddings: Text embeddings from language model
            texts: Input texts (required for graph processing)
            **kwargs: Additional arguments for base model
            
        Returns:
            Dictionary of model outputs
        """
        # Initial text processing by base model
        if embeddings is None and hasattr(self.base_model, 'embedding') and input_ids is not None:
            # Create embeddings from input_ids
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            token_embeddings = self.base_model.embedding(input_ids)
            position_embeddings = self.base_model.position_embedding(position_ids)
            embeddings = token_embeddings + position_embeddings
            embeddings = self.base_model.layer_norm(embeddings)
            embeddings = self.base_model.dropout(embeddings)
        
        # Early transformer layers
        if hasattr(self.base_model, 'transformer_layers'):
            for i, layer in enumerate(self.base_model.transformer_layers):
                # Process first half of transformer layers
                if i < len(self.base_model.transformer_layers) // 2:
                    embeddings = layer(embeddings, src_key_padding_mask=attention_mask)
        
        # Process with graph reasoning if texts provided
        if texts is not None and len(texts) > 0:
            reasoning_outputs = self.relation_reasoning(texts, embeddings)
            enhanced_embeddings = reasoning_outputs['integrated_embeddings']
            
            # Process remaining transformer layers
            if hasattr(self.base_model, 'transformer_layers'):
                for i, layer in enumerate(self.base_model.transformer_layers):
                    # Process second half of transformer layers
                    if i >= len(self.base_model.transformer_layers) // 2:
                        enhanced_embeddings = layer(enhanced_embeddings, src_key_padding_mask=attention_mask)
            
            # Process through the rest of the base model
            kwargs['embeddings'] = enhanced_embeddings
            kwargs['attention_mask'] = attention_mask
            base_outputs = self.base_model(**kwargs)
            
            # Add graph reasoning outputs
            base_outputs['graph_reasoning'] = reasoning_outputs
            
            return base_outputs
        else:
            # Standard processing without graph reasoning
            return self.base_model(embeddings=embeddings, attention_mask=attention_mask, **kwargs)


class GraphVisualizer:
    """
    Visualizes ethical relationship graphs for explainability.
    
    This class creates interactive visualizations of ethical graphs
    to explain the reasoning process.
    """
    
    @staticmethod
    def visualize_ethical_graph(
        relations: Dict[str, Any],
        title: str = "Ethical Relationship Graph"
    ) -> go.Figure:
        """
        Create an interactive visualization of ethical relationships.
        
        Args:
            relations: Dictionary of ethical relations
            title: Plot title
            
        Returns:
            Plotly figure with graph visualization
        """
        # Extract nodes and edges
        nodes = set()
        node_types = {}
        node_sentiments = {}
        edges = []
        
        # Add actors
        for actor in relations["actors"]:
            nodes.add(actor)
            node_types[actor] = "actor"
            node_sentiments[actor] = "neutral"
        
        # Add actions
        for action, sentiment in relations["actions"]:
            nodes.add(action)
            node_types[action] = "action"
            node_sentiments[action] = sentiment
        
        # Add values
        for value, category in relations["values"]:
            nodes.add(value)
            node_types[value] = f"value_{category}"
            node_sentiments[value] = "neutral"
        
        # Add consequences
        for consequence, sentiment in relations["consequences"]:
            nodes.add(consequence)
            node_types[consequence] = "consequence"
            node_sentiments[consequence] = sentiment
        
        # Add edges
        for relation in relations["relations"]:
            source = relation["source"]
            target = relation["target"]
            relation_type = relation["type"]
            
            if source in nodes and (not target or target in nodes):
                if target:
                    edges.append((source, target, relation_type))
        
        # Convert nodes to list
        nodes = list(nodes)
        
        # Create layout
        layout = {}
        
        # Use a circle layout
        n = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            layout[node] = (np.cos(angle), np.sin(angle))
        
        # Color mapping
        node_color_map = {
            "actor": "blue",
            "action": "red",
            "consequence": "purple"
        }
        
        # Add colors for value types
        for foundation in ["care", "fairness", "loyalty", "authority", "purity", "liberty"]:
            node_color_map[f"value_{foundation}"] = "green"
        
        # Edge color mapping
        edge_color_map = {
            "actor_action": "gray",
            "action_object": "black"
        }
        
        # Sentiment color modifiers
        sentiment_colors = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in nodes:
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Get base color from type
            base_color = node_color_map.get(node_types[node], "gray")
            
            # Adjust color by sentiment
            sentiment = node_sentiments[node]
            
            node_colors.append(base_color)
            
            # Adjust size by type
            if node_types[node] == "actor":
                node_sizes.append(15)
            elif node_types[node] == "action" or node_types[node].startswith("value_"):
                node_sizes.append(12)
            else:
                node_sizes.append(10)
        
        # Create edge traces
        edge_traces = []
        
        for source, target, edge_type in edges:
            x0, y0 = layout[source]
            x1, y1 = layout[target]
            
            # Get edge color
            edge_color = edge_color_map.get(edge_type, "gray")
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color=edge_color),
                hoverinfo='none'
            )
            
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=node_text
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=800
            )
        )
        
        # Add legend for node types
        for node_type, color in node_color_map.items():
            display_name = node_type.replace("_", ": ").title()
            
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                showlegend=True,
                name=display_name
            ))
        
        return fig
    
    @staticmethod
    def visualize_moral_foundations(
        moral_scores: Dict[str, float],
        title: str = "Moral Foundations Analysis"
    ) -> go.Figure:
        """
        Create a radar chart showing moral foundation scores.
        
        Args:
            moral_scores: Dictionary of moral foundation scores
            title: Plot title
            
        Returns:
            Plotly figure with radar chart
        """
        # Ensure all foundations are present
        foundations = ["care", "fairness", "loyalty", "authority", "purity", "liberty"]
        scores = [moral_scores.get(f, 0.0) for f in foundations]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=foundations,
            fill='toself',
            name='Moral Foundations'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=title,
            width=700,
            height=500
        )
        
        return fig
