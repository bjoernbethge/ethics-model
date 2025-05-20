"""
Advanced Graph-Based Ethical Reasoning

This module provides advanced graph reasoning capabilities for ethical analysis,
using NetworkX graphs and spaCy for NLP processing to understand
ethical relationships and dependencies in text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import networkx as nx
import spacy
from spacy.tokens import Doc, Token, Span
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch


class EthicalRelationExtractor:
    """
    Extracts ethical relations from text using spaCy NLP pipeline.
    
    This class analyzes text to identify ethical concepts, actors, actions,
    and their relationships using dependency parsing and named entity recognition.
    
    Args:
        model_name: spaCy model name (e.g., "en_core_web_sm")
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: Could not load spaCy model '{model_name}'. "
                  f"Please install with: python -m spacy download {model_name}")
            self.nlp = spacy.blank("en")
        # Sentencizer hinzufügen, falls nicht vorhanden
        if "sentencizer" not in self.nlp.pipe_names:
            try:
                self.nlp.add_pipe("sentencizer")
            except Exception as e:
                print(f"Warning: Could not add sentencizer: {e}")
        
        # Ethical concept dictionaries
        self.moral_actions = {
            "positive": ["help", "assist", "support", "protect", "save", "benefit", 
                         "contribute", "donate", "volunteer", "share", "cooperate", "aid",
                         "care", "nurture", "heal", "teach", "guide", "encourage"],
            "negative": ["harm", "hurt", "damage", "destroy", "kill", "injure", "steal", 
                         "cheat", "lie", "deceive", "manipulate", "exploit", "abuse",
                         "betray", "abandon", "neglect", "oppress", "discriminate"]
        }
        
        self.moral_values = {
            "care": ["care", "compassion", "kindness", "empathy", "sympathy", "help", 
                     "nurture", "welfare", "wellbeing", "concern"],
            "fairness": ["fair", "just", "equal", "equitable", "rights", "deserve",
                         "justice", "equality", "impartial", "unbiased"],
            "loyalty": ["loyal", "faithful", "committed", "patriotic", "solidarity",
                        "devotion", "allegiance", "unity", "brotherhood"],
            "authority": ["respect", "obedient", "tradition", "honor", "duty",
                          "hierarchy", "leadership", "order", "discipline"],
            "purity": ["pure", "sacred", "natural", "clean", "innocent",
                       "sanctity", "divine", "holy", "pristine"],
            "liberty": ["freedom", "liberty", "autonomy", "choice", "independence",
                        "self-determination", "rights", "sovereignty"]
        }
        
        self.emotional_indicators = {
            "positive": ["happy", "joy", "love", "hope", "grateful", "proud", "satisfied",
                         "excited", "pleased", "content", "optimistic"],
            "negative": ["angry", "sad", "fear", "hate", "disgusted", "ashamed", "guilty",
                         "anxious", "worried", "frustrated", "disappointed"]
        }
        
        # Moral entities - types that carry moral weight
        self.moral_entities = ["PERSON", "ORG", "GPE", "NORP"]  # People, organizations, places, groups
    
    def _categorize_word(self, word: str, pos: str = None) -> Tuple[bool, str, str]:
        """
        Categorize a word for moral significance.
        
        Args:
            word: The word to categorize
            pos: Part of speech tag
            
        Returns:
            Tuple of (is_moral, category, sentiment)
        """
        word_lower = word.lower()
        
        # Check moral actions
        for sentiment, actions in self.moral_actions.items():
            if any(action in word_lower for action in actions):
                return True, "action", sentiment
        
        # Check moral values
        for value, words in self.moral_values.items():
            if any(moral_word in word_lower for moral_word in words):
                return True, f"value_{value}", "neutral"
        
        # Check emotional indicators
        for sentiment, emotions in self.emotional_indicators.items():
            if any(emotion in word_lower for emotion in emotions):
                return True, "emotion", sentiment
        
        return False, "neutral", "neutral"
    
    def _extract_dependencies(self, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract dependency relationships from spaCy doc."""
        dependencies = []
        
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass", "dobj", "iobj", "pobj", "amod", "compound"]:
                head_text = token.head.text.lower()
                token_text = token.text.lower()
                
                # Check if either token or head has moral significance
                token_moral = self._categorize_word(token_text, token.pos_)
                head_moral = self._categorize_word(head_text, token.head.pos_)
                
                if token_moral[0] or head_moral[0]:
                    dependencies.append((head_text, token.dep_, token_text))
        
        return dependencies
    
    def _extract_entities(self, doc: Doc) -> Dict[str, List[Tuple[str, str]]]:
        """Extract and categorize entities from text."""
        entities = {
            "actors": [],
            "actions": [],
            "values": [],
            "emotions": [],
            "locations": [],
            "organizations": []
        }
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in self.moral_entities:
                if ent.label_ == "PERSON":
                    entities["actors"].append((ent.text, "person"))
                elif ent.label_ == "ORG":
                    entities["organizations"].append((ent.text, "organization"))
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append((ent.text, "location"))
                elif ent.label_ == "NORP":
                    entities["actors"].append((ent.text, "group"))
        
        # Categorize tokens by moral significance
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                is_moral, category, sentiment = self._categorize_word(token.text, token.pos_)
                
                if is_moral:
                    if category == "action":
                        entities["actions"].append((token.text, sentiment))
                    elif category.startswith("value_"):
                        value_type = category.split("_")[1]
                        entities["values"].append((token.text, value_type))
                    elif category == "emotion":
                        entities["emotions"].append((token.text, sentiment))
        
        return entities
    
    def create_graph(self, text: str) -> nx.DiGraph:
        """
        Create a NetworkX graph from the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            NetworkX directed graph representing ethical relationships
        """
        doc = self.nlp(text)
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Extract entities and dependencies
        entities = self._extract_entities(doc)
        dependencies = self._extract_dependencies(doc)
        
        # Add nodes with attributes
        node_id = 0
        node_mapping = {}
        
        # Add entity nodes
        for entity_type, entity_list in entities.items():
            for entity, subtype in entity_list:
                if entity not in node_mapping:
                    node_mapping[entity] = node_id
                    graph.add_node(node_id, 
                                   text=entity, 
                                   type=entity_type, 
                                   subtype=subtype,
                                   moral_weight=self._calculate_moral_weight(entity, entity_type))
                    node_id += 1
        
        # Add edges from dependencies
        for head, relation, dependent in dependencies:
            if head in node_mapping and dependent in node_mapping:
                head_id = node_mapping[head]
                dep_id = node_mapping[dependent]
                
                graph.add_edge(head_id, dep_id, 
                               relation=relation,
                               weight=self._calculate_edge_weight(head, dependent, relation))
        
        # Add co-occurrence edges for entities in the same sentence
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
                                       weight=0.5)
        
        return graph
    
    def _calculate_moral_weight(self, text: str, entity_type: str) -> float:
        """Calculate the moral significance of an entity."""
        is_moral, category, sentiment = self._categorize_word(text)
        
        if not is_moral:
            return 0.1
        
        # Base weight by category
        base_weights = {
            "action": 0.8,
            "value_care": 0.9,
            "value_fairness": 0.9,
            "value_loyalty": 0.7,
            "value_authority": 0.6,
            "value_purity": 0.6,
            "value_liberty": 0.8,
            "emotion": 0.5
        }
        
        weight = base_weights.get(category, 0.3)
        
        # Sentiment modifier
        if sentiment == "positive":
            weight *= 1.2
        elif sentiment == "negative":
            weight *= 1.5  # Negative actions often more morally significant
        
        return min(weight, 1.0)
    
    def _calculate_edge_weight(self, source: str, target: str, relation: str) -> float:
        """Calculate the weight of a relationship edge."""
        # Weight by dependency relation type
        relation_weights = {
            "nsubj": 0.9,      # Subject relationship
            "nsubjpass": 0.9,   # Passive subject
            "dobj": 0.8,        # Direct object
            "iobj": 0.7,        # Indirect object
            "amod": 0.6,        # Adjectival modifier
            "compound": 0.5,    # Compound
            "co_occurrence": 0.3
        }
        
        base_weight = relation_weights.get(relation, 0.4)
        
        # Boost weight if both entities are morally significant
        source_moral = self._categorize_word(source)[0]
        target_moral = self._categorize_word(target)[0]
        
        if source_moral and target_moral:
            base_weight *= 1.5
        
        return min(base_weight, 1.0)
    
    def extract_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract ethical relations and return them in a structured format.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted ethical relations
        """
        # Create graph
        graph = self.create_graph(text)
        
        # Extract structured information
        entities = {
            "actors": [],
            "actions": [],
            "values": [],
            "emotions": [],
            "locations": [],
            "organizations": []
        }
        
        relations = []
        
        # Process nodes
        for node_id, node_data in graph.nodes(data=True):
            entity_type = node_data.get("type", "unknown")
            entity_text = node_data.get("text", "")
            entity_subtype = node_data.get("subtype", "")
            
            if entity_type in entities:
                entities[entity_type].append((entity_text, entity_subtype))
        
        # Process edges
        for source, target, edge_data in graph.edges(data=True):
            source_text = graph.nodes[source].get("text", "")
            target_text = graph.nodes[target].get("text", "")
            relation_type = edge_data.get("relation", "")
            
            relations.append({
                "source": source_text,
                "target": target_text,
                "relation": relation_type,
                "weight": edge_data.get("weight", 0.5)
            })
        
        return {
            "entities": entities,
            "relations": relations,
            "graph": graph,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges()
        }
    
    def to_pyg_data(self, relations: Dict[str, Any]) -> Data:
        """
        Convert extracted relations to PyTorch Geometric Data object.
        
        Args:
            relations: Dictionary of ethical relations from extract_relations
            
        Returns:
            PyTorch Geometric Data object
        """
        graph = relations["graph"]
        
        if graph.number_of_nodes() == 0:
            # Return empty graph
            return Data(
                x=torch.zeros((1, 6), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float),
                num_nodes=1
            )
        
        # Create node features
        node_features = []
        node_list = list(graph.nodes())
        
        for node_id in node_list:
            node_data = graph.nodes[node_id]
            
            # Feature vector: [type_one_hot(4), moral_weight(1), sentiment(1)]
            feature = [0, 0, 0, 0]  # [actor, action, value, other]
            
            node_type = node_data.get("type", "unknown")
            if node_type in ["actors", "organizations"]:
                feature[0] = 1
            elif node_type == "actions":
                feature[1] = 1
            elif node_type == "values":
                feature[2] = 1
            else:
                feature[3] = 1
            
            # Add moral weight
            moral_weight = node_data.get("moral_weight", 0.0)
            feature.append(moral_weight)
            
            # Add sentiment encoding
            subtype = node_data.get("subtype", "neutral")
            if subtype == "positive":
                sentiment = 1.0
            elif subtype == "negative":
                sentiment = -1.0
            else:
                sentiment = 0.0
            feature.append(sentiment)
            
            node_features.append(feature)
        
        # Create edge indices and attributes
        edge_indices = []
        edge_attributes = []
        
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        
        for source, target, edge_data in graph.edges(data=True):
            source_idx = node_id_to_idx[source]
            target_idx = node_id_to_idx[target]
            
            # Add both directions for undirected graph representation
            edge_indices.extend([[source_idx, target_idx], [target_idx, source_idx]])
            
            # Edge attributes: [weight, relation_type_encoded]
            weight = edge_data.get("weight", 0.5)
            
            # Encode relation type
            relation = edge_data.get("relation", "unknown")
            relation_encoding = {
                "nsubj": 1.0, "nsubjpass": 0.9, "dobj": 0.8,
                "iobj": 0.7, "amod": 0.6, "compound": 0.5,
                "co_occurrence": 0.3
            }.get(relation, 0.4)
            
            edge_attr = [weight, relation_encoding]
            edge_attributes.extend([edge_attr, edge_attr])  # For both directions
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_list)
        )


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
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 3,
        conv_type: str = "gat",
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # Node embedding layer
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Edge embedding - using edge attributes directly
        self.edge_encoder = nn.Linear(2, hidden_channels) if conv_type == "gat" else None
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            if conv_type == "gcn":
                conv = GCNConv(in_dim, out_dim)
            elif conv_type == "gat":
                conv = GATConv(in_dim, out_dim, heads=4, concat=False, edge_dim=hidden_channels)
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
                conv = TransformerConv(in_dim, out_dim, heads=4, concat=False, edge_dim=hidden_channels)
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
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Initial node encoding
        x = self.node_encoder(x)
        
        # Process edge attributes if available and needed
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Apply graph convolutions
        for i in range(self.num_layers):
            # Apply convolution
            if self.conv_type in ["gat", "transformer"] and edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            
            # Apply batch normalization
            x = self.batch_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if hasattr(data, 'batch') and data.batch is not None:
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
        
        # Moral foundation analysis
        moral_foundations = self.moral_foundation_classifier(graph_embedding)
        
        # Compile results
        results = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'ethics_score': ethics_score,
            'manipulation_score': manipulation_score,
            'moral_foundations': moral_foundations
        }
        
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
        spacy_model: spaCy model name
    """
    
    def __init__(
        self,
        d_model: int = 512,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 32,
        gnn_num_layers: int = 3,
        gnn_conv_type: str = "gat",
        dropout: float = 0.3,
        spacy_model: str = "en_core_web_sm"
    ):
        super().__init__()
        
        # Relation extractor
        self.relation_extractor = EthicalRelationExtractor(spacy_model)
        
        # Ethical GNN
        self.gnn = EthicalGNN(
            in_channels=6,  # Feature dimension from relation_extractor
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
            try:
                batched_data = Batch.from_data_list(graph_data_list)
                
                # Run GNN
                gnn_outputs = self.gnn(batched_data)
                
                # Get graph embeddings
                graph_embeddings = gnn_outputs['graph_embedding']
                
                # Project to text embedding space
                projected_graph_embeddings = self.graph_to_text(graph_embeddings)
                
                # Reshape for integration with text
                if projected_graph_embeddings.dim() == 2:
                    projected_graph_embeddings = projected_graph_embeddings.view(batch_size, 1, -1)
                projected_graph_embeddings = projected_graph_embeddings.expand(-1, text_embeddings.size(1), -1)
                
                # Integrate with text embeddings
                combined_embeddings = torch.cat([text_embeddings, projected_graph_embeddings], dim=-1)
                integrated_embeddings = self.integration(combined_embeddings)
                
                return {
                    'integrated_embeddings': integrated_embeddings,
                    'graph_embeddings': graph_embeddings,
                    'gnn_outputs': gnn_outputs,
                    'graph_data': graph_data_list
                }
            except Exception as e:
                print(f"Warning: Graph processing failed: {e}")
                # Fallback to original embeddings
                return {
                    'integrated_embeddings': text_embeddings,
                    'graph_embeddings': None,
                    'gnn_outputs': None,
                    'graph_data': None
                }
        else:
            # If no graphs, return original embeddings
            return {
                'integrated_embeddings': text_embeddings,
                'graph_embeddings': None,
                'gnn_outputs': None,
                'graph_data': None
            }


class GraphReasoningEthicsModel(nn.Module):
    """
    Enhanced ethics model with advanced graph-based reasoning.
    
    This model incorporates spaCy-based ethical relation extraction
    and PyTorch Geometric graph neural networks for deeper ethical analysis.
    
    Args:
        base_model: Base ethics model
        d_model: Embedding dimension
        gnn_hidden_dim: Hidden dimension for GNN
        gnn_output_dim: Output dimension for GNN
        gnn_num_layers: Number of GNN layers
        gnn_conv_type: Type of GNN convolution
        spacy_model: spaCy model name
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        d_model: int = 512,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 32,
        gnn_num_layers: int = 3,
        gnn_conv_type: str = "gat",
        spacy_model: str = "en_core_web_sm"
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
            spacy_model=spacy_model
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
        
        # Early transformer layers (if available)
        if hasattr(self.base_model, 'transformer_layers'):
            for i, layer in enumerate(self.base_model.transformer_layers):
                # Process first half of transformer layers
                if i < len(self.base_model.transformer_layers) // 2:
                    embeddings = layer(embeddings, src_key_padding_mask=attention_mask)
        
        # Process with graph reasoning if texts provided
        if texts is not None and len(texts) > 0:
            reasoning_outputs = self.relation_reasoning(texts, embeddings)
            enhanced_embeddings = reasoning_outputs['integrated_embeddings']
            
            # Process remaining transformer layers (if available)
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
    
    This class creates visualizations of ethical graphs using NetworkX
    to explain the reasoning process.
    """
    
    @staticmethod
    def visualize_ethical_graph(
        graph: nx.DiGraph,
        title: str = "Ethical Relationship Graph",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a visualization of ethical relationships using matplotlib.
        
        Args:
            graph: NetworkX graph to visualize
            title: Plot title
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("Matplotlib not available for visualization. Consider using Plotly instead.")
            return
        
        if graph.number_of_nodes() == 0:
            print("Empty graph - nothing to visualize")
            return
        
        # Create layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Color mapping for node types
        color_map = {
            "actors": "lightblue",
            "actions": "lightcoral", 
            "values": "lightgreen",
            "emotions": "lightyellow",
            "locations": "lightpink",
            "organizations": "lightgray"
        }
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Draw nodes by type
        for node_type, color in color_map.items():
            nodes_of_type = [n for n, attr in graph.nodes(data=True) 
                             if attr.get('type') == node_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(graph, pos, nodelist=nodes_of_type, 
                                       node_color=color, node_size=500, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.5, arrows=True)
        
        # Draw labels
        labels = {n: attr.get('text', str(n)) for n, attr in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels, font_size=8)
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=node_type.title()) 
                           for node_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def analyze_graph_metrics(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze graph structure and return metrics.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary of graph metrics
        """
        if graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        metrics = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph),
            "n_components": nx.number_weakly_connected_components(graph)
        }
        
        # Node type distribution
        type_counts = {}
        for _, attr in graph.nodes(data=True):
            node_type = attr.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        metrics["node_type_distribution"] = type_counts
        
        # Centrality measures (for non-empty graphs)
        try:
            metrics["avg_degree_centrality"] = sum(nx.degree_centrality(graph).values()) / graph.number_of_nodes()
            metrics["avg_betweenness_centrality"] = sum(nx.betweenness_centrality(graph).values()) / graph.number_of_nodes()
        except:
            metrics["avg_degree_centrality"] = 0
            metrics["avg_betweenness_centrality"] = 0
        
        return metrics


# Example usage and helper functions
def create_enhanced_ethics_model(base_model: nn.Module, **kwargs) -> GraphReasoningEthicsModel:
    """
    Factory function to create an enhanced ethics model with graph reasoning.
    
    Args:
        base_model: Base ethics model
        **kwargs: Additional configuration options
        
    Returns:
        Enhanced ethics model with graph reasoning capabilities
    """
    return GraphReasoningEthicsModel(base_model, **kwargs)


def extract_and_visualize(text: str, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract ethical relations from text and optionally visualize them.
    
    Args:
        text: Input text to analyze
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary containing extracted relations and graph metrics
    """
    extractor = EthicalRelationExtractor()
    relations = extractor.extract_relations(text)
    graph = relations["graph"]
    
    # Analyze graph
    metrics = GraphVisualizer.analyze_graph_metrics(graph)
    
    # Visualize if requested
    if save_path:
        GraphVisualizer.visualize_ethical_graph(graph, save_path=save_path)
    
    return {
        "relations": relations,
        "metrics": metrics,
        "graph": graph
    }


if __name__ == "__main__":
    # Example usage
    test_text = """
    John helped Mary when she was in trouble. The government should protect 
    the rights of all citizens. However, some politicians manipulate public 
    opinion for their own benefit, which damages trust in democratic institutions.
    """
    
    print("Extracting ethical relations from text...")
    result = extract_and_visualize(test_text)
    
    print(f"Graph metrics: {result['metrics']}")
    print(f"Found {len(result['relations']['relations'])} relationships")
