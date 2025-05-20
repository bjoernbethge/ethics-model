"""
Semantic Graph Processing with NetworkX and spaCy

This module provides semantic graph creation and analysis using NetworkX
and spaCy for enhanced ethical and narrative reasoning.
"""

import torch
import torch.nn as nn
import networkx as nx
import spacy
from spacy.tokens import Doc, Token, Span
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from torch_geometric.data import Data
import numpy as np


class SemanticGraphConverter:
    """
    Converts NetworkX semantic graphs to formats compatible with PyTorch Geometric.
    """
    
    @staticmethod
    def networkx_to_pyg_data(
        graph: nx.DiGraph, 
        concept_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        default_dim: int = 128
    ) -> Data:
        """
        Converts a NetworkX semantic graph to a PyTorch Geometric Data object.
        
        Args:
            graph: NetworkX directed graph
            concept_embeddings: Optional dictionary mapping concept strings to embeddings
            default_dim: Default embedding dimension if no embeddings provided
            
        Returns:
            PyTorch Geometric Data object
        """
        if graph.number_of_nodes() == 0:
            # Return empty graph with single node
            return Data(
                x=torch.zeros((1, default_dim), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=1
            )
        
        # Extract nodes and create mapping
        nodes = list(graph.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Create edge index
        edge_index = []
        edge_attributes = []
        
        for source, target, edge_data in graph.edges(data=True):
            source_idx = node_map[source]
            target_idx = node_map[target]
            
            edge_index.append([source_idx, target_idx])
            
            # Extract edge attributes
            relation_type = edge_data.get('relation', 'unknown')
            weight = edge_data.get('weight', 1.0)
            edge_attributes.append([weight, hash(relation_type) % 100 / 100.0])  # Simple encoding
        
        # Create node features
        if concept_embeddings is not None:
            node_features = []
            default_embedding = torch.zeros(next(iter(concept_embeddings.values())).size(0) if concept_embeddings else default_dim)
            
            for node in nodes:
                node_data = graph.nodes[node]
                node_text = node_data.get('text', str(node))
                
                if node_text in concept_embeddings:
                    node_features.append(concept_embeddings[node_text])
                else:
                    # Create feature from node attributes
                    feature = default_embedding.clone()
                    
                    # Add semantic type information
                    semantic_type = node_data.get('semantic_type', 'unknown')
                    if semantic_type == 'agent':
                        feature[0] = 1.0
                    elif semantic_type == 'action':
                        feature[1] = 1.0
                    elif semantic_type == 'object':
                        feature[2] = 1.0
                    elif semantic_type == 'concept':
                        feature[3] = 1.0
                    
                    # Add moral valence
                    moral_valence = node_data.get('moral_valence', 0.0)
                    feature[4] = moral_valence
                    
                    node_features.append(feature)
                    
            node_features = torch.stack(node_features)
        else:
            # Create features from node attributes
            node_features = []
            
            for node in nodes:
                node_data = graph.nodes[node]
                
                # Base feature vector [agent, action, object, concept, moral_valence, ...]
                feature = torch.zeros(default_dim)
                
                # Semantic type encoding
                semantic_type = node_data.get('semantic_type', 'unknown')
                if semantic_type == 'agent':
                    feature[0] = 1.0
                elif semantic_type == 'action':
                    feature[1] = 1.0
                elif semantic_type == 'object':
                    feature[2] = 1.0
                elif semantic_type == 'concept':
                    feature[3] = 1.0
                
                # Moral attributes
                feature[4] = node_data.get('moral_valence', 0.0)
                feature[5] = node_data.get('emotional_intensity', 0.0)
                feature[6] = node_data.get('certainty', 0.5)
                feature[7] = node_data.get('importance', 0.5)
                
                node_features.append(feature)
            
            node_features = torch.stack(node_features)
        
        # Create edge tensors
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 2), dtype=torch.float)
        
        # Create PyG data object
        data = Data(
            x=node_features,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            num_nodes=len(nodes)
        )
        
        # Add node mapping for reference
        data.node_map = node_map
        data.nodes = nodes
        
        return data


class SemanticPatternExtractor:
    """
    Extracts semantic patterns from spaCy parsed documents.
    """
    
    def __init__(self, nlp_model: spacy.Language):
        self.nlp = nlp_model
        
        # Semantic role patterns
        self.agent_deps = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}
        self.object_deps = {"dobj", "iobj", "pobj", "obj"}
        self.predicate_deps = {"ROOT", "acl", "relcl", "xcomp", "ccomp"}
        
        # Moral and emotional indicators
        self.moral_concepts = {
            "positive": {"help", "aid", "assist", "support", "care", "protect", "save", 
                        "benefit", "improve", "heal", "nurture", "respect", "honor",
                        "fair", "just", "right", "good", "ethical", "moral", "virtuous"},
            "negative": {"harm", "hurt", "damage", "destroy", "kill", "injure", "abuse",
                        "neglect", "betray", "steal", "cheat", "lie", "deceive", "manipulate",
                        "wrong", "bad", "evil", "immoral", "unjust", "unfair"}
        }
        
        self.emotional_concepts = {
            "positive": {"happy", "joy", "love", "hope", "grateful", "proud", "satisfied",
                        "pleased", "excited", "confident", "calm", "peaceful"},
            "negative": {"angry", "sad", "fear", "hate", "disgusted", "ashamed", "guilty",
                        "anxious", "worried", "frustrated", "disappointed", "bitter"}
        }
        
        self.obligation_markers = {"must", "should", "ought", "need", "required", "duty", 
                                  "obligation", "responsibility", "imperative"}
    
    def get_moral_valence(self, text: str) -> float:
        """Calculate moral valence of a text span."""
        text_lower = text.lower()
        
        positive_score = sum(1 for word in self.moral_concepts["positive"] 
                           if word in text_lower)
        negative_score = sum(1 for word in self.moral_concepts["negative"] 
                           if word in text_lower)
        
        if positive_score + negative_score == 0:
            return 0.0
        
        return (positive_score - negative_score) / (positive_score + negative_score)
    
    def get_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity of a text span."""
        text_lower = text.lower()
        
        emotion_count = sum(1 for emotion_list in self.emotional_concepts.values()
                           for emotion in emotion_list if emotion in text_lower)
        
        # Normalize by text length
        words = text_lower.split()
        if len(words) == 0:
            return 0.0
        
        return min(emotion_count / len(words), 1.0)
    
    def extract_semantic_roles(self, doc: Doc) -> List[Dict[str, Any]]:
        """
        Extract semantic roles (agent, action, object) from a document.
        
        Args:
            doc: spaCy document
            
        Returns:
            List of semantic role structures
        """
        roles = []
        
        for sent in doc.sents:
            sent_roles = {
                "agents": [],
                "actions": [],
                "objects": [],
                "concepts": [],
                "relations": []
            }
            
            # Find the main verb (root)
            root_verb = None
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root_verb = token
                    break
            
            if root_verb:
                sent_roles["actions"].append({
                    "text": root_verb.text,
                    "lemma": root_verb.lemma_,
                    "semantic_type": "action",
                    "moral_valence": self.get_moral_valence(root_verb.text),
                    "emotional_intensity": self.get_emotional_intensity(root_verb.text)
                })
                
                # Find agents and objects related to this verb
                for child in root_verb.children:
                    if child.dep_ in self.agent_deps:
                        sent_roles["agents"].append({
                            "text": child.text,
                            "semantic_type": "agent",
                            "entity_type": child.ent_type_ if child.ent_type_ else "UNKNOWN",
                            "moral_valence": self.get_moral_valence(child.text)
                        })
                        
                        # Add relation
                        sent_roles["relations"].append({
                            "source": child.text,
                            "target": root_verb.text,
                            "relation": child.dep_,
                            "relation_type": "agent_action"
                        })
                    
                    elif child.dep_ in self.object_deps:
                        sent_roles["objects"].append({
                            "text": child.text,
                            "semantic_type": "object",
                            "entity_type": child.ent_type_ if child.ent_type_ else "UNKNOWN",
                            "moral_valence": self.get_moral_valence(child.text)
                        })
                        
                        # Add relation
                        sent_roles["relations"].append({
                            "source": root_verb.text,
                            "target": child.text,
                            "relation": child.dep_,
                            "relation_type": "action_object"
                        })
            
            # Extract other important concepts
            for token in sent:
                if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                    token not in [r["text"] for r in sent_roles["agents"] + 
                                  sent_roles["actions"] + sent_roles["objects"]]):
                    
                    moral_val = self.get_moral_valence(token.text)
                    if abs(moral_val) > 0.1:  # Only include morally relevant concepts
                        sent_roles["concepts"].append({
                            "text": token.text,
                            "semantic_type": "concept",
                            "pos": token.pos_,
                            "moral_valence": moral_val,
                            "emotional_intensity": self.get_emotional_intensity(token.text)
                        })
            
            roles.append(sent_roles)
        
        return roles
    
    def extract_ethical_patterns(self, doc: Doc) -> Dict[str, List[Any]]:
        """
        Extract ethical patterns from document.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary of ethical patterns
        """
        patterns = {
            "obligations": [],
            "moral_judgments": [],
            "causal_chains": [],
            "value_conflicts": []
        }
        
        for sent in doc.sents:
            # Find obligation statements
            for token in sent:
                if token.lemma_ in self.obligation_markers:
                    patterns["obligations"].append({
                        "text": sent.text,
                        "obligation_marker": token.text,
                        "span": (token.idx, token.idx + len(token))
                    })
            
            # Find moral judgments
            moral_words = []
            for token in sent:
                if (any(moral in token.text.lower() 
                       for moral_list in self.moral_concepts.values() 
                       for moral in moral_list)):
                    moral_words.append(token)
            
            if moral_words:
                patterns["moral_judgments"].append({
                    "text": sent.text,
                    "moral_words": [token.text for token in moral_words],
                    "valence": sum(self.get_moral_valence(token.text) for token in moral_words) / len(moral_words)
                })
            
            # Simple causal chain detection (because/therefore/so patterns)
            causal_markers = {"because", "therefore", "so", "thus", "hence", "consequently"}
            for token in sent:
                if token.lemma_ in causal_markers:
                    patterns["causal_chains"].append({
                        "text": sent.text,
                        "causal_marker": token.text,
                        "position": token.i
                    })
        
        return patterns


class SemanticGraphProcessor(nn.Module):
    """
    Neural module that processes text using spaCy to create and analyze
    semantic graphs for ethical reasoning.
    """
    
    def __init__(self, 
                 d_model: int,
                 spacy_model: str = "en_core_web_sm",
                 semantic_dim: int = 128):
        super().__init__()
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Warning: Could not load spaCy model '{spacy_model}'. "
                  f"Please install with: python -m spacy download {spacy_model}")
            self.nlp = spacy.blank("en")
        # Sentencizer hinzufügen, falls nicht vorhanden
        if "sentencizer" not in self.nlp.pipe_names:
            try:
                self.nlp.add_pipe("sentencizer")
            except Exception as e:
                print(f"Warning: Could not add sentencizer: {e}")
        
        # Initialize pattern extractor
        self.pattern_extractor = SemanticPatternExtractor(self.nlp)
        
        # Converters
        self.converter = SemanticGraphConverter()
        
        # Neural components
        self.semantic_encoder = nn.Sequential(
            nn.Linear(d_model, semantic_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(semantic_dim * 2, semantic_dim)
        )
        
        self.graph_fusion = nn.Sequential(
            nn.Linear(semantic_dim + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Graph attention for combining semantic information
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def create_semantic_graph(self, text: str) -> nx.DiGraph:
        """
        Create a semantic graph from text using spaCy analysis.
        
        Args:
            text: Input text
            
        Returns:
            NetworkX directed graph representing semantic structure
        """
        doc = self.nlp(text)
        graph = nx.DiGraph()
        
        # Extract semantic roles
        semantic_roles = self.pattern_extractor.extract_semantic_roles(doc)
        
        node_id = 0
        text_to_id = {}
        
        # Add nodes from semantic roles
        for sent_roles in semantic_roles:
            for role_type in ["agents", "actions", "objects", "concepts"]:
                for entity in sent_roles[role_type]:
                    entity_text = entity["text"]
                    
                    if entity_text not in text_to_id:
                        text_to_id[entity_text] = node_id
                        
                        graph.add_node(node_id,
                                     text=entity_text,
                                     semantic_type=entity["semantic_type"],
                                     moral_valence=entity.get("moral_valence", 0.0),
                                     emotional_intensity=entity.get("emotional_intensity", 0.0),
                                     entity_type=entity.get("entity_type", "UNKNOWN"),
                                     pos=entity.get("pos", "UNKNOWN"),
                                     certainty=0.8,  # Default certainty
                                     importance=0.5)  # Default importance
                        node_id += 1
            
            # Add edges from relations
            for relation in sent_roles["relations"]:
                source_text = relation["source"]
                target_text = relation["target"]
                
                if source_text in text_to_id and target_text in text_to_id:
                    source_id = text_to_id[source_text]
                    target_id = text_to_id[target_text]
                    
                    graph.add_edge(source_id, target_id,
                                 relation=relation["relation"],
                                 relation_type=relation["relation_type"],
                                 weight=0.8)
        
        # Add co-occurrence edges for entities in the same sentence
        for sent_roles in semantic_roles:
            all_entities = []
            for role_type in ["agents", "actions", "objects", "concepts"]:
                all_entities.extend([entity["text"] for entity in sent_roles[role_type]])
            
            # Connect entities that appear together
            for i, entity1 in enumerate(all_entities):
                for entity2 in all_entities[i+1:]:
                    if entity1 in text_to_id and entity2 in text_to_id:
                        id1, id2 = text_to_id[entity1], text_to_id[entity2]
                        
                        if not graph.has_edge(id1, id2):
                            graph.add_edge(id1, id2,
                                         relation="co_occurrence",
                                         relation_type="contextual",
                                         weight=0.3)
        
        return graph
    
    def extract_ethical_relations(self, graph: nx.DiGraph, text: str) -> Dict[str, List[Any]]:
        """
        Extract ethically relevant relations from semantic graph.
        
        Args:
            graph: NetworkX semantic graph
            text: Original text
            
        Returns:
            Dictionary of ethical relations
        """
        doc = self.nlp(text)
        ethical_patterns = self.pattern_extractor.extract_ethical_patterns(doc)
        
        ethical_relations = {
            'moral_agents': [],
            'moral_actions': [],
            'moral_objects': [],
            'obligations': [],
            'values': [],
            'consequences': [],
            'conflicts': []
        }
        
        # Extract from graph nodes
        for node_id, node_data in graph.nodes(data=True):
            semantic_type = node_data.get('semantic_type', 'unknown')
            moral_valence = node_data.get('moral_valence', 0.0)
            
            if abs(moral_valence) > 0.1:  # Morally significant
                if semantic_type == 'agent':
                    ethical_relations['moral_agents'].append({
                        'id': node_id,
                        'text': node_data['text'],
                        'valence': moral_valence,
                        'entity_type': node_data.get('entity_type', 'UNKNOWN')
                    })
                elif semantic_type == 'action':
                    ethical_relations['moral_actions'].append({
                        'id': node_id,
                        'text': node_data['text'],
                        'valence': moral_valence,
                        'intensity': node_data.get('emotional_intensity', 0.0)
                    })
                elif semantic_type in ['object', 'concept']:
                    ethical_relations['moral_objects'].append({
                        'id': node_id,
                        'text': node_data['text'],
                        'valence': moral_valence,
                        'type': semantic_type
                    })
        
        # Add extracted patterns
        ethical_relations['obligations'] = ethical_patterns['obligations']
        ethical_relations['conflicts'] = ethical_patterns.get('value_conflicts', [])
        
        return ethical_relations
    
    def forward(self, 
                text_batch: List[str], 
                embeddings: torch.Tensor,
                return_graphs: bool = False) -> Dict[str, Any]:
        """
        Process a batch of texts using semantic graph analysis.
        
        Args:
            text_batch: Batch of text strings
            embeddings: Text embeddings (batch_size, seq_len, d_model)
            return_graphs: Whether to return the raw graphs
            
        Returns:
            Dictionary containing:
                - graph_embeddings: Graph-enhanced embeddings
                - ethical_relations: Extracted ethical relations
                - graphs: Raw NetworkX graphs (if return_graphs=True)
        """
        batch_size = len(text_batch)
        
        # Process each text in batch
        graphs = []
        ethical_relations_batch = []
        graph_data_batch = []
        
        for i, text in enumerate(text_batch):
            # Create semantic graph
            graph = self.create_semantic_graph(text)
            graphs.append(graph)
            
            # Extract ethical relations
            relations = self.extract_ethical_relations(graph, text)
            ethical_relations_batch.append(relations)
            
            # Convert to PyG format
            # Use mean pooled embeddings for concept embeddings
            mean_embedding = embeddings[i].mean(dim=0)
            
            # Create concept embeddings dictionary
            concept_embeddings = {}
            for node_id, node_data in graph.nodes(data=True):
                concept_embeddings[node_data['text']] = mean_embedding
            
            graph_data = self.converter.networkx_to_pyg_data(
                graph,
                concept_embeddings=concept_embeddings,
                default_dim=embeddings.size(-1)
            )
            graph_data_batch.append(graph_data)
        
        # Encode semantic information
        # Pool embeddings for graph-level representation
        pooled_embeddings = embeddings.mean(dim=1)  # (batch_size, d_model)
        semantic_features = self.semantic_encoder(pooled_embeddings)  # (batch_size, semantic_dim)
        
        # Attention over semantic features
        attended_features, _ = self.graph_attention(
            semantic_features.unsqueeze(1),  # Add sequence dimension
            semantic_features.unsqueeze(1),
            semantic_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Fuse with original embeddings
        combined_features = torch.cat([attended_features, pooled_embeddings], dim=-1)
        enhanced_embeddings = self.graph_fusion(combined_features)
        
        # Expand back to sequence length
        enhanced_embeddings = enhanced_embeddings.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        
        result = {
            'graph_embeddings': enhanced_embeddings,
            'ethical_relations': ethical_relations_batch,
            'semantic_features': semantic_features
        }
        
        if return_graphs:
            result['graphs'] = graphs
            result['graph_data'] = graph_data_batch
            
        return result
    
    def get_graph_summary(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Generate summary statistics for a semantic graph.
        
        Args:
            graph: NetworkX semantic graph
            
        Returns:
            Dictionary of graph metrics
        """
        if graph.number_of_nodes() == 0:
            return {"empty": True}
        
        # Basic graph metrics
        summary = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph)
        }
        
        # Semantic type distribution
        type_counts = {}
        moral_valences = []
        
        for _, node_data in graph.nodes(data=True):
            semantic_type = node_data.get('semantic_type', 'unknown')
            type_counts[semantic_type] = type_counts.get(semantic_type, 0) + 1
            
            moral_valence = node_data.get('moral_valence', 0.0)
            if abs(moral_valence) > 0.01:
                moral_valences.append(moral_valence)
        
        summary["semantic_types"] = type_counts
        
        if moral_valences:
            summary["avg_moral_valence"] = np.mean(moral_valences)
            summary["moral_std"] = np.std(moral_valences)
            summary["moral_range"] = (min(moral_valences), max(moral_valences))
        else:
            summary["avg_moral_valence"] = 0.0
            summary["moral_std"] = 0.0
            summary["moral_range"] = (0.0, 0.0)
        
        # Centrality measures (if graph is not empty)
        try:
            centralities = nx.degree_centrality(graph)
            summary["max_centrality"] = max(centralities.values()) if centralities else 0.0
            summary["avg_centrality"] = np.mean(list(centralities.values())) if centralities else 0.0
        except:
            summary["max_centrality"] = 0.0
            summary["avg_centrality"] = 0.0
        
        return summary


# Helper functions and utilities
def create_semantic_processor(d_model: int, **kwargs) -> SemanticGraphProcessor:
    """
    Factory function to create a semantic graph processor.
    
    Args:
        d_model: Model dimension
        **kwargs: Additional configuration options
        
    Returns:
        Configured SemanticGraphProcessor
    """
    return SemanticGraphProcessor(d_model=d_model, **kwargs)


def analyze_text_semantics(text: str, processor: SemanticGraphProcessor) -> Dict[str, Any]:
    """
    Analyze semantic structure of a single text.
    
    Args:
        text: Input text
        processor: Configured semantic processor
        
    Returns:
        Dictionary containing semantic analysis
    """
    # Create dummy embeddings for single text analysis
    dummy_embeddings = torch.randn(1, 10, processor.semantic_encoder[0].in_features)
    
    result = processor([text], dummy_embeddings, return_graphs=True)
    
    graph = result['graphs'][0]
    summary = processor.get_graph_summary(graph)
    
    return {
        "semantic_graph": graph,
        "ethical_relations": result['ethical_relations'][0],
        "graph_summary": summary,
        "graph_data": result['graph_data'][0]
    }


if __name__ == "__main__":
    # Example usage
    processor = create_semantic_processor(d_model=128)
    
    test_text = """
    The doctor helped the patient recover from their illness. 
    This action demonstrates compassion and professional duty. 
    However, the treatment was expensive and not everyone can afford it.
    """
    
    print("Analyzing semantic structure...")
    analysis = analyze_text_semantics(test_text, processor)
    
    print(f"Graph summary: {analysis['graph_summary']}")
    print(f"Ethical relations found: {len(analysis['ethical_relations']['moral_actions'])} moral actions")
    print(f"Graph has {analysis['semantic_graph'].number_of_nodes()} nodes and {analysis['semantic_graph'].number_of_edges()} edges")
