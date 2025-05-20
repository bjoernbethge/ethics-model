"""
Explainability Module for Ethics Model

This module provides tools for explaining the ethics model's decisions,
including attention visualization, important features extraction, and
natural language explanations generation using NetworkX and spaCy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from transformers import PreTrainedTokenizer
import networkx as nx
import spacy
from spacy.tokens import Doc, Token, Span


class AttentionVisualizer:
    """
    Visualizes attention weights from the ethics model for explainability.
    
    Args:
        tokenizer: HuggingFace tokenizer for text processing
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def visualize_attention(
        self, 
        text: str, 
        attention_weights: torch.Tensor,
        token_contributions: Optional[torch.Tensor] = None,
        title: str = "Ethics Model Attention Weights"
    ) -> go.Figure:
        """
        Create a heatmap visualization of attention weights.
        
        Args:
            text: Input text
            attention_weights: Attention weights from model (batch_size, seq_len, seq_len)
            token_contributions: Optional tensor of token contributions to the final decision
            title: Plot title
            
        Returns:
            Plotly figure with attention heatmap
        """
        # Tokenize text to get tokens
        tokens = self.tokenizer.tokenize(text)
        tokens = [t.replace('##', '') for t in tokens]  # Clean BERT tokens
        
        # Reduce attention weights to relevant sequence length
        seq_len = min(len(tokens), attention_weights.shape[1])
        tokens = tokens[:seq_len]
        attn = attention_weights[0, :seq_len, :seq_len].cpu().detach().numpy()
        
        # Create heatmap
        fig = px.imshow(
            attn,
            x=tokens,
            y=tokens,
            labels=dict(x="Target Tokens", y="Source Tokens", color="Attention Weight"),
            title=title,
            color_continuous_scale="RdBu_r"
        )
        
        # Add token contributions if provided
        if token_contributions is not None:
            token_importance = token_contributions[0, :seq_len].cpu().detach().numpy()
            # Normalize to [0, 1]
            if token_importance.max() > token_importance.min():
                token_importance = (token_importance - token_importance.min()) / (token_importance.max() - token_importance.min())
            else:
                token_importance = np.zeros_like(token_importance)
            
            # Add a bar chart at the top showing token importance
            bar_trace = go.Bar(
                x=tokens,
                y=token_importance,
                marker=dict(color=token_importance, colorscale="Viridis"),
                name="Token Importance",
                yaxis="y2"
            )
            
            fig.add_trace(bar_trace)
            
            # Update layout for dual y-axes
            fig.update_layout(
                yaxis2=dict(
                    title="Token Importance",
                    overlaying="y",
                    side="right",
                    range=[0, 1]
                )
            )
        
        # Improve layout
        fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def visualize_token_attributions(
        self,
        text: str,
        attributions: torch.Tensor,
        ethics_score: float,
        manipulation_score: float,
        title: str = "Token Attributions for Ethical Analysis"
    ) -> go.Figure:
        """
        Create a visualization of token-level attributions.
        
        Args:
            text: Input text
            attributions: Token attributions (batch_size, seq_len)
            ethics_score: Ethics score from model
            manipulation_score: Manipulation score from model
            title: Plot title
            
        Returns:
            Plotly figure with token attributions
        """
        # Tokenize text
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        
        # Get attributions for actual tokens (not special tokens)
        attrib = attributions[0, :len(tokens)].cpu().detach().numpy()
        
        # Create figure
        fig = px.bar(
            x=tokens,
            y=attrib,
            title=title,
            labels=dict(x="Tokens", y="Attribution Score"),
            color=attrib,
            color_continuous_scale="RdBu"
        )
        
        # Add ethics and manipulation scores as annotations
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Ethics Score: {ethics_score:.2f}",
            showarrow=False,
            font=dict(size=14, color="green" if ethics_score > 0.5 else "red"),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.add_annotation(
            x=0.02,
            y=0.93,
            xref="paper",
            yref="paper",
            text=f"Manipulation Score: {manipulation_score:.2f}",
            showarrow=False,
            font=dict(size=14, color="green" if manipulation_score < 0.5 else "red"),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        # Improve layout
        fig.update_layout(
            width=900,
            height=500,
            xaxis=dict(tickangle=45),
            margin=dict(t=100, b=100)
        )
        
        return fig
    
    def visualize_attention_head_view(
        self,
        text: str,
        attention_weights: torch.Tensor,
        layer_idx: int = 0,
        title: str = "Multi-Head Attention View"
    ) -> go.Figure:
        """
        Create a visualization of multi-head attention patterns.
        
        Args:
            text: Input text
            attention_weights: Attention weights from model (batch_size, n_heads, seq_len, seq_len)
            layer_idx: Layer index to visualize
            title: Plot title
            
        Returns:
            Plotly figure with multi-head attention visualization
        """
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        tokens = [t.replace('##', '') for t in tokens]  # Clean BERT tokens
        
        # Check if we have multi-head attention
        if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
            n_heads = attention_weights.shape[1]
            seq_len = min(len(tokens), attention_weights.shape[2])
            tokens = tokens[:seq_len]
            
            # Create subplots for each attention head
            from plotly.subplots import make_subplots
            
            cols = min(4, n_heads)
            rows = (n_heads + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"Head {i+1}" for i in range(n_heads)],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for head in range(n_heads):
                row = (head // cols) + 1
                col = (head % cols) + 1
                
                attn = attention_weights[0, head, :seq_len, :seq_len].cpu().detach().numpy()
                
                heatmap = go.Heatmap(
                    z=attn,
                    x=tokens,
                    y=tokens,
                    colorscale="RdBu_r",
                    showscale=(head == 0),  # Only show colorbar for first head
                    hovertemplate="From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>"
                )
                
                fig.add_trace(heatmap, row=row, col=col)
            
            fig.update_layout(
                title=title,
                height=200 * rows,
                width=800
            )
            
            # Update all y-axes to reverse
            for i in range(1, n_heads + 1):
                fig.update_yaxes(autorange="reversed", row=(i-1)//cols + 1, col=(i-1)%cols + 1)
            
        else:
            # Fall back to single attention visualization
            fig = self.visualize_attention(text, attention_weights, title=title)
        
        return fig


class GraphExplainer:
    """
    Explains ethical decisions using NetworkX graphs built from spaCy analysis.
    
    Args:
        spacy_model: spaCy model name for NLP processing
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Warning: Could not load spaCy model '{spacy_model}'. "
                  f"Please install with: python -m spacy download {spacy_model}")
            # Create a blank model as fallback
            self.nlp = spacy.blank("en")
        
        # Ethical concept dictionaries
        self.moral_concepts = {
            "values": ["fairness", "justice", "equality", "freedom", "liberty", "care", 
                       "compassion", "loyalty", "authority", "purity", "sanctity"],
            "positive_actions": ["help", "assist", "support", "protect", "benefit", 
                                 "contribute", "share", "cooperate", "nurture", "heal"],
            "negative_actions": ["harm", "hurt", "damage", "exploit", "manipulate", 
                                 "deceive", "cheat", "steal", "abuse", "oppress"],
            "stakeholders": ["people", "person", "individual", "citizen", "community", 
                             "society", "public", "victim", "beneficiary"]
        }
    
    def _categorize_entity(self, text: str, pos: str = None, ent_type: str = None) -> Tuple[str, float]:
        """
        Categorize an entity by its ethical relevance.
        
        Args:
            text: Entity text
            pos: Part of speech
            ent_type: Named entity type
            
        Returns:
            Tuple of (category, relevance_score)
        """
        text_lower = text.lower()
        
        # Check for moral concepts
        for category, concepts in self.moral_concepts.items():
            if any(concept in text_lower for concept in concepts):
                if category == "values":
                    return "moral_value", 0.9
                elif category == "positive_actions":
                    return "positive_action", 0.8
                elif category == "negative_actions":
                    return "negative_action", 0.8
                elif category == "stakeholders":
                    return "stakeholder", 0.7
        
        # Check named entity types
        if ent_type in ["PERSON", "ORG", "GPE", "NORP"]:
            return "entity", 0.6
        
        # Check part of speech
        if pos in ["VERB", "ADJ"]:
            return "descriptor", 0.4
        
        return "other", 0.1
    
    def build_ethical_graph(self, text: str) -> nx.DiGraph:
        """
        Build a directed graph representing ethical relationships in the text.
        
        Args:
            text: Input text
            
        Returns:
            NetworkX directed graph
        """
        doc = self.nlp(text)
        graph = nx.DiGraph()
        
        # Track entities and their properties
        entities = {}
        entity_counter = 0
        
        # Add named entities
        for ent in doc.ents:
            if ent.text not in entities:
                category, relevance = self._categorize_entity(ent.text, ent.label_, ent.label_)
                entities[ent.text] = {
                    'id': entity_counter,
                    'category': category,
                    'relevance': relevance,
                    'type': 'named_entity',
                    'ent_type': ent.label_
                }
                entity_counter += 1
        
        # Add significant tokens (verbs, adjectives, nouns)
        for token in doc:
            if (token.pos_ in ["VERB", "ADJ", "NOUN"] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2 and
                token.text not in entities):
                
                category, relevance = self._categorize_entity(token.text, token.pos_)
                entities[token.text] = {
                    'id': entity_counter,
                    'category': category,
                    'relevance': relevance,
                    'type': 'concept',
                    'pos': token.pos_
                }
                entity_counter += 1
        
        # Add nodes to graph
        for entity_text, entity_data in entities.items():
            graph.add_node(
                entity_data['id'],
                text=entity_text,
                category=entity_data['category'],
                relevance=entity_data['relevance'],
                type=entity_data['type']
            )
        
        # Add edges based on dependency relationships
        for token in doc:
            if token.text in entities and token.head.text in entities:
                source_id = entities[token.head.text]['id']
                target_id = entities[token.text]['id']
                
                # Calculate edge weight based on dependency relation and moral relevance
                relation_weights = {
                    'nsubj': 0.9, 'nsubjpass': 0.9, 'dobj': 0.8, 'iobj': 0.7,
                    'amod': 0.6, 'compound': 0.5, 'prep': 0.4
                }
                
                edge_weight = relation_weights.get(token.dep_, 0.3)
                
                # Boost weight if both entities are morally relevant
                source_relevance = entities[token.head.text]['relevance']
                target_relevance = entities[token.text]['relevance']
                
                if source_relevance > 0.5 and target_relevance > 0.5:
                    edge_weight *= 1.5
                
                graph.add_edge(
                    source_id, 
                    target_id,
                    relation=token.dep_,
                    weight=min(edge_weight, 1.0)
                )
        
        # Add co-occurrence edges for entities in the same sentence
        for sent in doc.sents:
            sent_entities = []
            for token in sent:
                if token.text in entities:
                    sent_entities.append(entities[token.text]['id'])
            
            # Connect entities that co-occur
            for i, entity1 in enumerate(sent_entities):
                for entity2 in sent_entities[i+1:]:
                    if not graph.has_edge(entity1, entity2):
                        graph.add_edge(entity1, entity2, 
                                       relation="co_occurrence", 
                                       weight=0.3)
        
        return graph
    
    def extract_ethical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract ethically relevant entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of ethical entity types and their instances
        """
        doc = self.nlp(text)
        
        entities = {
            "actors": [],
            "actions": [],
            "values": [],
            "consequences": [],
            "stakeholders": []
        }
        
        # Process named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "NORP"]:
                entities["actors"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["stakeholders"].append(ent.text)
        
        # Process tokens for moral concepts
        for token in doc:
            if not token.is_stop and not token.is_punct:
                text_lower = token.text.lower()
                
                # Check for moral values
                if any(val in text_lower for val in self.moral_concepts["values"]):
                    entities["values"].append(token.text)
                
                # Check for actions
                elif (token.pos_ == "VERB" and 
                      any(act in text_lower for act in 
                          self.moral_concepts["positive_actions"] + 
                          self.moral_concepts["negative_actions"])):
                    entities["actions"].append(token.text)
                
                # Check for stakeholders
                elif any(stake in text_lower for stake in self.moral_concepts["stakeholders"]):
                    entities["stakeholders"].append(token.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def visualize_ethical_graph(self, text: str, title: str = "Ethical Relationship Graph") -> go.Figure:
        """
        Create a Plotly visualization of the ethical relationships in the text.
        
        Args:
            text: Input text
            title: Plot title
            
        Returns:
            Plotly figure with network visualization
        """
        graph = self.build_ethical_graph(text)
        
        if graph.number_of_nodes() == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="No ethical relationships detected",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=title)
            return fig
        
        # Create layout using spring layout
        try:
            pos = nx.spring_layout(graph, k=1, iterations=50)
        except:
            # Fallback to random layout
            pos = {node: (np.random.random(), np.random.random()) 
                   for node in graph.nodes()}
        
        # Color mapping for node categories
        color_map = {
            "moral_value": "green",
            "positive_action": "blue", 
            "negative_action": "red",
            "stakeholder": "orange",
            "entity": "purple",
            "descriptor": "gray",
            "other": "lightgray"
        }
        
        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_labels = []
        node_hover = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            x, y = pos[node]
            
            node_x.append(x)
            node_y.append(y)
            node_colors.append(color_map.get(node_data.get('category', 'other'), 'lightgray'))
            
            # Size based on relevance
            relevance = node_data.get('relevance', 0.1)
            node_sizes.append(max(10, relevance * 30))
            
            # Labels
            text = node_data.get('text', str(node))
            node_labels.append(text)
            node_hover.append(f"{text}<br>Category: {node_data.get('category', 'unknown')}<br>"
                               f"Relevance: {relevance:.2f}")
        
        # Prepare edge data
        edge_traces = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_data = graph.edges[edge]
            edge_weight = edge_data.get('weight', 0.3)
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=max(1, edge_weight * 3), color='gray'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=node_hover,
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=title,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=800
            )
        )
        
        # Add legend for node categories
        for category, color in color_map.items():
            if any(graph.nodes[node].get('category') == category for node in graph.nodes()):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    showlegend=True,
                    name=category.replace('_', ' ').title()
                ))
        
        return fig
    
    def analyze_ethical_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze ethical patterns in the text using graph metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of ethical pattern analysis
        """
        graph = self.build_ethical_graph(text)
        
        if graph.number_of_nodes() == 0:
            return {"error": "No ethical entities detected"}
        
        # Calculate graph metrics
        metrics = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph)
        }
        
        # Analyze node categories
        category_counts = {}
        total_relevance_by_category = {}
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            category = node_data.get('category', 'other')
            relevance = node_data.get('relevance', 0.1)
            
            category_counts[category] = category_counts.get(category, 0) + 1
            total_relevance_by_category[category] = total_relevance_by_category.get(category, 0) + relevance
        
        metrics["category_distribution"] = category_counts
        metrics["category_relevance"] = total_relevance_by_category
        
        # Calculate centrality measures for moral entities
        moral_centrality = {}
        try:
            centrality = nx.degree_centrality(graph)
            for node in graph.nodes():
                node_data = graph.nodes[node]
                if node_data.get('relevance', 0) > 0.5:  # Only for morally relevant nodes
                    moral_centrality[node_data.get('text', str(node))] = centrality[node]
        except:
            moral_centrality = {}
        
        metrics["moral_entity_centrality"] = moral_centrality
        
        # Identify potential ethical conflicts (connections between positive and negative concepts)
        conflicts = []
        for edge in graph.edges():
            source_data = graph.nodes[edge[0]]
            target_data = graph.nodes[edge[1]]
            
            if (source_data.get('category') == 'positive_action' and 
                target_data.get('category') == 'negative_action') or \
               (source_data.get('category') == 'negative_action' and 
                target_data.get('category') == 'positive_action'):
                conflicts.append({
                    'source': source_data.get('text', ''),
                    'target': target_data.get('text', ''),
                    'relation': graph.edges[edge].get('relation', '')
                })
        
        metrics["potential_conflicts"] = conflicts
        
        return metrics


class EthicsExplainer:
    """
    Comprehensive explainer for the ethics model.
    
    This class combines different explainability techniques to provide
    a thorough understanding of the model's ethical decisions using
    NetworkX and spaCy instead of GraphBrain.
    
    Args:
        model: Ethics model
        tokenizer: Tokenizer for text processing
        device: Computation device
        spacy_model: spaCy model name for NLP processing
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device = torch.device("cpu"),
        spacy_model: str = "en_core_web_sm"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize sub-explainers
        self.attention_viz = AttentionVisualizer(tokenizer)
        self.graph_explainer = GraphExplainer(spacy_model)
        
        # Ensure model is in eval mode for explanation
        self.model.eval()
    
    def compute_attributions(
        self, 
        text: str,
        llm: nn.Module,
        target: str = "ethics"
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Compute token attributions using integrated gradients.
        
        Args:
            text: Input text
            llm: Language model for embeddings
            target: Target output to explain ("ethics" or "manipulation")
            
        Returns:
            Tuple of model outputs and token attributions
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Function to get embeddings
        def get_embeddings(input_ids):
            with torch.no_grad():
                if hasattr(llm, 'model') and hasattr(llm.model, 'transformer'):
                    llm_outputs = llm.model.transformer(input_ids)
                elif hasattr(llm, 'transformer'):
                    llm_outputs = llm.transformer(input_ids)
                else:
                    # Fallback - assume llm is the transformer directly
                    llm_outputs = llm(input_ids)
                
                if hasattr(llm_outputs, 'last_hidden_state'):
                    return llm_outputs.last_hidden_state
                else:
                    return llm_outputs[0] if isinstance(llm_outputs, tuple) else llm_outputs
        
        # Get base embeddings
        base_embeddings = get_embeddings(inputs["input_ids"])
        
        # Compute outputs with gradients
        embeddings = base_embeddings.clone().detach().requires_grad_(True)
        
        outputs = self.model(
            embeddings=embeddings,
            attention_mask=inputs["attention_mask"],
            texts=[text]
        )
        
        # Select target score
        if target == "ethics":
            score = outputs["ethics_score"]
        else:
            score = outputs["manipulation_score"]
        
        # Compute gradients
        if score.requires_grad:
            score.backward(torch.ones_like(score))
            
            # Get gradients
            grads = embeddings.grad
            
            # Simple attribution: gradient * input
            attributions = grads * base_embeddings
            
            # Sum across embedding dimension
            token_attributions = attributions.sum(dim=2)
        else:
            # Fallback if gradients not available
            token_attributions = torch.zeros(base_embeddings.shape[:2])
        
        return outputs, token_attributions
    
    def explain(
        self, 
        text: str,
        llm: nn.Module
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for model's prediction on input text.
        
        Args:
            text: Input text
            llm: Language model for embeddings
            
        Returns:
            Dictionary with explanation components
        """
        try:
            # Get model prediction and attributions
            ethics_outputs, ethics_attributions = self.compute_attributions(text, llm, target="ethics")
            _, manipulation_attributions = self.compute_attributions(text, llm, target="manipulation")
            
            # Extract outputs
            ethics_score = ethics_outputs["ethics_score"].item()
            manipulation_score = ethics_outputs["manipulation_score"].item()
            attention_weights = ethics_outputs.get("attention_weights", torch.zeros(1, 1, 1))
            
            # Generate attention visualization
            attention_viz = self.attention_viz.visualize_attention(
                text, 
                attention_weights, 
                ethics_attributions
            )
            
            # Generate token attribution visualization
            token_viz = self.attention_viz.visualize_token_attributions(
                text,
                ethics_attributions,
                ethics_score,
                manipulation_score
            )
            
            # Generate graph visualization
            graph_viz = self.graph_explainer.visualize_ethical_graph(text)
            
            # Extract ethical entities
            ethical_entities = self.graph_explainer.extract_ethical_entities(text)
            
            # Analyze ethical patterns
            ethical_patterns = self.graph_explainer.analyze_ethical_patterns(text)
            
            # Get framework analysis if available
            framework_analysis = {}
            if 'framework_analysis' in ethics_outputs:
                framework_analysis = {
                    'dominant_framework': self._get_dominant_framework(ethics_outputs),
                    'framework_scores': self._get_framework_scores(ethics_outputs)
                }
            
            # Create narrative analysis if available
            narrative_analysis = {}
            if 'framing_analysis' in ethics_outputs:
                narrative_analysis = {
                    'framing_strength': ethics_outputs['framing_analysis']['framing_strength'].mean().item(),
                    'cognitive_dissonance': ethics_outputs.get('dissonance_analysis', {}).get('dissonance_score', torch.tensor(0.0)).mean().item(),
                    'manipulation_techniques': self._get_manipulation_techniques(ethics_outputs)
                }
            
            # Combine explanations
            explanation = {
                'text': text,
                'ethics_score': ethics_score,
                'manipulation_score': manipulation_score,
                'attention_visualization': attention_viz,
                'token_attribution_visualization': token_viz,
                'graph_visualization': graph_viz,
                'ethical_entities': ethical_entities,
                'ethical_patterns': ethical_patterns,
                'framework_analysis': framework_analysis,
                'narrative_analysis': narrative_analysis
            }
            
        except Exception as e:
            # Provide graceful error handling
            print(f"Warning: Error in explanation generation: {e}")
            explanation = {
                'text': text,
                'ethics_score': 0.5,
                'manipulation_score': 0.5,
                'error': str(e),
                'ethical_entities': {},
                'ethical_patterns': {},
                'framework_analysis': {},
                'narrative_analysis': {}
            }
        
        return explanation
    
    def _get_dominant_framework(self, outputs: Dict[str, Any]) -> str:
        """Get the dominant moral framework from outputs."""
        if 'framework_analysis' not in outputs:
            return "unknown"
            
        framework_outputs = outputs['framework_analysis'].get('framework_outputs', {})
        if not framework_outputs:
            return "unknown"
            
        framework_scores = {}
        
        for framework, output in framework_outputs.items():
            if hasattr(output, 'mean'):
                framework_scores[framework] = output.mean().item()
            else:
                framework_scores[framework] = float(output)
            
        if framework_scores:
            return max(framework_scores.items(), key=lambda x: x[1])[0]
        return "unknown"
    
    def _get_framework_scores(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """Get scores for each moral framework."""
        if 'framework_analysis' not in outputs:
            return {}
            
        framework_outputs = outputs['framework_analysis'].get('framework_outputs', {})
        framework_scores = {}
        
        for framework, output in framework_outputs.items():
            if hasattr(output, 'mean'):
                framework_scores[framework] = output.mean().item()
            else:
                framework_scores[framework] = float(output)
            
        return framework_scores
    
    def _get_manipulation_techniques(self, outputs: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get detected manipulation techniques with scores."""
        if 'manipulation_analysis' not in outputs:
            return []
            
        technique_scores = outputs['manipulation_analysis'].get('technique_scores', {})
        techniques = []
        
        for technique, score in technique_scores.items():
            if hasattr(score, 'mean'):
                techniques.append((technique, score.mean().item()))
            else:
                techniques.append((technique, float(score)))
            
        # Sort by score (descending)
        techniques.sort(key=lambda x: x[1], reverse=True)
        
        return techniques


# Utility functions for backward compatibility and ease of use
def create_explainer(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer, 
    device: torch.device = torch.device("cpu"),
    **kwargs
) -> EthicsExplainer:
    """
    Factory function to create an EthicsExplainer.
    
    Args:
        model: Ethics model to explain
        tokenizer: Tokenizer for text processing
        device: Computation device
        **kwargs: Additional arguments for explainer initialization
        
    Returns:
        Configured EthicsExplainer instance
    """
    return EthicsExplainer(model, tokenizer, device, **kwargs)


def quick_explain(
    text: str,
    model: nn.Module,
    llm: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Quick explanation function for a single text.
    
    Args:
        text: Text to explain
        model: Ethics model
        llm: Language model for embeddings
        tokenizer: Tokenizer
        device: Computation device
        
    Returns:
        Dictionary with explanation components
    """
    explainer = EthicsExplainer(model, tokenizer, device)
    return explainer.explain(text, llm)


if __name__ == "__main__":
    # Example usage
    print("Ethics Model Explainability Module")
    print("This module provides tools for explaining ethics model decisions")
    print("using NetworkX for graph analysis and spaCy for NLP processing.")
