"""
Explainability Module for Ethics Model

This module provides tools for explaining the ethics model's decisions,
including attention visualization, important features extraction, and
natural language explanations generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from transformers import PreTrainedTokenizer
import graphbrain as gb
from graphbrain import hgraph


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
            token_importance = (token_importance - token_importance.min()) / (token_importance.max() - token_importance.min() + 1e-6)
            
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
        
        # Map colors: positive attribution (ethical) in blue, negative (unethical) in red
        colors = ["red" if a < 0 else "blue" for a in attrib]
        
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


class GraphExplainer:
    """
    Explains ethical decisions using the graph structure from GraphBrain.
    
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
    
    def _get_graph(self, text: str) -> Optional[hgraph]:
        """Get hypergraph from text."""
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
    
    def extract_ethical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract ethically relevant entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of ethical entity types and their instances
        """
        hg = self._get_graph(text)
        if hg is None:
            return {}
            
        entities = {
            "actors": [],
            "actions": [],
            "values": [],
            "consequences": []
        }
        
        # Find actors (subjects of verbs)
        for edge in hg.edges():
            if len(edge) > 1 and gb.is_atom(edge[0]) and edge[0].type() == 'P':
                # First argument is often the subject/actor
                for i in range(1, min(2, len(edge))):
                    if gb.is_atom(edge[i]):
                        actor = str(edge[i])
                        if actor not in entities["actors"]:
                            entities["actors"].append(actor)
                
                # The predicate itself is an action
                action = str(edge[0])
                if action not in entities["actions"]:
                    entities["actions"].append(action)
        
        # Simplistic value detection (keywords)
        value_keywords = ["good", "bad", "right", "wrong", "moral", "ethical", 
                          "unethical", "fair", "unfair", "just", "unjust", 
                          "honest", "dishonest", "harm", "benefit"]
        
        for atom in hg.all_atoms():
            atom_str = str(atom).lower()
            if any(keyword in atom_str for keyword in value_keywords):
                if atom_str not in entities["values"]:
                    entities["values"].append(str(atom))
        
        return entities
    
    def visualize_ethical_graph(self, text: str) -> go.Figure:
        """
        Create a visualization of the ethical relationships in the text.
        
        Args:
            text: Input text
            
        Returns:
            Plotly figure with network visualization
        """
        hg = self._get_graph(text)
        if hg is None:
            return go.Figure()
        
        # Extract nodes and edges
        nodes = []
        edges = []
        node_types = {}
        
        # Actor/action detection heuristics
        action_indicators = ["do", "did", "does", "doing", "done"]
        moral_indicators = ["good", "bad", "right", "wrong", "moral", "ethical", 
                            "fair", "unfair", "just", "benefit", "harm"]
        
        # Process edges to build node and edge lists
        for edge in hg.edges():
            if len(edge) > 1:
                # Add nodes
                for i, item in enumerate(edge):
                    if gb.is_atom(item):
                        item_str = str(item)
                        if item_str not in nodes:
                            nodes.append(item_str)
                            
                            # Determine node type
                            if i == 0:  # First position typically a predicate/action
                                node_types[item_str] = "action"
                            elif i == 1:  # Second position often subject/actor
                                node_types[item_str] = "actor"
                            elif any(ind in item_str.lower() for ind in moral_indicators):
                                node_types[item_str] = "moral"
                            else:
                                node_types[item_str] = "other"
                
                # Add edges
                if gb.is_atom(edge[0]):
                    pred = str(edge[0])
                    for i in range(1, len(edge)):
                        if gb.is_atom(edge[i]):
                            arg = str(edge[i])
                            edges.append((pred, arg))
        
        # Build node trace
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_labels = []
        
        color_map = {
            "actor": "blue",
            "action": "red",
            "moral": "green",
            "other": "gray"
        }
        
        # Create a layout (using a circle for simplicity)
        n = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            node_x.append(np.cos(angle))
            node_y.append(np.sin(angle))
            node_colors.append(color_map[node_types[node]])
            node_sizes.append(15 if node_types[node] in ["actor", "action", "moral"] else 10)
            node_labels.append(node)
        
        # Create the plotly figure
        edge_traces = []
        for edge in edges:
            source_idx = nodes.index(edge[0])
            target_idx = nodes.index(edge[1])
            
            x0, y0 = node_x[source_idx], node_y[source_idx]
            x1, y1 = node_x[target_idx], node_y[target_idx]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
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
            hovertext=node_labels
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="Ethical Relationship Graph",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=800
            )
        )
        
        # Add a legend for node types
        for node_type, color in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                showlegend=True,
                name=node_type.capitalize()
            ))
        
        return fig


class EthicsExplainer:
    """
    Comprehensive explainer for the ethics model.
    
    This class combines different explainability techniques to provide
    a thorough understanding of the model's ethical decisions.
    
    Args:
        model: Ethics model
        tokenizer: Tokenizer for text processing
        device: Computation device
        parser_lang: Language for GraphBrain parser
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device = torch.device("cpu"),
        parser_lang: str = "en"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize sub-explainers
        self.attention_viz = AttentionVisualizer(tokenizer)
        self.graph_explainer = GraphExplainer(parser_lang)
        
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
        
        # Get embedding inputs
        emb_inputs = {}
        for k, v in inputs.items():
            emb_inputs[k] = v
        
        # Function to get embeddings
        def get_embeddings(input_ids):
            with torch.no_grad():
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                return llm_outputs.last_hidden_state
        
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
        score.backward(torch.ones_like(score))
        
        # Get gradients
        grads = embeddings.grad
        
        # Simple attribution: gradient * input
        attributions = grads * base_embeddings
        
        # Sum across embedding dimension
        token_attributions = attributions.sum(dim=2)
        
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
        # Get model prediction and attributions
        ethics_outputs, ethics_attributions = self.compute_attributions(text, llm, target="ethics")
        _, manipulation_attributions = self.compute_attributions(text, llm, target="manipulation")
        
        # Extract outputs
        ethics_score = ethics_outputs["ethics_score"].item()
        manipulation_score = ethics_outputs["manipulation_score"].item()
        attention_weights = ethics_outputs["attention_weights"]
        
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
        
        # Get framework analysis
        framework_analysis = {
            'dominant_framework': self._get_dominant_framework(ethics_outputs),
            'framework_scores': self._get_framework_scores(ethics_outputs)
        }
        
        # Create narrative analysis
        narrative_analysis = {
            'framing_strength': ethics_outputs['framing_analysis']['framing_strength'].mean().item(),
            'cognitive_dissonance': ethics_outputs['dissonance_analysis']['dissonance_score'].mean().item(),
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
            'framework_analysis': framework_analysis,
            'narrative_analysis': narrative_analysis
        }
        
        return explanation
    
    def _get_dominant_framework(self, outputs: Dict[str, Any]) -> str:
        """Get the dominant moral framework from outputs."""
        framework_outputs = outputs['framework_analysis']['framework_outputs']
        framework_scores = {}
        
        for framework, output in framework_outputs.items():
            framework_scores[framework] = output.mean().item()
            
        return max(framework_scores.items(), key=lambda x: x[1])[0]
    
    def _get_framework_scores(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """Get scores for each moral framework."""
        framework_outputs = outputs['framework_analysis']['framework_outputs']
        framework_scores = {}
        
        for framework, output in framework_outputs.items():
            framework_scores[framework] = output.mean().item()
            
        return framework_scores
    
    def _get_manipulation_techniques(self, outputs: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get detected manipulation techniques with scores."""
        technique_scores = outputs['manipulation_analysis']['technique_scores']
        techniques = []
        
        for technique, score in technique_scores.items():
            techniques.append((technique, score.mean().item()))
            
        # Sort by score (descending)
        techniques.sort(key=lambda x: x[1], reverse=True)
        
        return techniques
