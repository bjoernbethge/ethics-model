"""
Explainability Module for Ethics Model

This module provides tools to explain the predictions of the ethics model,
visualize attention patterns, and interpret ethical reasoning.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import graphbrain as gb
from matplotlib.figure import Figure
from IPython.display import HTML, display
import os
import json


class AttentionVisualizer:
    """
    Visualizes attention patterns from the ethics model to explain which parts
    of the text influenced the ethical judgment.
    
    Args:
        model: Ethics model instance
        tokenizer: Tokenizer used by the model
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_layer_names = self._get_attention_layer_names()
    
    def _get_attention_layer_names(self) -> List[str]:
        """Extract names of attention layers in the model."""
        layer_names = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                layer_names.append(name)
        return layer_names
    
    def explain(
        self, 
        text: str, 
        llm: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu'),
        layer_name: Optional[str] = None,
        return_html: bool = False,
        min_alpha: float = 0.3
    ) -> Union[Figure, HTML]:
        """
        Visualize attention weights to explain model prediction.
        
        Args:
            text: Input text to explain
            llm: Optional language model for embeddings
            device: Computation device
            layer_name: Specific attention layer to visualize (None for all)
            return_html: Whether to return HTML for notebook display
            min_alpha: Minimum alpha value for attention heatmap
            
        Returns:
            Matplotlib figure or HTML display object
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        ).to(device)
        
        # Process through LLM if provided
        if llm is not None:
            with torch.no_grad():
                llm_outputs = llm.model.transformer(inputs['input_ids']) if hasattr(llm, 'model') else llm.transformer(inputs['input_ids'])
                hidden_states = llm_outputs.last_hidden_state
                embeddings = hidden_states
        else:
            embeddings = None
        
        # Get model prediction and attention weights
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=None if embeddings is not None else inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                embeddings=embeddings,
                texts=[text]  # For GraphBrain processing
            )
        
        # Get attention weights
        attention_weights = outputs['attention_weights'].cpu().numpy()
        
        # Get token mapping
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Create visualization
        fig = self._create_attention_heatmap(tokens, attention_weights[0], text, min_alpha)
        
        # Get ethics and manipulation scores
        ethics_score = outputs['ethics_score'].item()
        manipulation_score = outputs['manipulation_score'].item()
        
        # Add scores to the figure title
        plt.suptitle(f"Ethics Score: {ethics_score:.3f} | Manipulation Score: {manipulation_score:.3f}")
        
        if return_html:
            # Create HTML version for interactive display
            return self._create_html_visualization(tokens, attention_weights[0], text, ethics_score, manipulation_score)
        
        return fig
    
    def _create_attention_heatmap(
        self, 
        tokens: List[str], 
        attention_weights: np.ndarray,
        original_text: str,
        min_alpha: float = 0.3
    ) -> Figure:
        """Create a heatmap visualization of attention weights."""
        # Normalize weights
        norm_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-6)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Map tokens back to original text
        # This is a simplification - in practice, you'd need more sophisticated mapping
        token_map = self._map_tokens_to_text(tokens, original_text)
        
        # Merge weights for subwords
        merged_weights = self._merge_subword_weights(tokens, norm_weights, token_map)
        
        # Create text spans with color coding
        colored_text = self._create_colored_text(original_text, merged_weights, min_alpha)
        
        # Display colored text
        ax.text(0.05, 0.5, colored_text, fontsize=12, wrap=True)
        ax.axis('off')
        
        return fig
    
    def _map_tokens_to_text(self, tokens: List[str], text: str) -> Dict[int, List[int]]:
        """Map tokenized subwords back to original text positions."""
        # This is a simplified implementation - in practice you'd need 
        # more sophisticated alignment especially for subword tokenizers
        token_map = {}
        text_pos = 0
        
        for i, token in enumerate(tokens):
            # Skip special tokens
            if token.startswith('[') or token.startswith('<'):
                continue
                
            # Clean token (remove ## for BERT or Ġ for GPT-2 etc.)
            clean_token = token.replace('##', '').replace('Ġ', '')
            
            if not clean_token:
                continue
                
            # Find position in text
            while text_pos < len(text):
                if text[text_pos:text_pos + len(clean_token)].lower() == clean_token.lower():
                    token_map[i] = list(range(text_pos, text_pos + len(clean_token)))
                    text_pos += len(clean_token)
                    break
                text_pos += 1
                
        return token_map
    
    def _merge_subword_weights(
        self, 
        tokens: List[str], 
        weights: np.ndarray,
        token_map: Dict[int, List[