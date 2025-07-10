"""
Ethics Model Main Architecture

The main architecture that combines all components for comprehensive
ethical analysis and narrative manipulation detection.
Enhanced with GraphBrain semantic hypergraph integration, explainability,
uncertainty quantification, and advanced graph reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from torch_geometric.data import Data

from .modules.moral import (
    MoralFrameworkEmbedding,
    MultiFrameworkProcessor,
    EthicalCrossDomainLayer,
    EthicalPrincipleEncoder
)
from .modules.attention import (
    EthicalAttention,
    MoralIntuitionAttention,
    NarrativeFrameAttention,
    DoubleProcessingAttention,
    GraphAttentionLayer
)
from .modules.activation import get_activation, ReCA
from .modules.narrative import (
    NarrativeManipulationDetector,
    FramingDetector,
    CognitiveDissonanceLayer,
    PropagandaDetector
)
# Import GraphBrain module
from .modules.graph_semantic import (
    SemanticGraphProcessor,
    HypergraphConverter
)


class EnhancedEthicsModel(nn.Module):
    """
    Enhanced ethics analysis model that integrates multiple components
    for detecting manipulation, analyzing moral frameworks, and processing
    ethical narratives.
    
    Includes GraphBrain semantic hypergraph processing for more sophisticated ethical analysis.
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 vocab_size: int = 30000,
                 max_seq_length: int = 512,
                 activation: str = "gelu",
                 use_gnn: bool = True,  # Default to using GNN
                 use_graphbrain: bool = True,  # Default to using GraphBrain
                 parser_lang: str = "en"):
        super().__init__()
        
        # Input embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Main processing layers
        self.moral_framework_processor = MultiFrameworkProcessor(d_model)
        self.ethical_attention = EthicalAttention(d_model, n_heads, activation=activation)
        self.moral_intuition = MoralIntuitionAttention(d_model, activation=activation)
        self.dual_process = DoubleProcessingAttention(d_model, n_heads, activation=activation)
        
        # Narrative analysis components
        self.narrative_frame_attention = NarrativeFrameAttention(d_model, activation=activation)
        self.framing_detector = FramingDetector(d_model)
        self.cognitive_dissonance = CognitiveDissonanceLayer(d_model)
        self.manipulation_detector = NarrativeManipulationDetector(d_model)
        self.propaganda_detector = PropagandaDetector(d_model)
        
        # Cross-domain ethical processing
        self.cross_domain_layer = EthicalCrossDomainLayer(d_model)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                activation="gelu",
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # GraphBrain integration
        self.use_graphbrain = use_graphbrain
        if use_graphbrain:
            self.semantic_graph_processor = SemanticGraphProcessor(
                d_model=d_model,
                parser_lang=parser_lang
            )
            
            # Graph integration layer
            self.graph_integration = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                get_activation(activation)
            )
        
        # Output projections
        self.ethics_score_projection = nn.Sequential(
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.manipulation_projection = nn.Sequential(
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Meta-cognitive layer for self-reflection
        self.meta_cognitive = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            get_activation(activation),
            nn.Linear(d_model, d_model // 2)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        self.use_gnn = use_gnn
        if use_gnn:
            self.graph_attention = GraphAttentionLayer(d_model, d_model, heads=n_heads, activation=activation)
        
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                moral_context: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                embeddings: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None,
                graph_data: Optional[List[Dict[str, Any]]] = None,
                symbolic_constraints: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process input through the complete ethics model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            moral_context: Optional moral context vector
            edge_index: Optional edge index for GNN processing (torch_geometric)
            embeddings: Optional LLM-Embeddings (batch_size, seq_len, d_model)
            texts: Optional raw text inputs (required for GraphBrain)
            graph_data: Optional preprocessed graph data for each text
            symbolic_constraints: Optional symbolic constraints
            
        Returns:
            Dictionary containing comprehensive ethics analysis
        """
        # Get embeddings
        if embeddings is not None:
            hidden_states = embeddings
        else:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            token_embeddings = self.embedding(input_ids)
            position_embeddings = self.position_embedding(position_ids)
            embeddings = token_embeddings + position_embeddings
            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)
            hidden_states = embeddings
        
        # Process through transformer layers
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # GraphBrain processing
        graph_enhanced_states = None
        if self.use_graphbrain and texts is not None:
            graph_outputs = self.semantic_graph_processor(texts, hidden_states)
            graph_enhanced_states = graph_outputs['graph_embeddings']
            
            # Integrate graph knowledge
            if graph_enhanced_states is not None:
                # Combine with hidden states
                combined = torch.cat([hidden_states, graph_enhanced_states], dim=-1)
                hidden_states = self.graph_integration(combined)
        
        # Optional: GNN-Verarbeitung (using provided edge_index or from graph_data)
        if self.use_gnn:
            if edge_index is not None:
                hidden_states = self.graph_attention(hidden_states, edge_index)
            elif graph_data is not None and 'edge_index' in graph_data[0]:
                # Note: This is simplified - in practice, you'd need batching for multiple graphs
                batch_edge_index = graph_data[0]['edge_index']
                if batch_edge_index is not None:
                    hidden_states = self.graph_attention(hidden_states, batch_edge_index)
        
        # Moral framework analysis
        framework_outputs = self.moral_framework_processor(hidden_states, symbolic_constraints=symbolic_constraints)
        
        # Ethical attention processing
        ethical_attended, attention_weights = self.ethical_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            moral_context=moral_context,
            mask=attention_mask,
            symbolic_constraints=symbolic_constraints
        )
        
        # Moral intuition processing
        intuition_outputs = self.moral_intuition(ethical_attended, moral_context)
        
        # Dual-process system analysis
        dual_process_output, system_outputs = self.dual_process(ethical_attended)
        
        # Narrative analysis
        narrative_outputs = self.narrative_frame_attention(dual_process_output)
        framing_outputs = self.framing_detector(dual_process_output, symbolic_constraints=symbolic_constraints)
        dissonance_outputs = self.cognitive_dissonance(dual_process_output)
        
        # Manipulation detection
        manipulation_outputs = self.manipulation_detector(dual_process_output)
        propaganda_outputs = self.propaganda_detector(dual_process_output)
        
        # Cross-domain ethical processing
        cross_domain_outputs = self.cross_domain_layer(dual_process_output)
        
        # Combine for final ethics score
        combined_features = torch.cat([
            dual_process_output.mean(dim=1),
            framework_outputs['consensus_output'].mean(dim=1),
            cross_domain_outputs.mean(dim=1)
        ], dim=-1)
        
        # Meta-cognitive processing
        meta_cognitive_output = self.meta_cognitive(combined_features)
        
        # Generate final scores
        ethics_score = self.ethics_score_projection(meta_cognitive_output)
        manipulation_score = self.manipulation_projection(meta_cognitive_output)
        
        # Compile comprehensive output
        outputs = {
            'ethics_score': ethics_score,
            'manipulation_score': manipulation_score,
            'framework_analysis': framework_outputs,
            'intuition_analysis': intuition_outputs,
            'dual_process_analysis': system_outputs,
            'narrative_analysis': narrative_outputs,
            'framing_analysis': framing_outputs,
            'dissonance_analysis': dissonance_outputs,
            'manipulation_analysis': manipulation_outputs,
            'propaganda_analysis': propaganda_outputs,
            'attention_weights': attention_weights,
            'hidden_states': hidden_states,
            'meta_cognitive_features': meta_cognitive_output
        }
        
        # Add GraphBrain outputs if used
        if self.use_graphbrain and graph_enhanced_states is not None:
            outputs['graph_outputs'] = {
                'enhanced_states': graph_enhanced_states,
                'ethical_relations': graph_outputs.get('ethical_relations', [])
            }
        
        return outputs
    
    def get_ethical_summary(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a human-readable summary of the ethics analysis.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Summarized ethics analysis
        """
        summary = {
            'overall_ethics_score': outputs['ethics_score'].item(),
            'manipulation_risk': outputs['manipulation_score'].item(),
            'dominant_framework': self._get_dominant_framework(outputs['framework_analysis']),
            'emotional_intensity': outputs['intuition_analysis']['emotional_intensity'].mean().item(),
            'system_conflict': self._calculate_system_conflict(outputs['dual_process_analysis']),
            'main_manipulation_techniques': self._identify_top_manipulation_techniques(outputs['manipulation_analysis']),
            'cognitive_dissonance_level': outputs['dissonance_analysis']['dissonance_score'].mean().item(),
            'framing_strength': outputs['framing_analysis']['framing_strength'].mean().item(),
            'propaganda_risk': outputs['propaganda_analysis']['intensity_score'].mean().item()
        }
        
        # Add semantic graph insights if available
        if 'graph_outputs' in outputs:
            ethical_relations = outputs['graph_outputs'].get('ethical_relations', [])
            if ethical_relations:
                summary['semantic_graph_insights'] = self._summarize_ethical_relations(ethical_relations)
        
        return summary
    
    def _get_dominant_framework(self, framework_outputs: Dict[str, Any]) -> str:
        """Identify the dominant moral framework."""
        framework_scores: Dict[str, float] = {}
        for framework, output in framework_outputs['framework_outputs'].items():
            framework_scores[framework] = output.mean().item()
        return max(framework_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_system_conflict(self, system_outputs: Dict[str, Any]) -> float:
        """Calculate conflict between dual processing systems."""
        weights: torch.Tensor = system_outputs['resolution_weights']
        conflict: float = torch.abs(weights[..., 0] - weights[..., 1]).mean().item()
        return conflict
    
    def _identify_top_manipulation_techniques(
        self, manipulation_outputs: Dict[str, Any], top_k: int = 3
    ) -> List[str]:
        """Identify the top manipulation techniques detected."""
        technique_scores: Dict[str, torch.Tensor] = manipulation_outputs['technique_scores']
        scores: List[Tuple[str, float]] = []
        for technique, score in technique_scores.items():
            scores.append((technique, score.mean().item()))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [technique for technique, _ in scores[:top_k]]
    
    def _summarize_ethical_relations(self, ethical_relations: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
        """Summarize ethical relations extracted from semantic graph."""
        # In a full implementation, this would provide rich insights
        # Here's a simplified placeholder
        summary = {}
        
        if not ethical_relations:
            return {"message": "No ethical relations extracted"}
            
        # Count entity types
        for relation_dict in ethical_relations:
            for key, entities in relation_dict.items():
                if key not in summary:
                    summary[key] = 0
                summary[key] += len(entities)
                
        return summary


# Maintain backward compatibility with original EthicsModel
class EthicsModel(EnhancedEthicsModel):
    """Backwards-compatible version of the original EthicsModel."""
    
    def __init__(self, *args, **kwargs):
        # Explicitly disable new features by default
        kwargs['use_graphbrain'] = kwargs.get('use_graphbrain', False)
        super().__init__(*args, **kwargs)


# Helper function for model initialization
def create_ethics_model(config: Dict[str, Any]) -> Union[EthicsModel, EnhancedEthicsModel]:
    """
    Factory function to create an ethics model with given configuration.
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        Initialized model instance (EthicsModel or EnhancedEthicsModel)
    """
    use_enhanced = config.get('use_enhanced', True)  # Default to enhanced model
    use_graphbrain = config.get('use_graphbrain', True)  # Default to using GraphBrain
    use_legacy = config.get('use_legacy', False)  # New option to force legacy model
    
    # Determine which model class to use
    if use_legacy:
        model_class = EthicsModel
    else:
        model_class = EnhancedEthicsModel
    
    return model_class(
        input_dim=config.get('input_dim', 512),
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        vocab_size=config.get('vocab_size', 30000),
        max_seq_length=config.get('max_seq_length', 512),
        activation=config.get('activation', "gelu"),
        use_gnn=config.get('use_gnn', True),
        use_graphbrain=use_graphbrain,
        parser_lang=config.get('parser_lang', "en")
    )


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'input_dim': 512,
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'vocab_size': 30000,
        'max_seq_length': 512,
        'activation': "gelu",
        'use_gnn': True,
        'use_enhanced': True,
        'use_graphbrain': True,
        'parser_lang': "en"
    }
    
    # Create model
    model = create_ethics_model(config)
    
    # Example input
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Example texts for semantic graph processing
    texts = [
        "Companies should prioritize profit over environmental concerns.",
        "We must protect natural resources for future generations."
    ]
    
    # Forward pass
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        texts=texts
    )
    
    # Get summary
    summary = model.get_ethical_summary(outputs)
    print("Ethics Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
