"""
Narrative Detection and Manipulation Analysis

Components for detecting manipulation in narratives, cognitive dissonance,
and framing techniques in text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from .activation import get_activation, ReCA
from torch_geometric.nn import GCNConv


class FramingDetector(nn.Module):
    """
    Detects different framing techniques in text including:
    - Loss vs gain framing
    - Moral framing
    - Episodic vs thematic framing
    """
    
    def __init__(self, 
                 d_model: int,
                 n_framing_types: int = 6,
                 activation: str = "gelu"):
        super().__init__()
        
        # Frame type encoders
        self.frame_encoders = nn.ModuleDict({
            'loss_gain': nn.Linear(d_model, 2),
            'moral': nn.Linear(d_model, 5),  # Care, Fairness, Loyalty, Authority, Purity
            'episodic_thematic': nn.Linear(d_model, 2),
            'problem_solution': nn.Linear(d_model, 2),
            'conflict_consensus': nn.Linear(d_model, 2),
            'urgency_deliberation': nn.Linear(d_model, 2)
        })
        
        # Framing strength detector
        self.strength_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            get_activation(activation),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Frame consistency checker
        self.consistency_checker = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x: torch.Tensor, symbolic_constraints: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """
        Detect framing techniques in text.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - frame_scores: Scores for each framing type
                - framing_strength: Overall framing strength
                - consistency_score: Frame consistency score
        """
        batch_size, seq_len, _ = x.size()
        
        # Detect frame types
        frame_scores = {}
        for frame_type, encoder in self.frame_encoders.items():
            scores = encoder(x)
            frame_scores[frame_type] = torch.softmax(scores, dim=-1)
        
        # Measure framing strength
        framing_strength = self.strength_detector(x)
        
        # Check frame consistency
        consistency_features, _ = self.consistency_checker(x)
        consistency_score = torch.cosine_similarity(
            consistency_features[:, :-1, :],
            consistency_features[:, 1:, :],
            dim=-1
        ).mean(dim=-1, keepdim=True)
        
        result = {
            'frame_scores': frame_scores,
            'framing_strength': framing_strength,
            'consistency_score': consistency_score
        }
        if symbolic_constraints is not None:
            symbolic_result = symbolic_constraints(result)
            if symbolic_result is not None:
                return symbolic_result
        return result


class CognitiveDissonanceLayer(nn.Module):
    """
    Detects and measures cognitive dissonance in ethical narratives
    by identifying contradictory moral claims or values.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_moral_values: int = 8,
                 activation: str = "gelu"):
        super().__init__()
        
        # Value conflict detection
        self.value_encoder = nn.Linear(d_model, n_moral_values)
        
        # Contradiction detector
        self.contradiction_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            get_activation(activation),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Dissonance resolution predictor
        self.resolution_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            get_activation(activation),
            nn.Linear(d_model // 2, 3),  # Justify, Change, Minimize
            nn.Softmax(dim=-1)
        )
        
        # Value importance weighting
        self.value_importance = nn.Parameter(torch.randn(n_moral_values))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect cognitive dissonance in text.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - value_conflicts: Detected conflicts between values
                - dissonance_score: Overall dissonance score
                - resolution_strategy: Predicted dissonance resolution
        """
        # Encode moral values in text
        value_activations = torch.softmax(self.value_encoder(x), dim=-1)
        
        # Detect contradictions between adjacent segments
        pair_features = torch.cat([x[:, :-1, :], x[:, 1:, :]], dim=-1)
        contradiction_scores = self.contradiction_detector(pair_features)
        
        # Calculate value conflicts
        value_importance_normalized = torch.softmax(self.value_importance, dim=0)
        value_conflicts = contradiction_scores * torch.sum(
            value_activations[:, :-1, :] * value_activations[:, 1:, :],
            dim=-1,
            keepdim=True
        )
        
        # Predict dissonance resolution strategy
        resolution_strategy = self.resolution_predictor(x)
        
        # Overall dissonance score
        dissonance_score = value_conflicts.mean(dim=1)
        
        return {
            'value_conflicts': value_conflicts,
            'dissonance_score': dissonance_score,
            'resolution_strategy': resolution_strategy,
            'value_activations': value_activations
        }


class NarrativeManipulationDetector(nn.Module):
    """
    Comprehensive detector for various manipulation techniques in narratives
    including emotional appeals, logical fallacies, and framing biases.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_manipulation_types: int = 8,
                 activation: str = "gelu"):
        super().__init__()
        
        # Manipulation technique detectors
        self.manipulation_detectors = nn.ModuleDict({
            'emotional_appeal': self._create_detector(d_model, activation),
            'false_dichotomy': self._create_detector(d_model, activation),
            'appeal_to_authority': self._create_detector(d_model, activation),
            'bandwagon': self._create_detector(d_model, activation),
            'loaded_language': self._create_detector(d_model, activation),
            'cherry_picking': self._create_detector(d_model, activation),
            'straw_man': self._create_detector(d_model, activation),
            'slippery_slope': self._create_detector(d_model, activation)
        })
        
        # Aggregate manipulation score
        self.aggregator = nn.Sequential(
            nn.Linear(len(self.manipulation_detectors), d_model // 2),
            get_activation(activation),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence calibrator
        self.calibrator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            get_activation(activation),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def _create_detector(self, d_model: int, activation: str) -> nn.Module:
        """Create a detector for a specific manipulation technique."""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            get_activation(activation),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect various manipulation techniques in text.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - technique_scores: Individual manipulation technique scores
                - aggregate_score: Overall manipulation score
                - confidence: Confidence in detection
                - manipulation_map: Spatial map of manipulation
        """
        # Detect each manipulation technique
        technique_scores = {}
        score_tensors = []
        
        for technique, detector in self.manipulation_detectors.items():
            score = detector(x)
            technique_scores[technique] = score
            score_tensors.append(score)
        
        # Combine scores
        combined_scores = torch.cat(score_tensors, dim=-1)
        
        # Calculate aggregate manipulation score
        aggregate_score = self.aggregator(combined_scores)
        
        # Calculate confidence
        confidence = self.calibrator(x)
        
        # Create manipulation map (2D representation)
        manipulation_map = aggregate_score * confidence
        
        return {
            'technique_scores': technique_scores,
            'aggregate_score': aggregate_score,
            'confidence': confidence,
            'manipulation_map': manipulation_map
        }


class PropagandaDetector(nn.Module):
    """
    Specialized detector for propaganda techniques and systematic
    manipulation patterns in communication.
    """
    
    def __init__(self, 
                 d_model: int,
                 n_propaganda_techniques: int = 14,
                 activation: str = "gelu"):
        super().__init__()
        
        # Propaganda technique encoding based on research
        self.technique_embeddings = nn.Embedding(n_propaganda_techniques, d_model)
        
        # Pattern matcher for propaganda techniques
        self.pattern_matcher = nn.Conv1d(
            in_channels=d_model,
            out_channels=n_propaganda_techniques,
            kernel_size=3,
            padding=1
        )
        
        # Propaganda intensity scorer
        self.intensity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            get_activation(activation),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Source credibility estimator
        self.credibility_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            get_activation(activation),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect propaganda techniques in text.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing:
                - technique_matches: Matched propaganda techniques
                - intensity_score: Propaganda intensity
                - credibility_score: Source credibility estimation
        """
        # Transpose for convolution
        x_transposed = x.transpose(1, 2)
        
        # Detect propaganda techniques
        technique_matches = self.pattern_matcher(x_transposed)
        technique_matches = torch.sigmoid(technique_matches.transpose(1, 2))
        
        # Score propaganda intensity
        intensity_score = self.intensity_scorer(x)
        
        # Estimate source credibility
        credibility_score = self.credibility_estimator(x.mean(dim=1, keepdim=True))
        
        return {
            'technique_matches': technique_matches,
            'intensity_score': intensity_score,
            'credibility_score': credibility_score,
            'inverse_credibility': 1 - credibility_score  # Higher when less credible
        }


class NarrativeGraphLayer(nn.Module):
    """
    GNN-Layer für narrative Graphstrukturen (GCNConv).
    Kann als Baustein für hybride Narrative-Modelle genutzt werden.
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu"):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.activation = get_activation(activation)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gcn(x, edge_index)
        return self.activation(x)
