"""
Visualization endpoints for the Ethics Model API.

This module adds routes for creating visualizations of model outputs,
including attention maps, framework activations, and manipulation detection.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..modules.retriever import EthicsModel
from .dependencies import get_llm, get_model, get_tokenizer
from .settings import Settings, get_settings

# Configure logging
logger = logging.getLogger("ethics_model.api.visualization")

# Create router
router = APIRouter(tags=["Visualization"])

# Visualization related models
class VisualizationInput(BaseModel):
    """Input model for visualization requests."""
    text: str = Field(..., 
                    description="Text to visualize", 
                    min_length=10, 
                    max_length=10000)
    visualization_type: str = Field(..., 
                                  description="Type of visualization to generate",
                                  enum=["attention", "frameworks", "manipulation", "framing", "dissonance"])
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Companies should prioritize profit over everything else, regardless of ethical concerns.",
                "visualization_type": "frameworks"
            }
        }


class VisualizationOutput(BaseModel):
    """Output model for visualization results."""
    visualization_data: Dict[str, Any] = Field(..., description="Visualization data")
    visualization_type: str = Field(..., description="Type of visualization")
    plot_config: Dict[str, Any] = Field(..., description="Plotly configuration")


@router.post("/visualize", response_model=VisualizationOutput)
async def create_visualization(
    input_data: VisualizationInput,
    model: EthicsModel = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
    llm = Depends(get_llm),
    settings: Settings = Depends(get_settings)
):
    """
    Create a visualization of model analysis for the provided text.
    
    Generates visualizations for different aspects of the model's analysis:
    - attention: Attention weight heatmap
    - frameworks: Moral framework activation
    - manipulation: Manipulation technique detection
    - framing: Narrative framing analysis
    - dissonance: Cognitive dissonance detection
    
    Returns visualization data and Plotly configuration that can be used
    to render the visualization on the client side.
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            input_data.text, 
            return_tensors='pt', 
            max_length=settings.max_sequence_length,
            truncation=True, 
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(settings.device)
        attention_mask = inputs['attention_mask'].to(settings.device)
        
        # Decode tokens for visualization labels
        tokens = []
        for token_id in input_ids[0]:
            token = tokenizer.decode([token_id])
            if token:
                tokens.append(token)
        
        # Get LLM embeddings
        with torch.no_grad():
            llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
            hidden_states = llm_outputs.last_hidden_state
        
        # Run model inference
        with torch.no_grad():
            outputs = model(
                embeddings=hidden_states,
                attention_mask=attention_mask
            )
        
        # Create visualization based on type
        if input_data.visualization_type == "attention":
            return create_attention_visualization(outputs, tokens)
        elif input_data.visualization_type == "frameworks":
            return create_frameworks_visualization(outputs, tokens)
        elif input_data.visualization_type == "manipulation":
            return create_manipulation_visualization(outputs, tokens)
        elif input_data.visualization_type == "framing":
            return create_framing_visualization(outputs, tokens)
        elif input_data.visualization_type == "dissonance":
            return create_dissonance_visualization(outputs, tokens)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported visualization type: {input_data.visualization_type}"
            )
    
    except Exception as e:
        logger.exception(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


def create_attention_visualization(outputs, tokens):
    """Create attention weight visualization."""
    # Get attention weights
    attention_weights = outputs["attention_weights"]
    
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()
    
    # Average across heads if needed
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=1)
    
    # Use only the part corresponding to actual tokens
    max_len = min(len(tokens), attention_weights.shape[-1])
    attention_weights = attention_weights[0, :max_len, :max_len]
    tokens = tokens[:max_len]
    
    # Create visualization data
    visualization_data = {
        "attention_weights": attention_weights.tolist(),
        "tokens": tokens,
        "title": "Attention Weights Heatmap"
    }
    
    # Create Plotly configuration
    plot_config = {
        "type": "heatmap",
        "layout": {
            "title": "Attention Weights",
            "xaxis": {"title": "Tokens", "tickvals": list(range(len(tokens))), "ticktext": tokens},
            "yaxis": {"title": "Tokens", "tickvals": list(range(len(tokens))), "ticktext": tokens}
        },
        "data": [{
            "z": attention_weights.tolist(),
            "x": tokens,
            "y": tokens,
            "colorscale": "Viridis",
            "showscale": True
        }]
    }
    
    return {
        "visualization_data": visualization_data,
        "visualization_type": "attention",
        "plot_config": plot_config
    }


def create_frameworks_visualization(outputs, tokens):
    """Create moral framework visualization."""
    # Get framework outputs
    framework_outputs = outputs["framework_analysis"]["framework_outputs"]
    
    # Convert tensors to lists
    framework_data = {}
    framework_values = []
    framework_names = []
    
    for framework_name, framework_output in framework_outputs.items():
        if isinstance(framework_output, torch.Tensor):
            value = float(framework_output.mean().cpu().detach().numpy())
        else:
            value = float(framework_output)
        
        framework_data[framework_name] = value
        framework_names.append(framework_name)
        framework_values.append(value)
    
    # Create visualization data
    visualization_data = {
        "framework_activations": framework_data,
        "title": "Moral Framework Activations"
    }
    
    # Create Plotly configuration
    plot_config = {
        "type": "bar",
        "layout": {
            "title": "Moral Framework Activations",
            "xaxis": {"title": "Framework"},
            "yaxis": {"title": "Activation", "range": [0, 1]}
        },
        "data": [{
            "x": framework_names,
            "y": framework_values,
            "type": "bar",
            "marker": {
                "color": [
                    "rgba(31, 119, 180, 0.8)",
                    "rgba(255, 127, 14, 0.8)",
                    "rgba(44, 160, 44, 0.8)",
                    "rgba(214, 39, 40, 0.8)",
                    "rgba(148, 103, 189, 0.8)"
                ]
            }
        }]
    }
    
    return {
        "visualization_data": visualization_data,
        "visualization_type": "frameworks",
        "plot_config": plot_config
    }


def create_manipulation_visualization(outputs, tokens):
    """Create manipulation technique visualization."""
    # Get manipulation outputs
    technique_scores = outputs["manipulation_analysis"]["technique_scores"]
    
    # Convert tensors to lists
    technique_data = {}
    technique_names = []
    technique_values = []
    
    for technique_name, technique_score in technique_scores.items():
        if isinstance(technique_score, torch.Tensor):
            value = float(technique_score.mean().cpu().detach().numpy())
        else:
            value = float(technique_score)
        
        technique_data[technique_name] = value
        technique_names.append(technique_name)
        technique_values.append(value)
    
    # Create visualization data
    visualization_data = {
        "technique_scores": technique_data,
        "manipulation_score": float(outputs["manipulation_score"].cpu().detach().numpy()[0][0]),
        "title": "Manipulation Technique Detection"
    }
    
    # Create Plotly configuration
    plot_config = {
        "type": "radar",
        "layout": {
            "title": "Manipulation Technique Detection",
            "polar": {
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1]
                }
            }
        },
        "data": [{
            "type": "scatterpolar",
            "r": technique_values,
            "theta": technique_names,
            "fill": "toself",
            "name": "Manipulation Techniques"
        }]
    }
    
    return {
        "visualization_data": visualization_data,
        "visualization_type": "manipulation",
        "plot_config": plot_config
    }


def create_framing_visualization(outputs, tokens):
    """Create framing visualization."""
    # Get framing outputs
    framing_outputs = outputs["framing_analysis"]
    
    # Convert tensors to lists
    frame_scores = {}
    frame_names = []
    frame_values = []
    
    if "frame_scores" in framing_outputs:
        for frame_name, frame_score in framing_outputs["frame_scores"].items():
            if isinstance(frame_score, torch.Tensor):
                value = float(frame_score.mean().cpu().detach().numpy())
            else:
                value = float(frame_score)
            
            frame_scores[frame_name] = value
            frame_names.append(frame_name)
            frame_values.append(value)
    
    # Create visualization data
    visualization_data = {
        "frame_scores": frame_scores,
        "framing_strength": float(framing_outputs["framing_strength"].mean().cpu().detach().numpy()) 
            if isinstance(framing_outputs["framing_strength"], torch.Tensor) 
            else float(framing_outputs["framing_strength"]),
        "consistency_score": float(framing_outputs["consistency_score"].mean().cpu().detach().numpy())
            if isinstance(framing_outputs["consistency_score"], torch.Tensor)
            else float(framing_outputs["consistency_score"]),
        "title": "Narrative Framing Analysis"
    }
    
    # Create Plotly configuration
    plot_config = {
        "type": "pie",
        "layout": {
            "title": "Narrative Framing Analysis",
        },
        "data": [{
            "labels": frame_names,
            "values": frame_values,
            "type": "pie",
            "hole": 0.4,
            "textinfo": "label+percent",
            "insidetextorientation": "radial"
        }]
    }
    
    return {
        "visualization_data": visualization_data,
        "visualization_type": "framing",
        "plot_config": plot_config
    }


def create_dissonance_visualization(outputs, tokens):
    """Create cognitive dissonance visualization."""
    # Get dissonance outputs
    dissonance_outputs = outputs["dissonance_analysis"]
    
    # Convert tensors to lists
    value_conflicts = {}
    conflict_names = []
    conflict_values = []
    
    if "value_conflicts" in dissonance_outputs:
        for conflict_name, conflict_score in dissonance_outputs["value_conflicts"].items():
            if isinstance(conflict_score, torch.Tensor):
                value = float(conflict_score.mean().cpu().detach().numpy())
            else:
                value = float(conflict_score)
            
            value_conflicts[conflict_name] = value
            conflict_names.append(conflict_name)
            conflict_values.append(value)
    
    # Create visualization data
    visualization_data = {
        "value_conflicts": value_conflicts,
        "dissonance_score": float(dissonance_outputs["dissonance_score"].mean().cpu().detach().numpy())
            if isinstance(dissonance_outputs["dissonance_score"], torch.Tensor)
            else float(dissonance_outputs["dissonance_score"]),
        "resolution_strategy": dissonance_outputs.get("resolution_strategy", "unknown"),
        "title": "Cognitive Dissonance Analysis"
    }
    
    # Create Plotly configuration (horizontal bar chart)
    plot_config = {
        "type": "bar",
        "layout": {
            "title": "Cognitive Dissonance Analysis",
            "xaxis": {"title": "Conflict Intensity"},
            "yaxis": {"title": "Value Conflicts"},
        },
        "data": [{
            "x": conflict_values,
            "y": conflict_names,
            "type": "bar",
            "orientation": "h",
            "marker": {
                "color": conflict_values,
                "colorscale": "RdYlGn_r",
                "showscale": True,
                "colorbar": {
                    "title": "Conflict Intensity"
                }
            }
        }]
    }
    
    return {
        "visualization_data": visualization_data,
        "visualization_type": "dissonance",
        "plot_config": plot_config
    }
