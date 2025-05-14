"""
FastAPI application for Ethics Model serving and inference.

This module defines the main FastAPI application with routes for
text analysis, ethical evaluation, and visualization using the Ethics Model.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Union

import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..model import EthicsModel, create_ethics_model
from .dependencies import get_model, get_tokenizer, get_llm
from .settings import Settings, get_settings
from .app_training import router as training_router
from .app_visualization import router as visualization_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ethics_model.api")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles model loading at startup and proper resource cleanup at shutdown.
    """
    logger.info("Starting Ethics Model API server")
    # Load models and resources at startup
    logger.info("Initializing models...")
    # The actual model loading happens in dependencies.py
    # This is just a placeholder for any additional startup logic
    
    yield  # Server is running
    
    # Cleanup at shutdown
    logger.info("Shutting down Ethics Model API server")


# Create FastAPI application
app = FastAPI(
    title="Ethics Model API",
    description="API for ethical text analysis, manipulation detection, and moral reasoning",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for training and visualization
app.include_router(training_router, prefix="/training")
app.include_router(visualization_router, prefix="/visualization")


# Input and output models
class TextInput(BaseModel):
    """Input model for text analysis requests."""
    text: str = Field(..., 
                      description="Text to analyze", 
                      min_length=10, 
                      max_length=10000, 
                      example="This is an example text for ethical analysis.")
    include_details: bool = Field(False, 
                                 description="Include detailed analysis in response")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Companies should prioritize profit over everything else, regardless of ethical concerns.",
                "include_details": True
            }
        }


class TextBatchInput(BaseModel):
    """Input model for batch text analysis."""
    texts: List[str] = Field(..., 
                           description="List of texts to analyze", 
                           max_items=50)
    include_details: bool = Field(False, 
                                description="Include detailed analysis in response")


class AnalysisResponse(BaseModel):
    """Response model for text analysis results."""
    ethics_score: float = Field(..., 
                              description="Overall ethics score (0-1)")
    manipulation_score: float = Field(..., 
                                    description="Manipulation risk score (0-1)")
    dominant_framework: str = Field(..., 
                                  description="Dominant moral framework")
    summary: Dict[str, Any] = Field(..., 
                                  description="Summary of the ethical analysis")
    details: Optional[Dict[str, Any]] = Field(None, 
                                           description="Detailed analysis results")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch text analysis results."""
    results: List[AnalysisResponse] = Field(..., 
                                          description="List of analysis results")
    average_ethics_score: float = Field(..., 
                                      description="Average ethics score across all texts")
    average_manipulation_score: float = Field(..., 
                                           description="Average manipulation score across all texts")


class AsyncTaskResponse(BaseModel):
    """Response model for asynchronous task status."""
    task_id: str = Field(..., description="Task ID for tracking the request")
    status: str = Field(..., description="Task status")


class TaskStatusResponse(BaseModel):
    """Response model for task status checks."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    result: Optional[AnalysisResponse] = Field(None, description="Task result if completed")


# In-memory task store (for demo purposes)
# In production, use a proper task queue like Celery/Redis
tasks = {}


# Routes
@app.get("/", tags=["Status"])
async def root():
    """Root endpoint with API information."""
    return {
        "status": "online",
        "api": "Ethics Model",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(
    input_data: TextInput,
    model: EthicsModel = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
    llm = Depends(get_llm),
    settings: Settings = Depends(get_settings)
):
    """
    Analyze text for ethical content and manipulation techniques.
    
    Returns comprehensive ethical analysis including:
    - Ethics score
    - Manipulation score
    - Moral framework analysis
    - Narrative framing detection
    - Manipulation technique identification
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
        
        # Generate summary
        ethical_summary = model.get_ethical_summary(outputs)
        
        # Create response
        response = {
            "ethics_score": float(outputs["ethics_score"].cpu().numpy()[0][0]),
            "manipulation_score": float(outputs["manipulation_score"].cpu().numpy()[0][0]),
            "dominant_framework": ethical_summary["dominant_framework"],
            "summary": ethical_summary,
            "details": outputs if input_data.include_details else None
        }
        
        # If include_details is True but outputs is complex, sanitize it
        if input_data.include_details:
            response["details"] = _sanitize_outputs(outputs)
        
        return response
        
    except Exception as e:
        logger.exception(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def analyze_text_batch(
    input_data: TextBatchInput,
    model: EthicsModel = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
    llm = Depends(get_llm),
    settings: Settings = Depends(get_settings)
):
    """
    Batch analyze multiple texts for ethical content.
    
    Returns analysis for each text and aggregated metrics across all texts.
    Limited to 50 texts per request.
    """
    try:
        if len(input_data.texts) > 50:
            raise HTTPException(status_code=400, detail="Maximum batch size is 50 texts")
        
        results = []
        total_ethics = 0
        total_manipulation = 0
        
        for text in input_data.texts:
            # Process each text separately
            # Similar to analyze_text but in a loop
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                max_length=settings.max_sequence_length,
                truncation=True, 
                padding='max_length'
            )
            input_ids = inputs['input_ids'].to(settings.device)
            attention_mask = inputs['attention_mask'].to(settings.device)
            
            with torch.no_grad():
                llm_outputs = llm.model.transformer(input_ids) if hasattr(llm, 'model') else llm.transformer(input_ids)
                hidden_states = llm_outputs.last_hidden_state
                
                outputs = model(
                    embeddings=hidden_states,
                    attention_mask=attention_mask
                )
            
            ethical_summary = model.get_ethical_summary(outputs)
            
            ethics_score = float(outputs["ethics_score"].cpu().numpy()[0][0])
            manipulation_score = float(outputs["manipulation_score"].cpu().numpy()[0][0])
            
            total_ethics += ethics_score
            total_manipulation += manipulation_score
            
            result = {
                "ethics_score": ethics_score,
                "manipulation_score": manipulation_score,
                "dominant_framework": ethical_summary["dominant_framework"],
                "summary": ethical_summary,
                "details": _sanitize_outputs(outputs) if input_data.include_details else None
            }
            
            results.append(result)
        
        # Calculate averages
        avg_ethics = total_ethics / len(input_data.texts)
        avg_manipulation = total_manipulation / len(input_data.texts)
        
        return {
            "results": results,
            "average_ethics_score": avg_ethics,
            "average_manipulation_score": avg_manipulation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")


@app.post("/analyze/async", response_model=AsyncTaskResponse, tags=["Analysis"])
async def analyze_text_async(
    input_data: TextInput,
    background_tasks: BackgroundTasks,
    model: EthicsModel = Depends(get_model),
    tokenizer = Depends(get_tokenizer),
    llm = Depends(get_llm),
    settings: Settings = Depends(get_settings)
):
    """
    Asynchronously analyze text for ethical content.
    
    Returns a task ID that can be used to check the status and retrieve
    results once processing is complete.
    """
    import uuid
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None}
    
    background_tasks.add_task(
        _process_analysis_task,
        task_id=task_id,
        text=input_data.text,
        include_details=input_data.include_details,
        model=model,
        tokenizer=tokenizer,
        llm=llm,
        settings=settings
    )
    
    return {"task_id": task_id, "status": "processing"}


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Analysis"])
async def get_task_status(task_id: str):
    """
    Check the status of an asynchronous analysis task.
    
    Returns the current status and the result if the task is completed.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "status": tasks[task_id]["status"],
        "result": tasks[task_id]["result"]
    }


@app.get("/frameworks", tags=["Metadata"])
async def get_moral_frameworks():
    """
    Get information about supported moral frameworks.
    
    Returns details about the ethical frameworks used in the analysis.
    """
    return {
        "frameworks": [
            {
                "id": "deontological",
                "name": "Deontological Ethics",
                "description": "Focused on the rightness or wrongness of actions themselves, as opposed to the consequences of those actions."
            },
            {
                "id": "utilitarian",
                "name": "Utilitarian Ethics",
                "description": "Focused on maximizing happiness and well-being for the most people."
            },
            {
                "id": "virtue",
                "name": "Virtue Ethics",
                "description": "Centered on the moral character of the person performing the action."
            },
            {
                "id": "care",
                "name": "Ethics of Care",
                "description": "Emphasizes the importance of response to the needs of others, especially those who are vulnerable."
            },
            {
                "id": "fairness",
                "name": "Fairness/Justice",
                "description": "Based on the idea that everyone should receive what they deserve, with fair procedures and outcomes."
            }
        ]
    }


@app.get("/manipulation-techniques", tags=["Metadata"])
async def get_manipulation_techniques():
    """
    Get information about detectable manipulation techniques.
    
    Returns details about the manipulation techniques that can be identified.
    """
    return {
        "techniques": [
            {
                "id": "false_dilemma",
                "name": "False Dilemma",
                "description": "Presenting only two options when others exist."
            },
            {
                "id": "loaded_language",
                "name": "Loaded Language",
                "description": "Using specific words to influence an audience's emotional response."
            },
            {
                "id": "bandwagon",
                "name": "Bandwagon",
                "description": "Appealing to popularity or the fact that many people do something."
            },
            {
                "id": "causal_oversimplification",
                "name": "Causal Oversimplification",
                "description": "Assuming a single cause or reason when there are multiple factors."
            },
            {
                "id": "appeal_to_authority",
                "name": "Appeal to Authority",
                "description": "Using the opinion or position of an authority figure to support an argument."
            },
            {
                "id": "appeal_to_fear",
                "name": "Appeal to Fear/Prejudice",
                "description": "Seeking to build support by instilling fear or prejudice."
            },
            {
                "id": "strawman",
                "name": "Strawman",
                "description": "Misrepresenting someone's argument to make it easier to attack."
            },
            {
                "id": "whataboutism",
                "name": "Whataboutism",
                "description": "Diverting attention by changing the subject to someone else's misconduct."
            },
            {
                "id": "red_herring",
                "name": "Red Herring",
                "description": "Introducing irrelevant material to distract from the main issue."
            },
            {
                "id": "blackwhite_fallacy",
                "name": "Black-and-White Fallacy",
                "description": "Presenting two alternative states as the only possibilities."
            }
        ]
    }


@app.get("/health", tags=["Status"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns the current health status of the API.
    """
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Helper functions
def _process_analysis_task(
    task_id: str,
    text: str,
    include_details: bool,
    model: EthicsModel,
    tokenizer,
    llm,
    settings: Settings
):
    """Background task for asynchronous text analysis."""
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            max_length=settings.max_sequence_length,
            truncation=True, 
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(settings.device)
        attention_mask = inputs['attention_mask'].to(settings.device)
        
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
        
        # Generate summary
        ethical_summary = model.get_ethical_summary(outputs)
        
        # Create response
        response = {
            "ethics_score": float(outputs["ethics_score"].cpu().numpy()[0][0]),
            "manipulation_score": float(outputs["manipulation_score"].cpu().numpy()[0][0]),
            "dominant_framework": ethical_summary["dominant_framework"],
            "summary": ethical_summary,
            "details": None
        }
        
        # If include_details is True, sanitize complex outputs
        if include_details:
            response["details"] = _sanitize_outputs(outputs)
        
        # Update task status
        tasks[task_id] = {"status": "completed", "result": response}
        
    except Exception as e:
        logger.exception(f"Error in async task: {str(e)}")
        tasks[task_id] = {"status": "failed", "result": {"error": str(e)}}


def _sanitize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize model outputs for JSON serialization.
    Converts PyTorch tensors to Python native types.
    """
    sanitized = {}
    
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            # Convert tensors to Python types
            sanitized[key] = value.cpu().detach().numpy().tolist()
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = _sanitize_outputs(value)
        elif isinstance(value, list):
            # Sanitize lists
            sanitized[key] = [
                item.cpu().detach().numpy().tolist() if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            # Keep other types as-is
            sanitized[key] = value
    
    return sanitized


# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Load settings
    settings = get_settings()
    
    # Run the API server
    uvicorn.run(
        "ethics_model.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
