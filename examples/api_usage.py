"""
Example usage of the Ethics Model API.

This script demonstrates the usage of visualization and training functions
of the Ethics Model API.
"""

import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ethics_model.api.client import EthicsModelClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize client
client = EthicsModelClient(base_url="http://localhost:8000")

# Example texts for various analyses
EXAMPLE_TEXTS = {
    "profit": "Companies should prioritize profit over everything else, regardless of ethical concerns.",
    "utilitarian": "We should always choose the action that produces the greatest happiness for the greatest number of people.",
    "deontological": "It is our duty to tell the truth, regardless of the consequences.",
    "virtue": "A virtuous person acts from character, not from calculation.",
    "manipulative": "Anyone who doesn't support this is either stupid or malicious - there's no other possibility."
}


def test_visualizations():
    """Test visualization functions."""
    logger.info("=== Testing visualization functions ===")
    
    # Create visualization for each text and visualization type
    for text_name, text in EXAMPLE_TEXTS.items():
        logger.info(f"Analyzing text: {text_name}")
        
        # Framework visualization
        frameworks_viz = client.visualize(text, visualization_type="frameworks")
        logger.info(f"  Framework visualization created: {len(frameworks_viz['visualization_data'])} data points")
        
        # Manipulation visualization
        manip_viz = client.visualize(text, visualization_type="manipulation")
        logger.info(f"  Manipulation visualization created: {len(manip_viz['visualization_data'])} data points")
        
        # Attention visualization (show details only for the first text)
        if text_name == "manipulative":
            attention_viz = client.visualize(text, visualization_type="attention")
            
            # Create Plotly visualization with the data
            logger.info("Attention heatmap will be displayed (close the window to continue)...")
            
            # Extract Plotly configuration
            plot_config = attention_viz["plot_config"]
            
            # Create figure
            fig = go.Figure(data=plot_config["data"], layout=plot_config["layout"])
            
            # Display (or save as HTML file)
            fig.write_html("attention_heatmap.html")
            logger.info("  Attention heatmap saved as 'attention_heatmap.html'.")


def test_training():
    """Test training functions."""
    logger.info("=== Testing training functions ===")
    
    # Create simple training dataset
    train_texts = list(EXAMPLE_TEXTS.values())
    
    # Simple labels for demonstration
    ethics_labels = [0.2, 0.8, 0.7, 0.9, 0.1]  # Low for 'profit' and 'manipulative'
    manipulation_labels = [0.7, 0.2, 0.3, 0.1, 0.9]  # High for 'profit' and 'manipulative'
    
    logger.info(f"Starting training with {len(train_texts)} examples...")
    
    # Start training with minimal parameters (for quick demo)
    task_id = client.train(
        train_texts=train_texts,
        ethics_labels=ethics_labels,
        manipulation_labels=manipulation_labels,
        epochs=2,  # Only a few epochs for the demo
        batch_size=2,
        learning_rate=1e-4,
        checkpoint_name="demo_model"
    )
    
    logger.info(f"Training started. Task ID: {task_id}")
    
    # Wait for training to complete
    status = client.get_training_status(task_id)
    
    while status["status"] not in ["completed", "failed"]:
        progress = status["progress"] * 100 if status["progress"] is not None else 0
        logger.info(f"Progress: {progress:.1f}% (Epoch {status['current_epoch']}/{status['total_epochs']})")
        
        if status["train_loss"] is not None:
            logger.info(f"  Train Loss: {status['train_loss']:.4f}")
        if status["val_loss"] is not None:
            logger.info(f"  Validation Loss: {status['val_loss']:.4f}")
        
        # Wait briefly and then update status
        time.sleep(2)
        status = client.get_training_status(task_id)
    
    if status["status"] == "completed":
        logger.info("Training completed successfully!")
        
        # Get results
        result = client.get_training_result(task_id)
        
        # Display results
        logger.info("=== Training Results ===")
        logger.info(f"Completed epochs: {result['epochs_completed']}")
        logger.info(f"Training ethics accuracy: {result['train_ethics_accuracy']:.4f}")
        logger.info(f"Training manipulation accuracy: {result['train_manipulation_accuracy']:.4f}")
        
        if result["val_ethics_accuracy"] is not None:
            logger.info(f"Validation ethics accuracy: {result['val_ethics_accuracy']:.4f}")
        if result["val_manipulation_accuracy"] is not None:
            logger.info(f"Validation manipulation accuracy: {result['val_manipulation_accuracy']:.4f}")
        
        logger.info(f"Training duration: {result['training_duration_seconds']:.1f} seconds")
        
        if result["checkpoint_path"]:
            logger.info(f"Model checkpoint saved at: {result['checkpoint_path']}")
        
        # Plot training history
        if result["train_loss_history"] and len(result["train_loss_history"]) > 0:
            epochs = list(range(1, len(result["train_loss_history"]) + 1))
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, result["train_loss_history"], 'b-', label='Training Loss')
            
            if result["val_loss_history"] and len(result["val_loss_history"]) > 0:
                plt.plot(epochs, result["val_loss_history"], 'r-', label='Validation Loss')
            
            plt.title('Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_history.png')
            logger.info("Training history saved as 'training_history.png'.")
    else:
        logger.error(f"Training failed: {status.get('error', 'Unknown error')}")


def test_inference_with_trained_model():
    """Test inference with the trained model."""
    logger.info("=== Testing inference with trained model ===")
    
    # New text for testing
    test_text = "It is completely absurd to claim that companies have a social responsibility that goes beyond profit maximization."
    
    # Perform analysis
    logger.info(f"Analyzing text: {test_text}")
    result = client.analyze(test_text, include_details=True)
    
    # Display results
    logger.info("=== Analysis Results ===")
    logger.info(f"Ethics score: {result['ethics_score']:.4f}")
    logger.info(f"Manipulation score: {result['manipulation_score']:.4f}")
    logger.info(f"Dominant framework: {result['dominant_framework']}")
    
    # Details
    logger.info("Summary:")
    for key, value in result["summary"].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            logger.info(f"  {key}: {', '.join(value)}")
        else:
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # Check API availability
    if not client.ping():
        logger.error("Error: API is not available. Please start the API server.")
        exit(1)
    
    try:
        # Test functions
        test_visualizations()
        test_training()
        test_inference_with_trained_model()
        
        logger.info("All tests completed successfully!")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
