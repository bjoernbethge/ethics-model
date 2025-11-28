"""
Example script to start the Ethics Model API server.

This script demonstrates how to start the Ethics Model API server
and use various configuration options.
"""

import argparse
import logging
import os

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for API configuration
os.environ["ETHICS_API_PORT"] = "8000"
os.environ["ETHICS_API_DEBUG"] = "True"
os.environ["ETHICS_API_MAX_SEQUENCE_LENGTH"] = "256"

# Optional: Set path to model checkpoint
# os.environ["ETHICS_API_CHECKPOINT_PATH"] = "checkpoints/model.pt"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Ethics Model API server")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host address to bind"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    
    return parser.parse_args()


def main():
    """Run the API server."""
    args = parse_args()
    
    logger.info(f"Starting Ethics Model API server on http://{args.host}:{args.port}")
    logger.info("Press CTRL+C to stop.")
    
    # Start API
    uvicorn.run(
        "ethics_model.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
