"""
Standalone script to run the Ethics Model API server.

This script provides a convenient way to start the API server
with configuration from environment variables or command line arguments.
"""

import argparse
import logging
import os
import sys

import uvicorn

from .settings import get_settings, Settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Ethics Model API server")
    
    parser.add_argument(
        "--host", 
        type=str, 
        help="Host address to bind (default: from settings/env)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        help="Port to bind (default: from settings/env)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with auto-reload"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        help="Path to model directory (default: from settings/env)"
    )
    parser.add_argument(
        "--checkpoint-path", 
        type=str, 
        help="Path to model checkpoint (default: from settings/env)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda"], 
        help="Device to run model on (default: from settings/env)"
    )
    
    return parser.parse_args()


def main():
    """Run the API server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("ethics_model.api")
    
    # Load settings
    settings = get_settings()
    
    # Parse command line arguments
    args = parse_args()
    
    # Override settings with command line arguments
    if args.host:
        os.environ["ETHICS_API_HOST"] = args.host
    if args.port:
        os.environ["ETHICS_API_PORT"] = str(args.port)
    if args.debug:
        os.environ["ETHICS_API_DEBUG"] = "True"
    if args.model_path:
        os.environ["ETHICS_API_MODEL_PATH"] = args.model_path
    if args.checkpoint_path:
        os.environ["ETHICS_API_CHECKPOINT_PATH"] = args.checkpoint_path
    if args.device:
        os.environ["ETHICS_API_DEVICE"] = args.device
    
    # Reload settings if environment variables were changed
    if any([args.host, args.port, args.debug, args.model_path, args.checkpoint_path, args.device]):
        # Clear the lru_cache to reload settings
        get_settings.cache_clear()
        settings = get_settings()
    
    # Log configuration
    logger.info(f"Starting Ethics Model API server on {settings.host}:{settings.port}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Device: {settings.device}")
    if settings.checkpoint_path:
        logger.info(f"Using checkpoint: {settings.checkpoint_path}")
    
    # Run the API server
    uvicorn.run(
        "ethics_model.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )


if __name__ == "__main__":
    main()
