"""
Test CUDA availability and device information.
"""
import logging
import torch

logger = logging.getLogger(__name__)


def test_cuda_available():
    """Test if CUDA is available and log device information."""
    assert torch.cuda.is_available(), "CUDA is not available!"
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Current device ID: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}") 