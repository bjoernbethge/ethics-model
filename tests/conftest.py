"""
Shared test configuration and fixtures for the ethics model test suite.
"""
import pytest
from unittest.mock import MagicMock

import torch


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def summary_writer():
    """Mock TensorBoard SummaryWriter for testing."""
    writer = MagicMock()
    writer.add_scalar = MagicMock()
    writer.add_text = MagicMock()
    writer.add_histogram = MagicMock()
    writer.close = MagicMock()
    return writer


@pytest.fixture
def cpu_or_cuda_profiler():
    """Mock PyTorch profiler for testing."""
    profiler = MagicMock()
    
    # Mock key_averages method
    mock_table = MagicMock()
    mock_table.table = MagicMock(return_value="Mock profiler table")
    profiler.key_averages = MagicMock(return_value=mock_table)
    
    return profiler


@pytest.fixture
def symbolic_constraints():
    """Mock symbolic constraints function for testing."""
    def mock_constraints(result):
        """Mock constraint that returns result unchanged."""
        return result
    return mock_constraints