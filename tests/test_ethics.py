"""
DEPRECATED: This test used the old sequence-based EthicsModel architecture.
The model is now graph-native (EthicsGNN). See test_pyg_27_features.py for tests.
"""
import pytest

pytest.skip(
    "test_ethics.py is deprecated - old sequence-based model removed. "
    "Use test_pyg_27_features.py for EthicsGNN tests.",
    allow_module_level=True
) 