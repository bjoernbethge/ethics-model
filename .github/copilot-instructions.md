# GitHub Copilot Instructions for EthicsModel

## Project Overview

EthicsModel is a modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. It integrates LLM embeddings (via Huggingface Transformers), GraphBrain semantic hypergraphs, and Graph Neural Networks (GNNs) for advanced ethical reasoning. The framework features explainability, uncertainty quantification, and graph-based reasoning with comprehensive visualizations.

## Tech Stack

- **Core Framework**: PyTorch 2.7.0 with CUDA 12.8 support
- **Python Version**: 3.11+
- **Package Manager**: uv (modern Python package manager)
- **Key Libraries**:
  - `torch-geometric` for Graph Neural Networks
  - `graphbrain` for semantic hypergraph analysis
  - `spacy` for NLP preprocessing (en_core_web_sm model)
  - `transformers` for LLM integration (Huggingface)
  - `plotly` for interactive visualizations
  - `polars` for fast data processing
  - `bitsandbytes` for quantization
  - `pytest` for testing
  - `tensorboard` for training visualization

## Development Setup

### Installation

1. **Sync dependencies**: `uv sync --extra full`
2. **CUDA setup**: Run `bash setup.sh` (Linux) or `./setup.ps1` (Windows)
3. **SpaCy model**: `python -m spacy download en_core_web_sm`
4. **Verify CUDA**: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

### Testing

- Run all tests: `pytest tests/`
- Run specific test file: `pytest tests/test_ethics.py`
- Tests are located in the `tests/` directory
- Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`

### Project Structure

```
src/ethics_model/
├── model.py                 # Core EnhancedEthicsModel
├── explainability.py        # EthicsExplainer, AttentionVisualizer
├── uncertainty.py           # UncertaintyEthicsModel, UncertaintyVisualizer
├── graph_reasoning.py       # GraphReasoningEthicsModel, EthicalRelationExtractor
├── ethics_dataset.py        # ETHICS dataset integration with Polars
├── cuda_training.py         # CUDAGraphTrainer for optimized training
├── training.py              # Training utilities
├── data_preprocessing.py    # Data preprocessing utilities
└── api/                     # API endpoints and services
    ├── app.py              # Main FastAPI app
    ├── app_training.py     # Training API
    └── app_visualization.py # Visualization API
```

## Code Style and Conventions

### General Guidelines

- **Type annotations**: Always use type hints for function parameters and return values
- **Docstrings**: Use clear, descriptive docstrings for modules, classes, and functions
- **Imports**: Organize imports in standard order (stdlib, third-party, local)
- **Modularity**: Keep components modular and reusable
- **Error handling**: Use appropriate exception handling with informative error messages

### PyTorch Conventions

- Use `nn.Module` subclasses for all model components
- Place custom layers and architectures in `src/ethics_model/modules/` (if extending)
- Use modern activation functions (GELU, ReCA) consistently
- Leverage CUDA optimization features (CUDA Graphs, Streams) when available
- Always handle both CPU and CUDA device placement gracefully

### Testing Conventions

- Test files are prefixed with `test_`
- Use `pytest` fixtures defined in `tests/conftest.py`
- Mock external dependencies and heavy computations
- Test both CPU and CUDA code paths when applicable
- Ensure tests are fast and focused

### Documentation

- Update README.md when adding major features
- Document API changes in docstrings
- Include usage examples for new functionality
- Keep PYPI_SETUP.md updated for packaging changes

## Dependencies and Packages

### Adding New Dependencies

- Add to `pyproject.toml` under appropriate section:
  - `dependencies`: Core runtime dependencies
  - `[project.optional-dependencies.train]`: Training dependencies
  - `[project.optional-dependencies.dev]`: Development dependencies
  - `[project.optional-dependencies.full]`: All dependencies
- Run `uv sync` to update `uv.lock`
- For CUDA packages, document manual installation steps in README.md

### CUDA and PyTorch

- **DO NOT** add torch to pyproject.toml dependencies directly
- CUDA-enabled PyTorch must be installed manually after uv sync
- Document specific PyTorch wheel URLs for different CUDA versions
- PyTorch Geometric extensions (torch-scatter, torch-sparse) require manual wheel installation

## Architecture Principles

### Modular Design

- Core model components are organized in modules
- Each module has a single, well-defined responsibility
- Use composition over inheritance where possible
- Keep interfaces clean and minimal

### Explainability First

- All model predictions should be explainable
- Provide attention visualization capabilities
- Support token attribution analysis
- Generate natural language explanations where appropriate

### Uncertainty Awareness

- Quantify prediction uncertainty using Monte Carlo Dropout or Evidential Deep Learning
- Calibrate uncertainty estimates
- Support decision thresholds based on uncertainty
- Identify cases requiring human review

### Graph-Based Reasoning

- Extract ethical relations as graphs
- Use GNNs for relational reasoning
- Support semantic hypergraph analysis via GraphBrain
- Enable interactive graph visualization

## Common Tasks

### Training a Model

```python
from ethics_model.model import EnhancedEthicsModel
from ethics_model.training import train_model

# Initialize model with LLM backbone
model = EnhancedEthicsModel(llm_model="bert-base-uncased")

# Train on ETHICS dataset
train_model(model, data_dir="path/to/ethics_dataset", epochs=10)
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific category
pytest tests/test_ethics.py

# With coverage
pytest tests/ --cov=src/ethics_model
```

### Using Docker

- Development container available via `docker-compose.yml`
- Service name: `ethics-model-dev`
- Includes all dependencies and CUDA setup
- Code mounted for live updates

## Known Limitations

- PyTorch Geometric CUDA wheels are not on PyPI; require manual installation
- bitsandbytes requires platform-specific installation (Linux vs Windows)
- SpaCy language model must be downloaded separately
- CUDA 12.8 is required for optimal performance
- Some tests may require GPU access

## File Exclusions

When making changes, avoid committing:
- `__pycache__/` directories
- `*.pyc`, `*.pyo` files
- `checkpoints/*` (model checkpoints)
- `outputs/*` (generated outputs)
- Virtual environment directories (`venv/`, `.env/`, etc.)
- IDE-specific files (`.idea/`, `.vscode/`)
- Log files and tensorboard logs
- `.cache/`, `.ruff_cache/`

These are managed via `.gitignore`.

## CI/CD

- **Jekyll Pages**: Deploys documentation on push to main
- **PyPI Publishing**: Publishes package on release creation
- No automated testing workflow currently configured

## Contributing Guidelines

When adding new features:
1. Create tests first (TDD approach preferred)
2. Ensure backward compatibility
3. Update documentation
4. Run full test suite before committing
5. Keep changes minimal and focused
6. Use descriptive commit messages
