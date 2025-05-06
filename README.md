# EthicsModel

A modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. Supports integration with LLM embeddings (e.g., Huggingface Transformers) and Graph Neural Networks (GNNs, e.g., torch-geometric). Includes advanced attention mechanisms, moral framework modeling, and comprehensive unit tests.

## Features
- Modular architecture for ethical reasoning and manipulation detection
- LLM embedding support (e.g., GPT-2, Gemma, etc.)
- Optional GNN layers for graph-based text analysis
- Modern activation functions (GELU, ReCA, etc.)
- Multi-task loss, augmentation, and real dataset support
- Logging, checkpoints, and TensorBoard integration
- Fully type-annotated and tested (pytest)

## Installation

Use **uv** for fast, modern dependency management directly from `pyproject.toml`.

### 1. Minimal installation (core dependencies)
Install only the core dependencies:
```bash
uv sync
```

### 2. Training dependencies (LLM, augmentation, etc.)
Recommended for training and experiments:
```bash
uv sync --extra train
```

### 3. Development & testing dependencies
For development and running tests:
```bash
uv sync --extra dev
```

### 4. Full installation (all features: training, dev, tests)
Install everything at once:
```bash
uv sync --extra full
```

---

### 5. bitsandbytes installation (required for quantization, LLM fine-tuning, etc.)

#### Windows
After syncing dependencies, install bitsandbytes manually:
```bash
uv pip install --force-reinstall "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-win_amd64.whl"
```

#### Linux
Official multi-backend wheel (recommended for CUDA, ROCm, Intel, Apple Silicon):
```bash
# Note, if you don't want to reinstall BNBs dependencies, append the `--no-deps` flag!
uv pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'
```

---

## Running Unit Tests

After installing dependencies (see above), run the tests as usual:
```bash
pytest tests/
```
