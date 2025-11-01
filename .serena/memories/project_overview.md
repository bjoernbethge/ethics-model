# EthicsModel Project Overview

## Purpose
EthicsModel is a modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. It integrates:
- LLM embeddings (Huggingface Transformers)
- Graph Neural Networks (torch-geometric)
- Advanced attention mechanisms
- Moral framework modeling
- Comprehensive ethical reasoning capabilities

## Key Features
- Modular architecture for ethical reasoning and manipulation detection
- LLM embedding support (GPT-2, Gemma, etc.)
- Optional GNN layers for graph-based text analysis (argumentation, actor networks, narrative graphs, value relations)
- Modern activation functions (GELU, ReCA, etc.)
- Multi-task loss, augmentation, and real dataset support
- Logging, checkpoints, and TensorBoard integration
- Fully type-annotated and tested (pytest)

## Tech Stack
- **Language:** Python 3.12+
- **Framework:** PyTorch 2.6.0 with CUDA 12.6
- **GNN:** torch-geometric
- **LLM:** Transformers, PEFT, Accelerate
- **Testing:** pytest, pytest-cov, pytest-xdist
- **Profiling:** TensorBoard, torch-tb-profiler, pytorch-symbolic
- **Package Manager:** uv (strict requirement)
- **Build System:** Hatchling

## Project Structure
```
ethics-model/
├── src/ethics_model/           # Main package
│   ├── modules/                # Core neural modules
│   │   ├── activation.py       # Activation functions (ReCA, etc.)
│   │   ├── attention.py        # Attention mechanisms
│   │   ├── moral.py            # Moral framework models
│   │   └── narrative.py        # Narrative analysis
│   ├── api/                    # API endpoints
│   ├── model.py                # Main EthicsModel class
│   ├── training.py             # Training utilities
│   └── data.py                 # Data handling
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── pyproject.toml              # Project configuration
├── setup_cuda_win.ps1          # Windows CUDA setup
└── setup_cuda.sh               # Linux CUDA setup
```

## Python Version
Python 3.12 (managed by uv)
