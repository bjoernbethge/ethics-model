# Dependencies and Package Management

## Package Manager: uv (STRICT REQUIREMENT)
**NEVER use pip, npm, or any other package manager. ONLY use `uv`.**

## Current Dependencies

### Core Dependencies
- `pytorch-symbolic` - Symbolic execution for PyTorch
- `torch` (2.6.0+cu126) - PyTorch with CUDA 12.6
- `torchvision` - PyTorch vision library
- `torchaudio` - PyTorch audio library
- `torch-geometric` - Graph Neural Networks

### Training Dependencies (`--extra train`)
- `accelerate` - Distributed training
- `datasets` - Huggingface datasets
- `hf-transfer` - Fast Huggingface downloads
- `huggingface-hub` - Huggingface model hub
- `nlpaug` - NLP data augmentation
- `optimum` - Optimization tools
- `peft` - Parameter-efficient fine-tuning
- `pyarrow` - Columnar data format
- `safetensors` - Safe tensor serialization
- `tensorboard` - Training visualization
- `torch-tb-profiler` - PyTorch profiling for TensorBoard
- `transformers` - Huggingface transformers
- `transformers-stream-generator` - Streaming generation

### Dev Dependencies (`--extra dev`)
- `pytest` - Testing framework
- `pytest-cov` - Coverage plugin
- `pytest-xdist` - Parallel test execution

## PyTorch Index Configuration
The project uses a custom PyTorch index for CUDA 12.8 builds:
```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
```

## UV Configuration
```toml
[tool.uv]
python-preference = "only-managed"
python-downloads = "automatic"
link-mode = "copy"  # Important for Windows
compile-bytecode = true
```

## CUDA Setup
After initial dependency sync, run:
```powershell
# Windows
./setup_cuda_win.ps1

# Linux
bash setup_cuda.sh
```

This installs:
1. `torch==2.6.0+cu126` from CUDA index
2. `bitsandbytes` Windows wheel

## PyTorch Geometric Extensions
**Note:** For CUDA support with torch-geometric extensions (torch-scatter, torch-sparse, torch-cluster, torch-spline-conv), manual installation of matching wheels may be required. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Adding Dependencies
```powershell
# Core dependency
uv add package-name

# Training dependency
uv add --extra train package-name

# Dev dependency
uv add --dev package-name

# Remove dependency
uv remove package-name
```
