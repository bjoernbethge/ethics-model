# EthicsModel

A modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. Supports integration with LLM embeddings (e.g., Huggingface Transformers), GraphBrain semantic hypergraphs, and Graph Neural Networks (GNNs, e.g., torch-geometric). Features explainability, uncertainty quantification, and advanced graph-based reasoning with comprehensive visualizations using Plotly Express.

---

## Quick Installation

**1. Sync dependencies (core, training, dev, tests):**
```bash
uv sync --extra full
```

**2. CUDA & bitsandbytes setup:**
- **Linux:**
  ```bash
  bash setup_cuda.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  ./setup_cuda_win.ps1
  ```
  These scripts will:
  - Install the correct CUDA-enabled PyTorch wheel (`torch==2.6.0+cu126`)
  - Install the latest bitsandbytes wheel for your platform
  - Print `Done.` when finished

**3. Install SpaCy language model for GraphBrain:**
```bash
python -m spacy download en_core_web_sm
```

**4. Test CUDA availability:**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**5. Run all tests:**
```bash
pytest tests/
```

---

## Enhanced Features

### 1. Explainability
- **Attention Visualization**: Visualize attention patterns to understand model focus
- **Token Attribution**: Analyze contribution of individual tokens to ethical judgments
- **Graph Visualization**: Explore ethical relationships in semantic graphs
- **Natural Language Explanations**: Generate human-readable ethical analyses

### 2. Uncertainty Quantification
- **Monte Carlo Dropout**: Estimate prediction uncertainty through sampling
- **Evidential Deep Learning**: Quantify uncertainty through evidential reasoning
- **Uncertainty Calibration**: Ensure uncertainty estimates reliably reflect error rates
- **Decision Support**: Identify cases requiring human intervention based on uncertainty

### 3. Advanced Graph Reasoning
- **Ethical Relation Extraction**: Extract ethical concepts, actors, actions, and relationships
- **Graph Neural Networks**: Process ethical relationships using specialized GNNs
- **Moral Foundation Analysis**: Map ethical judgments to underlying moral foundations
- **Ethical Graph Visualization**: Interactive exploration of ethical relationship graphs

---

## Features
- Modular architecture for ethical reasoning and manipulation detection
- LLM embedding support (e.g., GPT-2, Gemma, etc.)
- GraphBrain integration for semantic hypergraph analysis
- Graph Neural Networks (GNNs) for relational reasoning
- Modern activation functions (GELU, ReCA, etc.)
- Multi-task loss, augmentation, and real dataset support
- Logging, checkpoints, and TensorBoard integration
- CUDA-optimized training with CUDA Graphs and CUDA Streams
- Robust explainability and visualization tools
- Fully type-annotated and tested (pytest)

---

## Training on ETHICS Dataset

Train the enhanced model on the ETHICS dataset:

```bash
python examples/train_on_ethics_dataset.py \
  --data_dir path/to/ethics_dataset \
  --llm_model bert-base-uncased \
  --batch_size 32 \
  --epochs 10 \
  --output_dir ./checkpoints
```

## Feature Showcase

Explore the enhanced model capabilities:

```bash
python examples/enhanced_model_showcase.py \
  --model_path ./checkpoints/model.pt \
  --enhancement all \
  --output_dir ./outputs
```

---

## Module Output Overview

| Module | Main Components | Purpose |
|--------|----------------|---------|
| `model.py` | `EnhancedEthicsModel` | Core architecture with GraphBrain integration |
| `explainability.py` | `EthicsExplainer`, `AttentionVisualizer` | Explaining model decisions |
| `uncertainty.py` | `UncertaintyEthicsModel`, `UncertaintyVisualizer` | Quantifying prediction uncertainty |
| `graph_reasoning.py` | `GraphReasoningEthicsModel`, `EthicalRelationExtractor` | Advanced ethical relationship reasoning |
| `ethics_dataset.py` | `ETHICSDataset`, `ETHICSMultiDomainDataset` | ETHICS dataset integration with Polars |
| `cuda_training.py` | `CUDAGraphTrainer` | Optimized training with CUDA Graphs and Streams |

---

## Modular Design

All core model components (custom `nn.Module` classes, layers, blocks, architectures) are organized in the `src/ethics_model/modules/` submodule. The main model is implemented in `src/ethics_model/model.py`.

- **Extendability:** Add your own layers or architectures by creating new files in `modules/` and importing them in your main model.
- **Usage Example:**
  ```python
  from ethics_model.model import EnhancedEthicsModel
  from ethics_model.explainability import EthicsExplainer
  ```

---

## PyTorch Geometric (GNN) and CUDA

**Note:**
For CUDA support with torch-geometric and its extensions (torch-scatter, torch-sparse, torch-cluster, torch-spline-conv), you must manually install the matching wheels. These are not available on PyPI or the PyTorch index. See the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

---

## Testing

Run the full test suite:

```bash
pytest tests/
```