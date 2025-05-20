# EthicsModel

A modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. Uses NetworkX and spaCy for ethical relationship extraction and graph-based reasoning with comprehensive testing and real-world applications.

---

## 🚀 Quick Installation

**1. Sync dependencies:**
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

**3. Install spaCy language model:**
```bash
python -m spacy download en_core_web_sm
```

**4. Test installation:**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pytest tests/ -v
```

---

## ✨ Recent Improvements

### 🔄 **Refactored Architecture (v0.2.0)**
- **Replaced GraphBrain with NetworkX + spaCy** for better compatibility and maintenance
- **Unified testing suite** with real components (no mocks)
- **Enhanced data processing** with improved caching and error handling
- **Comprehensive graph reasoning** using proven NLP techniques
- **Better memory management** for large datasets

### 🧪 **Real Testing**
All tests now use actual components instead of mocks:
- **Data processing tests** with real tokenizers and datasets
- **Graph extraction tests** with actual spaCy processing
- **Model integration tests** with complete pipelines
- **End-to-end training tests** with real data flows

---

## 🏗️ Architecture Overview

```
EthicsModel/
├── src/ethics_model/
│   ├── data.py              # Enhanced data processing with NetworkX graphs
│   ├── model.py             # Core ethics model with modular design
│   ├── training.py          # Comprehensive training with validation
│   ├── graph_reasoning.py   # NetworkX + spaCy graph reasoning
│   └── modules/             # Modular components
│       ├── attention.py     # Ethical attention mechanisms
│       ├── moral.py         # Moral framework processing
│       ├── narrative.py     # Manipulation detection
│       └── activation.py    # Modern activation functions
├── tests/                   # Comprehensive test suite (no mocks)
├── examples/                # Real-world usage examples
└── README.md               # This file
```

---

## 🧠 Features

### **Core Capabilities**
- **Multi-task Learning**: Simultaneous ethics scoring and manipulation detection
- **LLM Integration**: Works with any Hugging Face transformer model
- **Graph Neural Networks**: Advanced relationship modeling with PyTorch Geometric
- **Real-time Analysis**: Fast inference with CUDA optimization
- **Comprehensive Testing**: 100+ tests with real components

### **Advanced Analysis**
- **NetworkX Graph Processing**: Ethical relationship extraction from text
- **spaCy NLP Pipeline**: Robust text processing and entity recognition  
- **Moral Framework Analysis**: Deontological, utilitarian, virtue ethics
- **Manipulation Detection**: Propaganda, framing, cognitive dissonance
- **Uncertainty Quantification**: Confidence estimation for predictions

### **Developer Experience**
- **Modular Design**: Easy to extend and customize
- **Type Safety**: Fully type-annotated codebase
- **Memory Efficient**: Optimized for large datasets
- **Well Documented**: Comprehensive examples and tutorials
- **Production Ready**: Robust error handling and logging

---

## 📊 Quick Start

### **Basic Usage**
```python
from src.ethics_model.data import MultiTaskDataset, collate_ethics_batch
from src.ethics_model.model import create_ethics_model
from torch.utils.data import DataLoader

# Prepare data
texts = ["John helped Mary", "Politicians sometimes mislead voters"]
ethics_scores = [0.9, 0.2]
manipulation_scores = [0.1, 0.8]

# Create dataset
dataset = MultiTaskDataset(
    texts=texts,
    ethics_labels=ethics_scores,
    manipulation_labels=manipulation_scores,
    tokenizer=your_tokenizer
)

# Create model
config = {
    'd_model': 512,
    'n_layers': 6,
    'vocab_size': 30000,
    'use_semantic_graphs': True
}
model = create_ethics_model(config)

# Train or evaluate
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_ethics_batch)
```

### **Graph-Enhanced Analysis**
```python
from src.ethics_model.data import GraphEthicsDataset
from src.ethics_model.graph_reasoning import extract_and_visualize

# Enhanced dataset with graph features
graph_dataset = GraphEthicsDataset(
    texts=texts,
    ethics_labels=ethics_scores,
    manipulation_labels=manipulation_scores,
    tokenizer=tokenizer,
    spacy_model="en_core_web_sm"
)

# Analyze ethical relationships
result = extract_and_visualize("The company exploited workers for profit")
print(f"Graph has {result['metrics']['n_nodes']} ethical entities")
```

---

## Module Output Overview

| Module | Main Components | Purpose |
|--------|----------------|---------|
| `model.py` | `EnhancedEthicsModel` | Core architecture with GraphBrain integration |
| `explainability.py` | `EthicsExplainer`, `AttentionVisualizer`, `GraphExplainer` | Explaining model decisions using NetworkX and spaCy |
| `uncertainty.py` | `UncertaintyEthicsModel`, `UncertaintyVisualizer` | Quantifying prediction uncertainty |
| `graph_reasoning.py` | `GraphReasoningEthicsModel`, `EthicalRelationExtractor` | Advanced ethical relationship reasoning using NetworkX and spaCy |
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