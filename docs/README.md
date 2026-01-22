# Documentation

Welcome to the Ethics Model documentation! This directory contains comprehensive guides for using, training, and deploying the Ethics Model.

## Quick Links

- **[Training Guide](./TRAINING.md)** - Learn how to train models on the ETHICS dataset or custom data
- **[API Documentation](./API.md)** - REST API reference and integration examples
- **[Model Architecture](./ARCHITECTURE.md)** - Technical details about the model architecture
- **[Evaluation Guide](./EVALUATION.md)** - How to evaluate and benchmark models (coming soon)

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bjoernbethge/ethics-model.git
cd ethics-model
```

2. Install dependencies:
```bash
uv sync --extra full
```

3. Set up CUDA (for GPU support):
```bash
# Linux
bash setup.sh

# Windows PowerShell
./setup.ps1
```

4. Install SpaCy language model:
```bash
python -m spacy download en_core_web_sm
```

### Quick Example

```python
from ethics_model.model import create_ethics_model
import torch

# Load a checkpoint
checkpoint = torch.load('checkpoints/basic_model.pt', map_location='cpu')

# Create model
model = create_ethics_model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use the model
# (See Training Guide for complete examples)
```

## Documentation Structure

### Core Documentation

- **[TRAINING.md](./TRAINING.md)** - Comprehensive training guide
  - Training on ETHICS dataset
  - Training with custom data
  - Training parameters and optimization
  - Monitoring and checkpoints
  - Troubleshooting

- **[API.md](./API.md)** - API documentation
  - Quick start guide
  - Endpoint reference
  - Python client examples
  - Configuration options
  - Deployment guidelines

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Model architecture details
  - Core architecture overview
  - Component descriptions
  - Advanced features
  - Model variants
  - Performance characteristics

### Additional Resources

- **[../README.md](../README.md)** - Main project README with installation and quick start
- **[../PYPI_SETUP.md](../PYPI_SETUP.md)** - PyPI packaging and distribution guide
- **[../checkpoints/README.md](../checkpoints/README.md)** - Checkpoint format and usage
- **[../examples/](../examples/)** - Example scripts and notebooks

## Features Overview

### 1. Ethical Text Analysis

Analyze texts for ethical content, moral frameworks, and manipulation techniques:

- Ethics classification
- Manipulation detection  
- Narrative framing analysis
- Moral foundation mapping

### 2. Advanced Reasoning

Leverage state-of-the-art techniques:

- LLM embeddings (BERT, GPT-2, etc.)
- Graph Neural Networks
- Semantic hypergraphs (GraphBrain)
- Multi-head attention

### 3. Explainability

Understand model decisions:

- Attention visualization
- Token attribution
- Natural language explanations
- Graph relationship visualization

### 4. Uncertainty Quantification

Know when to trust predictions:

- Monte Carlo Dropout
- Evidential Deep Learning
- Uncertainty calibration
- Confidence estimation

### 5. Production-Ready API

Deploy easily with FastAPI:

- RESTful endpoints
- Batch processing
- Asynchronous operations
- Training via API
- Interactive visualizations

## Use Cases

### Research

- Ethical AI alignment research
- Social science text analysis
- Moral psychology studies
- Manipulation detection research

### Applications

- Content moderation
- Educational tools
- Fact-checking systems
- News analysis
- Social media monitoring

### Development

- Building ethical AI systems
- Integrating ethical reasoning
- Creating explainable AI
- Researching moral frameworks

## Project Status

### ✅ Complete

- [x] Core model architecture
- [x] LLM integration
- [x] Graph reasoning components
- [x] Explainability features
- [x] Uncertainty quantification
- [x] Training scripts
- [x] REST API
- [x] Python client
- [x] Docker support
- [x] Comprehensive documentation
- [x] Basic checkpoint for testing

### 🚧 In Progress

- [ ] Frontend/UI (See [MISSING_FRONTEND.md](./MISSING_FRONTEND.md))
- [ ] Evaluation guide
- [ ] Production-trained checkpoints
- [ ] Additional examples

### 📋 Planned

- [ ] Model zoo with pre-trained variants
- [ ] CLI tool for analysis
- [ ] Web demo
- [ ] Integration examples
- [ ] Performance benchmarks
- [ ] Deployment templates

## Contributing

Contributions are welcome! Please see the main README for contribution guidelines.

### Documentation Contributions

To improve documentation:

1. Create or update markdown files in this directory
2. Follow the existing structure and style
3. Include code examples where appropriate
4. Test all code examples
5. Submit a pull request

## Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas
- **Documentation**: Check this directory first

## Additional Notes

### Prerequisites

- Python 3.11+
- PyTorch 2.7.0+
- CUDA 12.8 (optional, for GPU support)
- 8GB+ RAM (16GB+ recommended for training)

### Dependencies

Main dependencies:
- `torch` - PyTorch deep learning framework
- `transformers` - Huggingface transformers for LLM
- `torch-geometric` - Graph neural networks
- `graphbrain` - Semantic hypergraphs (optional)
- `spacy` - NLP preprocessing
- `plotly` - Interactive visualizations
- `polars` - Fast data processing
- `fastapi` - REST API framework

See [pyproject.toml](../pyproject.toml) for complete dependency list.

### License

MIT License - See [LICENSE](../LICENSE) for details.

### Citation

If you use this project in your research, please cite:

```bibtex
@software{ethics_model,
  author = {Björn Bethge},
  title = {EthicsModel: A PyTorch Framework for Ethical Text Analysis},
  year = {2024},
  url = {https://github.com/bjoernbethge/ethics-model}
}
```

## Acknowledgments

This project builds upon:
- The ETHICS dataset by Hendrycks et al. (2021)
- Huggingface Transformers
- PyTorch and PyTorch Geometric
- GraphBrain semantic hypergraph library
- The broader open-source AI community

---

**Last Updated**: January 2026

For the latest updates, check the [main README](../README.md) or visit the [GitHub repository](https://github.com/bjoernbethge/ethics-model).
