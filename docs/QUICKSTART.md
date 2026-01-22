# Quick Start Guide

Get up and running with the Ethics Model in under 10 minutes!

## Prerequisites

- Python 3.11+ installed
- `uv` package manager (or standard pip)
- (Optional) CUDA-capable GPU for training

## Installation (5 minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/bjoernbethge/ethics-model.git
cd ethics-model
```

### 2. Install Dependencies

Using uv (recommended):
```bash
pip install uv
uv sync --extra full
```

Using pip (alternative):
```bash
pip install -r requirements.txt  # If available
```

### 3. Install PyTorch

For CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA (GPU):
```bash
# Linux
bash setup.sh

# Windows PowerShell
./setup.ps1
```

### 4. Install SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Generate Basic Checkpoint

```bash
python examples/create_basic_checkpoint.py
```

You should see:
```
✓ Checkpoint created successfully!
  Location: /path/to/checkpoints/basic_model.pt
  Size: 139.08 MB
```

## Usage Examples (5 minutes)

### Example 1: Using the API (Recommended)

Start the API server:
```bash
python -m ethics_model.api.run
```

Then open http://localhost:8000/docs in your browser to see the interactive API documentation.

Or use the Python client:
```python
from ethics_model.api.client import EthicsModelClient

# Initialize client
client = EthicsModelClient(base_url="http://localhost:8000")

# Analyze text
result = client.analyze(
    "Companies should prioritize profit above all else.",
    include_details=True
)

print(f"Ethics Score: {result['ethics_score']:.2f}")
print(f"Manipulation Score: {result['manipulation_score']:.2f}")
print(f"Dominant Framework: {result['dominant_framework']}")
```

### Example 2: Direct Model Usage

```python
import torch
from ethics_model.model import create_ethics_model

# Load checkpoint
checkpoint = torch.load('checkpoints/basic_model.pt', map_location='cpu')

# Create model
model = create_ethics_model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model loaded successfully!")
print(f"Model config: {checkpoint['config']}")
```

### Example 3: Using Example Scripts

```bash
# Run the showcase example
python examples/enhanced_model_showcase.py \
  --model_path ./checkpoints/basic_model.pt \
  --enhancement all \
  --output_dir ./outputs
```

## Next Steps

### To Train a Model

See the [Training Guide](./TRAINING.md) for detailed instructions:

```bash
python examples/train_on_ethics_dataset.py \
  --data_dir path/to/ethics_dataset \
  --llm_model bert-base-uncased \
  --batch_size 32 \
  --epochs 10 \
  --output_dir ./checkpoints
```

### To Deploy the API

See the [API Documentation](./API.md) for deployment instructions:

```bash
# With custom checkpoint
python -m ethics_model.api.run \
  --checkpoint-path ./checkpoints/basic_model.pt \
  --port 8000 \
  --host 0.0.0.0
```

### To Understand the Architecture

Read the [Architecture Guide](./ARCHITECTURE.md) to understand how the model works.

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Install PyTorch first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "No module named 'spacy'"

**Solution**: Install spacy and download the model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Issue: "GraphBrain compilation error"

**Solution**: GraphBrain has compatibility issues with Python 3.12+. Either:
1. Use Python 3.11, or
2. Disable GraphBrain features with `--disable_graphbrain` flag

### Issue: "CUDA not available"

**Solution**: 
- For CPU-only usage, this is expected and normal
- For GPU usage, ensure CUDA 12.8 is installed and run the setup script

### Issue: "Checkpoint file not found"

**Solution**: Generate it first:
```bash
python examples/create_basic_checkpoint.py
```

## Testing Your Installation

Run the test suite to verify everything works:

```bash
pytest tests/
```

You should see all tests passing (or skipped if GPU not available).

## Quick Reference

### Project Structure

```
ethics-model/
├── src/ethics_model/        # Core model code
│   ├── model.py             # Main model
│   ├── explainability.py    # Explainability features
│   ├── uncertainty.py       # Uncertainty quantification
│   ├── graph_reasoning.py   # Graph reasoning
│   └── api/                 # REST API
├── examples/                # Example scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
├── checkpoints/             # Model checkpoints (generated)
└── outputs/                 # Output files (generated)
```

### Key Commands

```bash
# Install dependencies
uv sync --extra full

# Generate checkpoint
python examples/create_basic_checkpoint.py

# Start API server
python -m ethics_model.api.run

# Run tests
pytest tests/

# Train model
python examples/train_on_ethics_dataset.py --data_dir path/to/data

# View TensorBoard logs
tensorboard --logdir ./checkpoints/
```

### Documentation Quick Links

- [Training Guide](./TRAINING.md) - How to train models
- [API Documentation](./API.md) - API reference
- [Architecture Guide](./ARCHITECTURE.md) - Technical details
- [Missing Frontend](./MISSING_FRONTEND.md) - UI roadmap
- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md) - Status and issues

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Examples**: See the `examples/` directory
- **Issues**: https://github.com/bjoernbethge/ethics-model/issues
- **README**: See the main [README.md](../README.md)

## What's Next?

Now that you have the Ethics Model running, you can:

1. **Analyze texts** using the API or Python client
2. **Train your own model** on custom data
3. **Explore the examples** to learn more features
4. **Read the documentation** for advanced usage
5. **Contribute** by creating issues or pull requests

## Contributing

Contributions are welcome! Areas that need help:

- [ ] Frontend/UI development (see [MISSING_FRONTEND.md](./MISSING_FRONTEND.md))
- [ ] Training production models
- [ ] Adding more examples
- [ ] Improving documentation
- [ ] Fixing bugs

See [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) for a list of open issues.

---

**Estimated time to complete**: 10 minutes  
**Difficulty level**: Beginner  
**Last updated**: January 2026

Enjoy using the Ethics Model! 🎉
