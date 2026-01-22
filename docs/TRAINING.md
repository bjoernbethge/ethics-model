# Training Guide for Ethics Model

This guide covers how to train the Ethics Model on your own data or the ETHICS dataset.

## Prerequisites

Before training, ensure you have:

1. Installed all dependencies with `uv sync --extra full`
2. Set up CUDA support (if using GPU): Run `bash setup.sh` (Linux) or `./setup.ps1` (Windows)
3. Installed SpaCy language model: `python -m spacy download en_core_web_sm`
4. Verified CUDA availability: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

## Training on ETHICS Dataset

The ETHICS dataset from Hendrycks et al. (2021) contains five domains of ethical reasoning:
- Justice
- Deontology
- Virtue Ethics
- Utilitarianism
- Commonsense Morality

### Download the Dataset

First, download the ETHICS dataset from [Hugging Face](https://huggingface.co/datasets/hendrycks/ethics) or the original source.

### Basic Training

Train the model using the provided training script:

```bash
python examples/train_on_ethics_dataset.py \
  --data_dir path/to/ethics_dataset \
  --llm_model bert-base-uncased \
  --batch_size 32 \
  --epochs 10 \
  --output_dir ./checkpoints
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Directory containing the ETHICS dataset |
| `--llm_model` | bert-base-uncased | Pretrained language model to use |
| `--batch_size` | 32 | Batch size for training |
| `--epochs` | 10 | Number of training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--d_model` | 512 | Model dimension |
| `--n_layers` | 6 | Number of transformer layers |
| `--n_heads` | 8 | Number of attention heads |
| `--max_length` | 128 | Maximum sequence length |
| `--output_dir` | ./checkpoints | Directory to save checkpoints |

### Advanced Options

**CUDA Optimizations:**
- `--disable_amp`: Disable automatic mixed precision training
- `--disable_cuda_graphs`: Disable CUDA Graphs optimization
- `--num_workers`: Number of dataloader workers (default: 4)

**GraphBrain Options:**
- `--disable_graphbrain`: Disable GraphBrain semantic hypergraphs
- `--parser_lang`: Language for GraphBrain parser (default: en)

**Other Options:**
- `--domains`: Comma-separated list of specific domains to train on
- `--cpu`: Force CPU usage
- `--patience`: Patience for early stopping (default: 2)
- `--grad_clip`: Gradient clipping value (default: 1.0)

### Example: Training on Specific Domains

```bash
python examples/train_on_ethics_dataset.py \
  --data_dir path/to/ethics_dataset \
  --domains justice,deontology \
  --llm_model bert-base-uncased \
  --batch_size 16 \
  --epochs 20
```

## Training with Custom Data

For training on your own data, you can use the training API or create a custom training script.

### Using the Training API

Start the API server and use the training endpoint:

```python
from ethics_model.api.client import EthicsModelClient

client = EthicsModelClient(base_url="http://localhost:8000")

# Prepare your training data
train_texts = [
    "Helping others in need is the right thing to do.",
    "Manipulating people for personal gain is unethical.",
    "Honesty should be valued above all else."
]
ethics_labels = [0.9, 0.1, 0.85]
manipulation_labels = [0.1, 0.9, 0.15]

# Start training
task_id = client.train(
    train_texts=train_texts,
    ethics_labels=ethics_labels,
    manipulation_labels=manipulation_labels,
    epochs=5,
    batch_size=8,
    learning_rate=1e-4
)

# Check training status
status = client.get_training_status(task_id)
print(f"Training Status: {status['status']}, Progress: {status['progress']}")
```

### Using the Training Script with LLM

For fine-tuning with a language model:

```bash
python examples/train_with_llm.py \
  --model_name bert-base-uncased \
  --train_file your_training_data.csv \
  --batch_size 16 \
  --epochs 10 \
  --output_dir ./checkpoints
```

## Monitoring Training

### TensorBoard

Training logs are automatically saved to TensorBoard. View them with:

```bash
tensorboard --logdir ./checkpoints/ethics_model_*/logs
```

Then open http://localhost:6006 in your browser.

### Training Metrics

The training script tracks:
- Training loss
- Validation loss
- Validation accuracy
- Ethics score metrics
- Manipulation detection metrics
- Uncertainty metrics (if enabled)

## Checkpoints

Checkpoints are saved to the output directory:
- `best_model.pt`: Best model based on validation performance
- `final_model.pt`: Final model after all epochs
- `logs/`: TensorBoard logs

### Loading a Checkpoint

```python
import torch
from ethics_model.model import create_ethics_model

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
config = checkpoint['config']

# Create model
model = create_ethics_model(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Training Best Practices

1. **Start with a smaller learning rate** (1e-5 to 1e-4) for fine-tuning
2. **Use gradient clipping** to prevent exploding gradients
3. **Monitor validation loss** for early stopping
4. **Use mixed precision training** (AMP) for faster training on GPU
5. **Batch size**: Start with 16-32 and adjust based on GPU memory
6. **Save checkpoints frequently** to avoid losing progress

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
- Reduce batch size: `--batch_size 8` or `--batch_size 4`
- Reduce max sequence length: `--max_length 64`
- Disable GraphBrain: `--disable_graphbrain`
- Use gradient accumulation in your custom training loop

### Slow Training

To speed up training:
- Enable CUDA Graphs (enabled by default)
- Enable AMP (enabled by default)
- Increase batch size if memory allows
- Increase number of workers: `--num_workers 8`
- Use a smaller language model

### GraphBrain Issues

If GraphBrain installation fails:
- Use `--disable_graphbrain` flag
- The model will still work without semantic hypergraph features

## Advanced Training Options

### Multi-Task Learning

The model supports multi-task learning for:
- Ethics classification
- Manipulation detection
- Narrative framing
- Moral foundation analysis

### Transfer Learning

You can fine-tune a pre-trained Ethics Model:

```python
import torch
from ethics_model.model import create_ethics_model

# Load pre-trained model
checkpoint = torch.load('pretrained_model.pt')
model = create_ethics_model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune on your data
# ... (continue with your training loop)
```

### Distributed Training

For multi-GPU training, wrap the model with DataParallel:

```python
import torch.nn as nn

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## Next Steps

After training:
1. Evaluate the model on test data
2. Generate visualizations with `examples/enhanced_model_showcase.py`
3. Deploy the model using the FastAPI server
4. Create explainability visualizations
5. Quantify uncertainty in predictions

For more information, see:
- [API Documentation](./API.md)
- [Model Architecture](./ARCHITECTURE.md)
- [Evaluation Guide](./EVALUATION.md)
