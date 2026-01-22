# Model Architecture

This document describes the architecture of the Ethics Model, a modern PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding.

## Overview

The Ethics Model is designed with a modular architecture that combines:

1. **LLM Embeddings** via Huggingface Transformers
2. **Graph Neural Networks** (GNNs) via torch-geometric
3. **Semantic Hypergraphs** via GraphBrain
4. **Attention Mechanisms** for interpretability
5. **Uncertainty Quantification** for confidence estimation

## Core Architecture

### EnhancedEthicsModel

The main model class in `src/ethics_model/model.py` combines multiple components:

```
Input Text
    ↓
[LLM Embeddings] (BERT, GPT-2, etc.)
    ↓
[Transformer Encoder Layers]
    ├── Multi-Head Self-Attention
    ├── Feed-Forward Networks
    └── Modern Activations (GELU, ReCA)
    ↓
[Graph Reasoning Module] (Optional)
    ├── Ethical Relation Extraction
    ├── Graph Neural Networks
    └── GraphBrain Hypergraphs
    ↓
[Output Heads]
    ├── Ethics Classification
    ├── Manipulation Detection
    ├── Narrative Framing
    └── Moral Foundation Analysis
```

### Key Components

#### 1. LLM Integration

The model accepts embeddings from any Huggingface Transformer:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
llm = AutoModel.from_pretrained("bert-base-uncased")

# Generate embeddings
inputs = tokenizer(text, return_tensors="pt")
embeddings = llm(**inputs).last_hidden_state
```

Supported models:
- BERT (all variants)
- GPT-2
- RoBERTa
- DistilBERT
- ALBERT
- Any other Huggingface Transformer

#### 2. Transformer Encoder

Custom transformer encoder with:

```python
class EthicsTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="gelu"
    ):
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
```

Modern activation functions:
- **GELU** (Gaussian Error Linear Unit)
- **ReCA** (Rectified Cubic Activation)
- **Swish**
- **ReLU** (traditional)

#### 3. Graph Reasoning Module

The graph reasoning component extracts and processes ethical relationships:

**Ethical Relation Extraction:**
```python
class EthicalRelationExtractor:
    def extract(self, text):
        return {
            "actors": ["person", "organization"],
            "actions": ["helping", "manipulating"],
            "values": ["honesty", "profit"],
            "relations": [("actor", "performs", "action")]
        }
```

**Graph Neural Network:**
```python
class GraphReasoningEthicsModel(nn.Module):
    def __init__(self):
        self.gnn_layers = [
            GCNConv(in_channels, hidden_channels),
            GATConv(hidden_channels, hidden_channels),
            GraphConv(hidden_channels, out_channels)
        ]
```

**GraphBrain Integration:**
```python
from graphbrain import hgraph

# Parse text into semantic hypergraph
parser = hgraph.Parser(lang='en')
hypergraph = parser.parse(text)

# Extract semantic relations
relations = hypergraph.all()
```

#### 4. Attention Mechanism

Multi-head self-attention for interpretability:

```python
class MultiHeadAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
```

Attention weights can be visualized for explainability.

#### 5. Output Heads

Multiple task-specific output heads:

```python
class EthicsModel(nn.Module):
    def __init__(self, d_model):
        # Ethics classification head
        self.ethics_head = nn.Linear(d_model, 1)
        
        # Manipulation detection head
        self.manipulation_head = nn.Linear(d_model, 1)
        
        # Narrative framing head
        self.narrative_head = nn.Linear(d_model, num_frames)
        
        # Moral foundation head
        self.foundation_head = nn.Linear(d_model, num_foundations)
    
    def forward(self, embeddings):
        return {
            'ethics_score': torch.sigmoid(self.ethics_head(embeddings)),
            'manipulation_score': torch.sigmoid(self.manipulation_head(embeddings)),
            'narrative_probs': F.softmax(self.narrative_head(embeddings), dim=-1),
            'foundation_probs': F.softmax(self.foundation_head(embeddings), dim=-1)
        }
```

## Advanced Features

### 1. Explainability

**EthicsExplainer** (in `src/ethics_model/explainability.py`):

```python
from ethics_model.explainability import EthicsExplainer, AttentionVisualizer

explainer = EthicsExplainer(model)

# Get token attributions
attributions = explainer.attribute_tokens(text, target_output="ethics_score")

# Generate natural language explanation
explanation = explainer.generate_explanation(text, predictions)

# Visualize attention patterns
visualizer = AttentionVisualizer()
fig = visualizer.visualize_attention(text, attention_weights)
```

Features:
- Token-level attribution analysis
- Attention weight visualization
- Natural language explanations
- Counterfactual analysis

### 2. Uncertainty Quantification

**UncertaintyEthicsModel** (in `src/ethics_model/uncertainty.py`):

```python
from ethics_model.uncertainty import UncertaintyEthicsModel

model = UncertaintyEthicsModel(base_model)

# Get predictions with uncertainty
predictions = model(text, compute_uncertainty=True)

print(f"Ethics Score: {predictions['ethics_score']:.3f} ± {predictions['uncertainty']:.3f}")
print(f"Confidence: {predictions['confidence']:.3f}")
```

Methods:
- **Monte Carlo Dropout**: Sample predictions with dropout enabled
- **Evidential Deep Learning**: Quantify epistemic and aleatoric uncertainty
- **Uncertainty Calibration**: Ensure uncertainty estimates are well-calibrated

### 3. Graph-Based Reasoning

**GraphReasoningEthicsModel** (in `src/ethics_model/graph_reasoning.py`):

```python
from ethics_model.graph_reasoning import GraphReasoningEthicsModel

model = GraphReasoningEthicsModel(
    use_gnn=True,
    use_graphbrain=True,
    gnn_layers=3
)

# Forward pass with graph reasoning
outputs = model(
    embeddings=embeddings,
    texts=texts,
    graph_data=graph_data
)
```

Features:
- Automatic ethical relation extraction
- Graph neural network processing
- Semantic hypergraph integration
- Interactive graph visualization

## Training Architecture

### CUDA-Optimized Training

**CUDAGraphTrainer** (in `src/ethics_model/cuda_training.py`):

```python
from ethics_model.cuda_training import train_ethics_model

model = train_ethics_model(
    model=model,
    llm=llm,
    train_dataloader=train_loader,
    optimizer=optimizer,
    device=device,
    use_amp=True,  # Automatic Mixed Precision
    use_cuda_graphs=True  # CUDA Graphs optimization
)
```

Optimizations:
- **CUDA Graphs**: Record and replay computation graphs
- **Automatic Mixed Precision (AMP)**: FP16 for faster training
- **CUDA Streams**: Overlap computation and data transfer
- **Gradient Accumulation**: Simulate larger batch sizes

### Multi-Task Learning

The model supports multi-task learning with shared representations:

```python
# Multi-task loss
loss = (
    ethics_weight * ethics_loss +
    manipulation_weight * manipulation_loss +
    narrative_weight * narrative_loss +
    foundation_weight * foundation_loss
)
```

## Model Configuration

Example configuration:

```python
config = {
    'input_dim': 768,  # LLM hidden size
    'd_model': 512,  # Model dimension
    'n_layers': 6,  # Number of transformer layers
    'n_heads': 8,  # Number of attention heads
    'd_ff': 2048,  # Feed-forward dimension
    'dropout': 0.1,  # Dropout rate
    'activation': 'gelu',  # Activation function
    'max_seq_length': 512,  # Maximum sequence length
    'vocab_size': 50257,  # Vocabulary size
    'use_gnn': True,  # Enable graph neural networks
    'use_graphbrain': True,  # Enable GraphBrain
    'gnn_layers': 3,  # Number of GNN layers
    'num_moral_foundations': 5,  # Number of moral foundations
    'num_narrative_frames': 10  # Number of narrative frames
}
```

## Model Variants

### 1. Base Model

Simple transformer-based model without graph reasoning:

```python
model = EnhancedEthicsModel(
    use_gnn=False,
    use_graphbrain=False
)
```

### 2. Graph-Enhanced Model

Includes GNN but not GraphBrain:

```python
model = EnhancedEthicsModel(
    use_gnn=True,
    use_graphbrain=False
)
```

### 3. Full Model

All features enabled:

```python
model = EnhancedEthicsModel(
    use_gnn=True,
    use_graphbrain=True
)
```

### 4. Uncertainty Model

Base model with uncertainty quantification:

```python
from ethics_model.uncertainty import UncertaintyEthicsModel

base_model = EnhancedEthicsModel()
model = UncertaintyEthicsModel(base_model)
```

## Performance Characteristics

### Model Size

| Variant | Parameters | Memory (GPU) |
|---------|-----------|--------------|
| Base | ~50M | ~200 MB |
| With GNN | ~75M | ~300 MB |
| Full Model | ~100M | ~400 MB |

### Inference Speed

On NVIDIA RTX 3090:
- Base model: ~100 texts/second
- With GNN: ~60 texts/second
- Full model: ~40 texts/second

### Training Time

On NVIDIA RTX 3090, ETHICS dataset:
- Base model: ~2 hours (10 epochs)
- Full model: ~4 hours (10 epochs)

## Extensibility

The modular design allows easy extension:

### Adding Custom Layers

Create custom layers in `src/ethics_model/modules/`:

```python
# src/ethics_model/modules/custom_layer.py
import torch.nn as nn

class CustomEthicsLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.layer(x)
```

### Adding New Output Heads

Extend the model with new task-specific heads:

```python
class ExtendedEthicsModel(EnhancedEthicsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom output head
        self.custom_head = nn.Linear(self.d_model, num_classes)
    
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        
        # Add custom prediction
        outputs['custom_score'] = self.custom_head(outputs['embeddings'])
        
        return outputs
```

### Custom Loss Functions

Implement custom loss functions:

```python
class CustomEthicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ethics_loss = nn.BCELoss()
        self.custom_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        ethics_loss = self.ethics_loss(predictions['ethics_score'], targets['ethics'])
        custom_loss = self.custom_loss(predictions['custom_score'], targets['custom'])
        
        return ethics_loss + 0.5 * custom_loss
```

## Technical Details

### Attention Mechanism

The model uses scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Feed-Forward Network

Each transformer layer includes a position-wise feed-forward network:

$$\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$$

### Layer Normalization

Pre-norm architecture for better gradient flow:

$$\text{Output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

## Best Practices

1. **Start with the base model** for faster iteration
2. **Add graph reasoning** if you have relational data
3. **Enable uncertainty quantification** for high-stakes applications
4. **Use explainability features** to understand predictions
5. **Fine-tune on your domain** for best performance
6. **Monitor GPU memory** when enabling all features
7. **Use batch processing** for efficient inference

## References

- Hendrycks et al. (2021). "Aligning AI With Shared Human Values"
- Vaswani et al. (2017). "Attention is All You Need"
- Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
- Sensoy et al. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty"

## Next Steps

- See [Training Guide](./TRAINING.md) for training details
- See [API Documentation](./API.md) for deployment
- See [Evaluation Guide](./EVALUATION.md) for testing
