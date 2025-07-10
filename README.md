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
  - Install the correct CUDA-enabled PyTorch wheel (`torch==2.7.0+cu128`)
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

## Development with Docker (New!)

We've added Docker support to make development easier and more consistent across different environments.

**Using the Development Container with JetBrains IDEs:**

1. **Setup the development container:**
   - **Linux/macOS:**
     ```bash
     chmod +x setup_dev_container.sh
     ./setup_dev_container.sh
     ```
   - **Windows (PowerShell):**
     ```powershell
     .\setup_dev_container.ps1
     ```

2. **Configure JetBrains IDE (PyCharm/IntelliJ) to use the container:**
   - Open the project in your JetBrains IDE
   - Go to File > Settings > Project: ethics-model > Python Interpreter
   - Click on the gear icon and select 'Add'
   - Choose 'Docker Compose' from the left panel
   - Select the docker-compose.yml file in your project
   - Select the 'ethics-model-dev' service
   - Click 'OK' to add the interpreter

3. **Run the project inside the container:**
   - All Python runs, tests, and debugging will now use the containerized environment
   - The container includes all dependencies and proper CUDA setup
   - Code changes are reflected immediately due to volume mounting

**Benefits:**
- Consistent development environment for all contributors
- Proper CUDA and bitsandbytes setup
- All dependencies pre-installed
- Isolated environment without affecting your system Python

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

| Module (File)         | Main Classes/Functions         | Output Values (What)                                                                 | How (Computation)                                                                                 |
|---------------------- |-------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `activation.py`       | `get_activation`, `ReCA`      | Activated tensors for neural layers                                                  | Modern activation functions (ReLU, GELU, Swish, ReCA, etc.)                                       |
| `attention.py`        | `EthicalAttention`            | `output`, `attention_weights`                                                        | Multi-head attention, optionally with moral context                                               |
|                      | `MoralIntuitionAttention`      | `intuitive_response`, `emotional_intensity`, `foundation_activations`                | Moral foundation embeddings, intuition scorer, emotion amplifier                                  |
|                      | `NarrativeFrameAttention`      | `frame_scores`, `manipulation_scores`, `manipulative_segments`, `frame_transitions`  | Frame classifiers, GRU for transitions, manipulation detector                                     |
|                      | `DoubleProcessingAttention`    | `final_output`, `system_outputs` (system1/2 output, resolution weights)              | Dual-process (System 1/2) attention and conflict resolution                                       |
|                      | `GraphAttentionLayer`          | Activated graph features                                                             | GATConv (graph attention)                                                                         |
| `moral.py`           | `MoralFrameworkEmbedding`      | `combined`, `framework_outputs`                                                      | Projects input through multiple moral frameworks, weighted and combined                           |
|                      | `EthicalCrossDomainLayer`      | `domain_fused`                                                                      | Projects and fuses representations from different ethical domains                                 |
|                      | `MultiFrameworkProcessor`      | `framework_outputs`, `conflict_scores`, `consensus_output`, `combined_output`        | Detects conflicts/consensus between frameworks                                                    |
|                      | `EthicalPrincipleEncoder`      | `encoded_principles`                                                                | Encodes and models interactions between ethical principles                                        |
|                      | `MoralFrameworkGraphLayer`     | Activated graph features                                                             | GCNConv (graph convolution)                                                                       |
| `narrative.py`       | `FramingDetector`              | `frame_scores`, `framing_strength`, `consistency_score`                              | Detects framing types, strength, and consistency                                                  |
|                      | `CognitiveDissonanceLayer`     | `value_conflicts`, `dissonance_score`, `resolution_strategy`, `value_activations`    | Detects value conflicts, predicts dissonance resolution                                           |
|                      | `NarrativeManipulationDetector`| `technique_scores`, `aggregate_score`, `confidence`, `manipulation_map`              | Detects manipulation techniques, aggregates and calibrates scores                                 |
|                      | `PropagandaDetector`           | `technique_matches`, `intensity_score`, `credibility_score`, `inverse_credibility`   | Detects propaganda techniques, scores intensity and credibility                                   |
|                      | `NarrativeGraphLayer`          | Activated graph features                                                             | GCNConv for narrative graphs                                                                      |
| `ethics.py`          | `EthicsModel`                  | See below                                                                            | Integrates all modules, orchestrates full analysis                                                |

#### EthicsModel Output Dictionary
The main model (`EthicsModel`) returns a dictionary with the following keys:
- `ethics_score`: Overall ethics score
- `manipulation_score`: Manipulation risk score
- `framework_analysis`: Results from the moral framework processor
- `intuition_analysis`: Results from the intuition module
- `dual_process_analysis`: Results from dual-process (System 1/2) analysis
- `narrative_analysis`: Narrative framing and manipulation analysis
- `framing_analysis`: Detailed framing analysis
- `dissonance_analysis`: Cognitive dissonance analysis
- `manipulation_analysis`: Manipulation technique detection
- `propaganda_analysis`: Propaganda risk and credibility
- `attention_weights`, `hidden_states`, `meta_cognitive_features`: Internal model states

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