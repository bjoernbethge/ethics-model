# EthicsModel

A modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. Supports integration with LLM embeddings (e.g., Huggingface Transformers) and Graph Neural Networks (GNNs, e.g., torch-geometric). Includes advanced attention mechanisms, moral framework modeling, and comprehensive unit tests.

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

**3. Test CUDA availability:**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**4. Run all tests:**
```bash
pytest tests/
```

---

## Features
- Modular architecture for ethical reasoning and manipulation detection
- LLM embedding support (e.g., GPT-2, Gemma, etc.)
- Optional GNN layers for graph-based text analysis (argumentation, actor networks, narrative graphs, value relations)
- Modern activation functions (GELU, ReCA, etc.)
- Multi-task loss, augmentation, and real dataset support
- Logging, checkpoints, and TensorBoard integration
- Fully type-annotated and tested (pytest)

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
  from ethics_model.modules.attention import EthicalAttention
  from ethics_model.model import EthicsModel
  ```

---

## PyTorch Geometric (GNN) and CUDA

**Note:**
For CUDA support with torch-geometric and its extensions (torch-scatter, torch-sparse, torch-cluster, torch-spline-conv), you must manually install the matching wheels. These are not available on PyPI or the PyTorch index. See the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

---

## Fast LLM Training Test

A minimal, fast-running test for LLM training is included in `tests/test_llm_training.py`. It uses a mini dataset and checks that training runs without errors.

Run the test:
```bash
pytest tests/test_llm_training.py
```
