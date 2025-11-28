# EthicsModel

A modern, modular PyTorch framework for ethical text analysis, manipulation detection, and narrative understanding. Supports integration with LLM embeddings (e.g., Huggingface Transformers) and Graph Neural Networks (GNNs, e.g., torch-geometric). Includes advanced attention mechanisms, moral framework modeling, and comprehensive unit tests.

---

## Quick Installation

```bash
uv sync --extra full
```

That's it! This installs:
- PyTorch 2.8.0+ with CUDA 12.8
- PyTorch Geometric 2.7.0
- All training dependencies (transformers, peft, accelerate, etc.)
- GRetriever dependencies (pcst_fast, sentencepiece)
- Dev tools (pytest, profiling)

**Run tests:**
```bash
pytest tests/
```

---

## Features
- **Modern PyTorch 2.8+**: torch.compile() enabled by default (3-5× speedup)
- **PyTorch Geometric 2.7.0**: GATv2Conv, EdgeIndex optimization, variance-preserving aggregation
- **GRetriever Integration**: LLM + GNN knowledge graph reasoning with Qwen3-3B-Instruct
- Modular architecture for ethical reasoning and manipulation detection
- LLM embedding support (Transformers, PEFT, LoRA)
- GNN layers for graph-based text analysis (argumentation, actor networks, narrative graphs, value relations)
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
| `retriever.py`       | `EthicsModel`                  | See below                                                                            | Integrates all modules, orchestrates full analysis                                                |

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

All core model components (custom `nn.Module` classes, layers, blocks, architectures) are organized in the `src/ethics_model/modules/` submodule. The main GNN model is implemented in `src/ethics_model/modules/gnn.py`.

- **Extendability:** Add your own layers or architectures by creating new files in `modules/` and importing them.
- **Usage Example:**
  ```python
  from ethics_model import EthicsModel, EthicsGNN, EthicsGNNConfig, create_ethics_gnn
  
  # Create graph-native GNN
  config = EthicsGNNConfig(hidden_dim=256, num_layers=2, num_heads=4)
  gnn = create_ethics_gnn(config)
  
  # Or use GRetriever-based EthicsModel
  model = EthicsModel(gnn_hidden_dim=256, num_gnn_layers=2)
  ```

---

## Fast LLM Training Test

A minimal, fast-running test for LLM training is included in `tests/test_llm_training.py`. It uses a mini dataset and checks that training runs without errors.

Run the test:
```bash
pytest tests/test_llm_training.py
```
