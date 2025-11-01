<!-- 0a5da79b-a391-482c-98de-3f99256bc2c7 a2104f72-a7ee-4abc-82a8-398f72fa4ffb -->
# Modern PyTorch 2.9 Refactoring - Windows Compatible

Complete refactoring to PyTorch 2.9 and Torch-Geometric 2.7 best practices. Remove all legacy code, implement modern architectures, update tests and examples.

## 1. Core Architecture Updates

### activation.py

- Add `SwiGLU` activation class (better than GELU)
- Add `RMSNorm` class (faster than LayerNorm)
- Update `get_activation()` to support "swiglu" and "rmsnorm"
- Keep ReCA as is

### model.py

**Create Config Dataclass:**

```python
@dataclass
class EthicsModelConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2  # for GQA
    vocab_size: int = 30000
    max_seq_length: int = 2048
    activation: Literal["gelu", "swish", "swiglu", "mish", "reca"] = "swiglu"
    use_gnn: bool = False
    use_rope: bool = True
    dropout: float = 0.1
```

**Replace Positional Embeddings:**

- Remove `self.position_embedding = nn.Embedding(max_seq_length, d_model)` (line 57)
- Add `RotaryPositionEmbedding` class before `EthicsModel`
- Implement RoPE in forward pass

**Update Model Class:**

- Accept `EthicsModelConfig` instead of individual params
- Replace all `nn.LayerNorm` with `RMSNorm`
- Set default activation to "swiglu"
- Add TypedDict for outputs

**Update create_ethics_model():**

- Accept config dict or `EthicsModelConfig`
- Use `mode="max-autotune"` for torch.compile

## 2. Attention Module Updates

### attention.py

**EthicalAttention:**

- Replace manual attention computation with optimized version
- Replace `nn.LayerNorm` with `RMSNorm` (if exists)
- Keep symbolic constraints support

**GroupedQueryAttention (NEW):**

- Add new `GroupedQueryAttention` class for memory efficiency
- Use fewer K/V heads than Q heads
- Integrate into `EthicalAttention` with flag

**All Attention Classes:**

- Update to use SwiGLU where activation is used
- Replace LayerNorm with RMSNorm

## 3. Moral Framework Updates

### moral.py

- Replace `nn.LayerNorm` with `RMSNorm` in all classes
- Update activation to default "swiglu"
- Keep Torch-Geometric GNN layers as-is (already modern with EdgeIndex)

## 4. Narrative Module Updates

### narrative.py

- Replace LayerNorm with RMSNorm
- Update default activation to "swiglu"
- Keep GNN implementations (already using EdgeIndex)

## 5. Tests Updates

### test_ethics.py

- Update model instantiation to use `EthicsModelConfig`
- Adjust assertions for new output structure
- Keep symbolic_constraints tests

### test_attention.py, test_moral.py, test_narrative.py

- Update to use new config pattern
- Update activation tests to include "swiglu"
- Test RMSNorm behavior

## 6. Examples Updates

### train_with_llm.py

- Update model creation to use `EthicsModelConfig`
- Change activation from "gelu" to "swiglu"
- Update any LayerNorm references

### Other examples

- Update all model instantiations to new config pattern
- Ensure torch.compile is used properly

## Implementation Order

1. `activation.py` - Add SwiGLU and RMSNorm
2. `model.py` - Add Config dataclass and RoPE
3. `model.py` - Replace LayerNorm with RMSNorm, add RoPE forward logic
4. `attention.py` - Update all attention modules
5. `moral.py` - Update normalization and activation
6. `narrative.py` - Update normalization and activation
7. `tests/` - Update all test files
8. `examples/` - Update train_with_llm.py and others

### To-dos

- [ ] Add SwiGLU and RMSNorm classes to activation.py
- [ ] Create EthicsModelConfig dataclass with modern defaults
- [ ] Implement RotaryPositionEmbedding and integrate into EthicsModel
- [ ] Replace all LayerNorm with RMSNorm in model.py
- [ ] Update all attention modules with RMSNorm and SwiGLU
- [ ] Update moral.py with RMSNorm and SwiGLU defaults
- [ ] Update narrative.py with RMSNorm and SwiGLU defaults
- [ ] Update all tests to use EthicsModelConfig and new APIs
- [ ] Update train_with_llm.py and other examples with new config