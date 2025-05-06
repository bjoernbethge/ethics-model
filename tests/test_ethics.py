import torch
from ethics_model.ethics import EthicsModel

def test_ethics_model_forward(summary_writer, cpu_or_cuda_profiler):
    model = EthicsModel(input_dim=16, d_model=16, n_layers=2, n_heads=2, vocab_size=100, max_seq_length=10, use_gnn=False)
    input_ids = torch.randint(0, 100, (2, 5))
    attention_mask = torch.ones(2, 5)
    outputs = model(input_ids, attention_mask)
    summary_writer.add_text('profiler/key_operators', str(cpu_or_cuda_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5)))
    summary_writer.add_scalar('dummy/ethics_score', float(outputs['ethics_score'][0]), 0)
    assert isinstance(outputs, dict)
    for key in [
        'ethics_score', 'manipulation_score', 'framework_analysis',
        'intuition_analysis', 'dual_process_analysis', 'narrative_analysis',
        'framing_analysis', 'dissonance_analysis', 'manipulation_analysis',
        'propaganda_analysis', 'attention_weights', 'hidden_states', 'meta_cognitive_features']:
        assert key in outputs 