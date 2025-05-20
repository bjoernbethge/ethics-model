"""
Test the refactored explainability module with NetworkX and spaCy.
"""
import pytest
import torch
import networkx as nx
from ethics_model.explainability import GraphExplainer, AttentionVisualizer, EthicsExplainer, create_explainer, quick_explain
from transformers import AutoTokenizer, AutoModel
import spacy

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

def test_attention_visualizer(tokenizer):
    viz = AttentionVisualizer(tokenizer)
    text = "This is a test"
    attention_weights = torch.randn(1, 4, 4)
    fig = viz.visualize_attention(text, attention_weights)
    assert hasattr(fig, 'to_html')

def test_graph_explainer_build_graph(nlp):
    explainer = GraphExplainer("en_core_web_sm")
    text = "John helps people."
    graph = explainer.build_ethical_graph(text)
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0

def test_graph_explainer_extract_entities(nlp):
    explainer = GraphExplainer("en_core_web_sm")
    text = "John helps with fairness."
    entities = explainer.extract_ethical_entities(text)
    assert "actors" in entities
    assert "actions" in entities
    assert "values" in entities

def test_graph_explainer_visualize(nlp):
    explainer = GraphExplainer("en_core_web_sm")
    text = "John helps people."
    fig = explainer.visualize_ethical_graph(text)
    assert hasattr(fig, 'to_html')

def test_ethics_explainer(tokenizer):
    class DummyModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return {"ethics_score": torch.tensor([[0.8]]), "manipulation_score": torch.tensor([[0.2]])}
    class DummyLLM(torch.nn.Module):
        def forward(self, input_ids):
            class Output:
                def __init__(self):
                    self.last_hidden_state = torch.randn(1, input_ids.shape[1], 768)
            return Output()
    model = DummyModel()
    llm = DummyLLM()
    explainer = EthicsExplainer(model, tokenizer)
    tokens = tokenizer("Test text", return_tensors='pt', max_length=16, padding='max_length')
    outputs, attributions = explainer.compute_attributions("Test text", llm)
    assert "ethics_score" in outputs
    assert attributions.shape[0] == 1

def test_quick_explain(tokenizer):
    class DummyModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return {"ethics_score": torch.tensor([[0.8]]), "manipulation_score": torch.tensor([[0.2]])}
    model = DummyModel()
    explanation = quick_explain("Test text", model, model, tokenizer)
    assert "text" in explanation
    assert "ethics_score" in explanation
