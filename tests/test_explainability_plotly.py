"""
Test the refactored explainability module with Plotly visualizations.
"""
import pytest
import networkx as nx
from ethics_model.graph_reasoning import GraphVisualizer

@pytest.fixture
def toy_graph():
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.5)
    G.add_edge(1, 2, weight=0.8)
    G.add_node(3)
    return G

def test_visualize_ethical_graph_with_plotly(toy_graph):
    # Die Funktion soll mit echten Graphen laufen
    fig = GraphVisualizer.visualize_ethical_graph(toy_graph)
    assert fig is None or hasattr(fig, 'to_html')

if __name__ == "__main__":
    pytest.main([__file__])
