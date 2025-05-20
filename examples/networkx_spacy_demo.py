#!/usr/bin/env python3
"""
Demo script showing the refactored NetworkX + spaCy based ethical analysis.

This example demonstrates how to use the refactored graph reasoning capabilities
that replace GraphBrain with NetworkX and spaCy for ethical relationship extraction.
"""

import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_ethical_relation_extraction():
    """Demonstrate ethical relation extraction with NetworkX and spaCy."""
    print("=== Ethical Relation Extraction Demo ===\n")
    
    try:
        from ethics_model.graph_reasoning import EthicalRelationExtractor, GraphVisualizer
        
        # Initialize extractor (will try to load spaCy model)
        print("Initializing Ethical Relation Extractor...")
        extractor = EthicalRelationExtractor()
        print("✓ Extractor initialized successfully\n")
        
        # Test text
        test_text = """
        The company should help their employees during difficult times.
        However, the management decided to prioritize profits over worker welfare.
        This decision may harm employee trust and damage the company's reputation.
        Ethical leadership requires balancing stakeholder interests fairly.
        """
        
        print(f"Analyzing text:\n{test_text}\n")
        
        # Extract relations
        print("Extracting ethical relations...")
        relations = extractor.extract_relations(test_text)
        
        print(f"✓ Extracted {relations['n_nodes']} nodes and {relations['n_edges']} edges")
        
        # Display entities found
        entities = relations['entities']
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"\n{entity_type.title()}:")
                for entity, subtype in entity_list:
                    print(f"  - {entity} ({subtype})")
        
        # Display relations
        if relations['relations']:
            print(f"\nRelations found:")
            for relation in relations['relations'][:5]:  # Show first 5
                print(f"  - {relation['source']} --{relation['relation']}--> {relation['target']}")
        
        # Convert to PyTorch Geometric format
        print("\nConverting to PyTorch Geometric format...")
        graph_data = extractor.to_pyg_data(relations)
        print(f"✓ Created PyG Data with {graph_data.num_nodes} nodes")
        print(f"  - Node features shape: {graph_data.x.shape}")
        print(f"  - Edge index shape: {graph_data.edge_index.shape}")
        
        # Analyze graph metrics
        graph = relations['graph']
        metrics = GraphVisualizer.analyze_graph_metrics(graph)
        print(f"\nGraph Metrics:")
        for key, value in metrics.items():
            if key != 'node_type_distribution':
                print(f"  - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in ethical relation extraction: {e}")
        return False


def demo_semantic_graph_processing():
    """Demonstrate semantic graph processing."""
    print("\n=== Semantic Graph Processing Demo ===\n")
    
    try:
        from ethics_model.modules.graph_semantic import (
            SemanticGraphProcessor, 
            create_semantic_processor,
            analyze_text_semantics
        )
        
        print("Initializing Semantic Graph Processor...")
        processor = create_semantic_processor(d_model=128)
        print("✓ Processor initialized successfully\n")
        
        # Test text
        test_text = """
        The doctor helped the patient recover from their illness.
        This demonstrates professional compassion and medical duty.
        However, some patients cannot afford the expensive treatment.
        """
        
        print(f"Analyzing semantic structure of:\n{test_text}\n")
        
        # Analyze semantics
        print("Extracting semantic patterns...")
        analysis = analyze_text_semantics(test_text, processor)
        
        # Display results
        summary = analysis['graph_summary']
        print("Semantic Analysis Results:")
        for key, value in summary.items():
            if not key.startswith('semantic_types'):
                print(f"  - {key}: {value}")
        
        if 'semantic_types' in summary:
            print(f"\nSemantic Type Distribution:")
            for stype, count in summary['semantic_types'].items():
                print(f"  - {stype}: {count}")
        
        # Display ethical relations
        ethical_relations = analysis['ethical_relations']
        for relation_type, entities in ethical_relations.items():
            if entities:
                print(f"\n{relation_type.replace('_', ' ').title()}:")
                for entity in entities[:3]:  # Show first 3
                    if isinstance(entity, dict):
                        print(f"  - {entity.get('text', entity)}")
                    else:
                        print(f"  - {entity}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in semantic graph processing: {e}")
        return False


def demo_integrated_model():
    """Demonstrate the integrated model with graph reasoning."""
    print("\n=== Integrated Model Demo ===\n")
    
    try:
        from ethics_model.model import create_ethics_model
        
        print("Creating enhanced ethics model...")
        
        # Create model configuration
        config = {
            'input_dim': 128,
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 64,
            'use_semantic_graphs': True,
            'spacy_model': 'en_core_web_sm'
        }
        
        model = create_ethics_model(config)
        print("✓ Enhanced model created successfully")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        seq_len = 20
        
        # Create dummy inputs
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test texts for semantic processing
        texts = [
            "Companies should prioritize ethical business practices.",
            "Profit maximization often conflicts with social responsibility."
        ]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts
            )
        
        print("✓ Forward pass completed successfully")
        print(f"  - Ethics score shape: {outputs['ethics_score'].shape}")
        print(f"  - Manipulation score shape: {outputs['manipulation_score'].shape}")
        
        # Get ethical summary
        summary = model.get_ethical_summary(outputs)
        print(f"\nEthical Analysis Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.3f}")
            elif isinstance(value, list):
                print(f"  - {key}: {', '.join(value[:3])}")  # Show first 3
            else:
                print(f"  - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in integrated model demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all demos."""
    print("NetworkX + spaCy Refactoring Demo")
    print("=" * 50)
    
    success_count = 0
    total_demos = 3
    
    # Run demos
    if demo_ethical_relation_extraction():
        success_count += 1
    
    if demo_semantic_graph_processing():
        success_count += 1
    
    if demo_integrated_model():
        success_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Demo Results: {success_count}/{total_demos} successful")
    
    if success_count == total_demos:
        print("🎉 All demos completed successfully!")
        print("\nThe refactoring from GraphBrain to NetworkX + spaCy is working correctly.")
    else:
        print(f"⚠️  Some demos failed. This may be due to missing dependencies.")
        print("\nTo install required dependencies:")
        print("  pip install spacy networkx matplotlib")
        print("  python -m spacy download en_core_web_sm")
    
    return success_count == total_demos


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
