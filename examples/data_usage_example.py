"""
Example usage of the refactored ethics model data processing.

This script demonstrates how to use the new NetworkX and spaCy-based
data processing pipeline for ethical text analysis.
"""

import torch
from torch.utils.data import DataLoader
import tempfile
import json
from pathlib import Path

# Mock tokenizer for demonstration (in real usage, use transformers tokenizer)
class MockTokenizer:
    def __init__(self, vocab_size=30000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def __call__(self, text, **kwargs):
        # Simple word-based tokenization for demo
        words = text.split()[:kwargs.get('max_length', self.max_length)]
        
        # Convert to token IDs (in reality, use proper vocabulary)
        input_ids = [hash(word) % self.vocab_size for word in words]
        attention_mask = [1] * len(input_ids)
        
        # Pad to required length
        max_len = kwargs.get('max_length', self.max_length)
        while len(input_ids) < max_len:
            input_ids.append(0)
            attention_mask.append(0)
        
        # Truncate if too long
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    # Sample texts with varying ethical content
    texts = [
        "John helped Mary carry her heavy bags up the stairs.",
        "The company deliberately misled consumers about product safety.",
        "She showed great compassion when caring for elderly patients.",
        "The politician manipulated statistics to support false claims.",
        "Volunteers worked together to build homes for homeless families.",
        "The corporation exploited workers by paying below minimum wage.",
        "Teachers patiently guided students through difficult concepts.",
        "Online trolls spread hate speech targeting vulnerable groups.",
        "The community organized to support local food banks.",
        "Hackers stole personal data and sold it for profit.",
        "Researchers shared their findings openly to benefit humanity.",
        "The executive embezzled funds meant for employee pensions.",
        "Neighbors helped each other during the natural disaster.",
        "The website used dark patterns to trick users into subscriptions.",
        "Medical professionals risked their safety to treat patients.",
        "The company dumped toxic waste into the local water supply."
    ]
    
    # Ethics scores (0 = unethical, 1 = highly ethical)
    ethics_scores = [
        0.9,   # Helping with bags
        0.1,   # Misleading consumers
        0.95,  # Showing compassion
        0.05,  # Manipulating statistics
        0.9,   # Volunteering
        0.1,   # Exploiting workers
        0.8,   # Patient teaching
        0.0,   # Spreading hate
        0.85,  # Community support
        0.05,  # Stealing data
        0.9,   # Open research
        0.0,   # Embezzlement
        0.9,   # Helping neighbors
        0.2,   # Dark patterns
        0.95,  # Medical care
        0.0    # Toxic dumping
    ]
    
    # Manipulation scores (0 = no manipulation, 1 = high manipulation)
    manipulation_scores = [
        0.0,   # Helping with bags
        0.9,   # Misleading consumers
        0.0,   # Showing compassion
        0.95,  # Manipulating statistics
        0.0,   # Volunteering
        0.7,   # Exploiting workers
        0.1,   # Patient teaching
        0.8,   # Spreading hate
        0.0,   # Community support
        0.85,  # Stealing data
        0.05,  # Open research
        0.9,   # Embezzlement
        0.0,   # Helping neighbors
        0.9,   # Dark patterns
        0.0,   # Medical care
        0.8    # Toxic dumping
    ]
    
    return texts, ethics_scores, manipulation_scores


def demonstrate_basic_dataset():
    """Demonstrate basic multi-task dataset usage."""
    print("=== Basic Multi-Task Dataset Demo ===")
    
    # Import the data module
    try:
        from src.ethics_model.data import MultiTaskDataset, collate_ethics_batch
    except ImportError:
        print("Error: Could not import ethics model data module")
        return
    
    # Create sample data
    texts, ethics_scores, manipulation_scores = create_sample_dataset()
    
    # Initialize tokenizer
    tokenizer = MockTokenizer(max_length=64)
    
    # Create dataset
    dataset = MultiTaskDataset(
        texts=texts,
        ethics_labels=ethics_scores,
        manipulation_labels=manipulation_scores,
        tokenizer=tokenizer,
        max_length=64,
        augment=False,  # Disable augmentation for demo
        include_raw_text=True
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Examine a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Text: {sample['text']}")
    print(f"  Ethics score: {sample['ethics_label'].item():.3f}")
    print(f"  Manipulation score: {sample['manipulation_label'].item():.3f}")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_ethics_batch
    )
    
    print(f"\nDataLoader created with batch size 4")
    
    # Process one batch
    batch = next(iter(dataloader))
    print(f"\nBatch contents:")
    print(f"  Batch size: {batch['batch_size']}")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Ethics labels shape: {batch['ethics_label'].shape}")
    print(f"  Manipulation labels shape: {batch['manipulation_label'].shape}")
    print(f"  Number of texts: {len(batch['texts'])}")
    
    return dataset


def demonstrate_graph_dataset():
    """Demonstrate graph-enhanced dataset usage."""
    print("\n=== Graph-Enhanced Dataset Demo ===")
    
    try:
        from src.ethics_model.data import GraphEthicsDataset
    except ImportError:
        print("Error: Could not import GraphEthicsDataset")
        return
    
    # Create sample data (subset for demo)
    texts, ethics_scores, manipulation_scores = create_sample_dataset()
    texts = texts[:4]  # Use first 4 samples for demo
    ethics_scores = ethics_scores[:4]
    manipulation_scores = manipulation_scores[:4]
    
    # Initialize tokenizer
    tokenizer = MockTokenizer(max_length=64)
    
    # Create graph-enhanced dataset
    print("Creating graph-enhanced dataset...")
    try:
        dataset = GraphEthicsDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer,
            max_length=64,
            spacy_model="en_core_web_sm",
            preprocess_graphs=True,  # Preprocess for demo
            cache_graphs=True,
            max_graph_nodes=20
        )
        
        print(f"Created graph dataset with {len(dataset)} samples")
        
        # Examine a sample with graph data
        sample = dataset[0]
        print(f"\nSample 0 with graph data:")
        print(f"  Text: {sample['text']}")
        print(f"  Has graph: {sample.get('graph_has_graph', False)}")
        if 'graph_node_features' in sample:
            print(f"  Node features shape: {sample['graph_node_features'].shape}")
        if 'graph_edge_index' in sample:
            print(f"  Edge index shape: {sample['graph_edge_index'].shape}")
        if 'graph_edge_attr' in sample:
            print(f"  Edge attributes shape: {sample['graph_edge_attr'].shape}")
        
    except Exception as e:
        print(f"Graph processing failed (this is expected if spaCy model not installed): {e}")
        print("To use graph features, install spaCy and download the language model:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")


def demonstrate_data_utilities():
    """Demonstrate data utility functions."""
    print("\n=== Data Utilities Demo ===")
    
    try:
        from src.ethics_model.data import create_data_splits, save_to_json, load_from_json
    except ImportError:
        print("Error: Could not import data utilities")
        return
    
    # Create sample data
    texts, ethics_scores, manipulation_scores = create_sample_dataset()
    
    # Create data splits
    print("Creating train/validation/test splits...")
    train_data, val_data, test_data = create_data_splits(
        texts, ethics_scores, manipulation_scores,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42
    )
    
    print(f"  Training samples: {len(train_data['texts'])}")
    print(f"  Validation samples: {len(val_data['texts'])}")
    print(f"  Test samples: {len(test_data['texts'])}")
    
    # Save and load data
    print("\nTesting save/load functionality...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save training data
        save_to_json(
            train_data['texts'], 
            train_data['ethics_labels'], 
            train_data['manipulation_labels'],
            temp_path
        )
        print(f"  Saved data to: {temp_path}")
        
        # Load data back
        loaded_texts, loaded_ethics, loaded_manip = load_from_json(temp_path)
        print(f"  Loaded {len(loaded_texts)} samples")
        
        # Verify data integrity
        assert loaded_texts == train_data['texts']
        assert loaded_ethics == train_data['ethics_labels']
        assert loaded_manip == train_data['manipulation_labels']
        print("  Data integrity verified ✓")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def demonstrate_training_integration():
    """Demonstrate integration with training pipeline."""
    print("\n=== Training Integration Demo ===")
    
    try:
        from src.ethics_model.data import MultiTaskDataset, collate_ethics_batch
    except ImportError:
        print("Error: Could not import required modules")
        return
    
    # Create datasets
    texts, ethics_scores, manipulation_scores = create_sample_dataset()
    
    # Split data
    n_train = int(0.8 * len(texts))
    train_texts = texts[:n_train]
    train_ethics = ethics_scores[:n_train]
    train_manip = manipulation_scores[:n_train]
    
    val_texts = texts[n_train:]
    val_ethics = ethics_scores[n_train:]
    val_manip = manipulation_scores[n_train:]
    
    # Create datasets
    tokenizer = MockTokenizer(max_length=64)
    
    train_dataset = MultiTaskDataset(
        texts=train_texts,
        ethics_labels=train_ethics,
        manipulation_labels=train_manip,
        tokenizer=tokenizer,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = MultiTaskDataset(
        texts=val_texts,
        ethics_labels=val_ethics,
        manipulation_labels=val_manip,
        tokenizer=tokenizer,
        augment=False  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_ethics_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_ethics_batch
    )
    
    print(f"Training data loader: {len(train_loader)} batches")
    print(f"Validation data loader: {len(val_loader)} batches")
    
    # Simulate training loop
    print("\nSimulating training loop:")
    for epoch in range(2):  # Just 2 epochs for demo
        print(f"  Epoch {epoch + 1}")
        
        # Training phase
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # Only process 2 batches per epoch for demo
                break
            
            batch_size = batch['batch_size']
            avg_ethics = batch['ethics_label'].mean().item()
            avg_manip = batch['manipulation_label'].mean().item()
            
            print(f"    Train batch {batch_idx + 1}: size={batch_size}, "
                  f"avg_ethics={avg_ethics:.3f}, avg_manip={avg_manip:.3f}")
        
        # Validation phase
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 1:  # Only process 1 validation batch for demo
                break
            
            batch_size = batch['batch_size']
            avg_ethics = batch['ethics_label'].mean().item()
            avg_manip = batch['manipulation_label'].mean().item()
            
            print(f"    Val batch {batch_idx + 1}: size={batch_size}, "
                  f"avg_ethics={avg_ethics:.3f}, avg_manip={avg_manip:.3f}")


def main():
    """Run all demonstrations."""
    print("Ethics Model Data Processing Demo")
    print("=================================")
    
    # Basic dataset demo
    dataset = demonstrate_basic_dataset()
    
    # Graph dataset demo
    demonstrate_graph_dataset()
    
    # Utilities demo
    demonstrate_data_utilities()
    
    # Training integration demo
    demonstrate_training_integration()
    
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("Key improvements in the refactored data module:")
    print("• NetworkX replaces GraphBrain for better compatibility")
    print("• spaCy provides robust NLP processing")
    print("• Improved error handling and caching")
    print("• Comprehensive testing and validation")
    print("• Better memory management for large datasets")
    print("• Cleaner API with consistent interfaces")


if __name__ == "__main__":
    main()
