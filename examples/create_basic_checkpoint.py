"""
Simple script to create a basic trained checkpoint for the Ethics Model.

This script creates a minimal checkpoint that can be used for testing and demonstration
purposes when the full ETHICS dataset is not available.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path


def create_dummy_checkpoint():
    """
    Create a basic checkpoint with model configuration and random weights.
    
    This is useful for:
    - Testing the API without full model training
    - Demonstrating checkpoint loading
    - Development and debugging
    """
    
    # Define model configuration
    config = {
        'input_dim': 768,  # BERT base hidden size
        'd_model': 512,
        'n_layers': 4,  # Reduced for faster loading
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'activation': 'gelu',
        'vocab_size': 30522,  # BERT vocab size
        'max_seq_length': 512,
        'use_gnn': False,  # Disabled for simplicity
        'use_graphbrain': False,  # Disabled for simplicity
        'num_moral_foundations': 5,
        'num_narrative_frames': 10
    }
    
    # Create a simple model state dict
    # This represents the minimum structure needed for the model
    model_state = {
        'embedding.weight': torch.randn(config['vocab_size'], config['input_dim']),
        'encoder.weight': torch.randn(config['d_model'], config['input_dim']),
        'encoder.bias': torch.randn(config['d_model']),
    }
    
    # Add transformer layer weights
    for i in range(config['n_layers']):
        prefix = f'transformer.layers.{i}'
        model_state.update({
            f'{prefix}.attention.q_proj.weight': torch.randn(config['d_model'], config['d_model']),
            f'{prefix}.attention.q_proj.bias': torch.randn(config['d_model']),
            f'{prefix}.attention.k_proj.weight': torch.randn(config['d_model'], config['d_model']),
            f'{prefix}.attention.k_proj.bias': torch.randn(config['d_model']),
            f'{prefix}.attention.v_proj.weight': torch.randn(config['d_model'], config['d_model']),
            f'{prefix}.attention.v_proj.bias': torch.randn(config['d_model']),
            f'{prefix}.attention.out_proj.weight': torch.randn(config['d_model'], config['d_model']),
            f'{prefix}.attention.out_proj.bias': torch.randn(config['d_model']),
            f'{prefix}.ffn.linear1.weight': torch.randn(config['d_ff'], config['d_model']),
            f'{prefix}.ffn.linear1.bias': torch.randn(config['d_ff']),
            f'{prefix}.ffn.linear2.weight': torch.randn(config['d_model'], config['d_ff']),
            f'{prefix}.ffn.linear2.bias': torch.randn(config['d_model']),
            f'{prefix}.norm1.weight': torch.ones(config['d_model']),
            f'{prefix}.norm1.bias': torch.zeros(config['d_model']),
            f'{prefix}.norm2.weight': torch.ones(config['d_model']),
            f'{prefix}.norm2.bias': torch.zeros(config['d_model']),
        })
    
    # Add output heads
    model_state.update({
        'ethics_head.weight': torch.randn(1, config['d_model']),
        'ethics_head.bias': torch.randn(1),
        'manipulation_head.weight': torch.randn(1, config['d_model']),
        'manipulation_head.bias': torch.randn(1),
        'narrative_head.weight': torch.randn(config['num_narrative_frames'], config['d_model']),
        'narrative_head.bias': torch.randn(config['num_narrative_frames']),
        'foundation_head.weight': torch.randn(config['num_moral_foundations'], config['d_model']),
        'foundation_head.bias': torch.randn(config['num_moral_foundations']),
    })
    
    # Create optimizer state (empty for initialization)
    optimizer_state = {
        'state': {},
        'param_groups': [{
            'lr': 0.0001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'amsgrad': False
        }]
    }
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'config': config,
        'epoch': 0,
        'best_val_loss': float('inf'),
        'training_info': {
            'created_by': 'create_basic_checkpoint.py',
            'description': 'Basic checkpoint for testing and demonstration',
            'trained_on': 'synthetic_data',
            'notes': 'This is a minimal checkpoint with random weights for testing purposes.'
        }
    }
    
    return checkpoint


def main():
    """Create and save a basic checkpoint."""
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create checkpoint
    print("Creating basic checkpoint...")
    checkpoint = create_dummy_checkpoint()
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / 'basic_model.pt'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✓ Checkpoint created successfully!")
    print(f"  Location: {checkpoint_path}")
    print(f"  Size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\nConfiguration:")
    for key, value in checkpoint['config'].items():
        print(f"  {key}: {value}")
    
    print("\n⚠ Note: This checkpoint has random weights and is for testing only.")
    print("   For production use, train the model on actual data using:")
    print("   python examples/train_on_ethics_dataset.py --data_dir path/to/data")
    
    # Verify checkpoint can be loaded
    print("\nVerifying checkpoint can be loaded...")
    try:
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model_state_dict' in loaded_checkpoint
        assert 'config' in loaded_checkpoint
        print("✓ Checkpoint verified successfully!")
    except Exception as e:
        print(f"✗ Error verifying checkpoint: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
