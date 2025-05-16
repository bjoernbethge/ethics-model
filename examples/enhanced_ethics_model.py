"""
Enhanced Ethics Model Example

This example demonstrates how to use the enhanced ethics model with
GraphBrain semantic hypergraphs for ethical analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from ethics_model.model import EnhancedEthicsModel, create_ethics_model
from ethics_model.data import EnhancedEthicsDataset, collate_with_graphs
from ethics_model.training import train
import os
from argparse import ArgumentParser


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load LLM model and tokenizer
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm = AutoModel.from_pretrained(args.llm_model).to(device)
    
    # Example texts and labels
    texts = [
        "Companies should prioritize profit over environmental concerns.",
        "We must protect natural resources for future generations.",
        "The ends justify the means in politics.",
        "Individual freedom is more important than collective welfare.",
        "It's acceptable to lie if it prevents someone from being hurt."
    ]
    
    # Simple dataset creation (0: unethical, 1: ethical)
    ethics_labels = [0.2, 0.8, 0.3, 0.5, 0.4]
    manipulation_labels = [0.7, 0.2, 0.6, 0.3, 0.5]
    
    # Create dataset
    print("Creating dataset...")
    dataset = EnhancedEthicsDataset(
        texts=texts,
        ethics_labels=ethics_labels,
        manipulation_labels=manipulation_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_graphbrain=not args.disable_graphbrain,
        preprocess_graphs=True,
        parser_lang=args.parser_lang
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_with_graphs
    )
    
    # Create ethics model
    print("Creating ethics model...")
    model_config = {
        'input_dim': llm.config.hidden_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'vocab_size': tokenizer.vocab_size,
        'max_seq_length': args.max_length,
        'activation': args.activation,
        'use_gnn': True,  # Default to using GNN
        'use_graphbrain': True if not args.disable_graphbrain else False,  # Default enabled unless disabled
        'parser_lang': args.parser_lang
    }
    
    model = create_ethics_model(model_config).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    # Train the model
    if not args.inference_only:
        print("Training model...")
        model = train(
            model=model,
            llm=llm,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            grad_clip=args.grad_clip,
            checkpoint_path=args.checkpoint_path
        )
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading model from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # Inference example
    print("\nRunning inference...")
    model.eval()
    
    test_text = "Social media platforms should limit free speech to prevent harm."
    inputs = tokenizer(test_text, return_tensors='pt', max_length=args.max_length, 
                     truncation=True, padding='max_length').to(device)
    
    with torch.no_grad():
        # Get LLM embeddings
        llm_outputs = llm.model.transformer(inputs.input_ids) if hasattr(llm, 'model') else llm.transformer(inputs.input_ids)
        hidden_states = llm_outputs.last_hidden_state
        
        # Run model inference
        outputs = model(
            embeddings=hidden_states,
            attention_mask=inputs.attention_mask,
            texts=[test_text]
        )
        
        # Get ethical summary
        summary = model.get_ethical_summary(outputs)
    
    # Display results
    print("\nEthics Analysis Results:")
    print("========================")
    print(f"Text: {test_text}")
    print(f"Ethics Score: {summary['overall_ethics_score']:.2f}")
    print(f"Manipulation Risk: {summary['manipulation_risk']:.2f}")
    print(f"Dominant Framework: {summary['dominant_framework']}")
    print(f"Main Manipulation Techniques: {', '.join(summary['main_manipulation_techniques'])}")
    
    # Display GraphBrain results if available
    if 'semantic_graph_insights' in summary:
        print("\nSemantic Graph Insights:")
        for key, value in summary['semantic_graph_insights'].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Enhanced Ethics Model Example")
    
    # Model parameters
    parser.add_argument("--llm_model", type=str, default="bert-base-uncased",
                        help="Pretrained language model to use")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--activation", type=str, default="gelu",
                        help="Activation function")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=2,
                        help="Patience for early stopping")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/model.pt",
                        help="Path to save/load model checkpoint")
    
    # Feature flags
    parser.add_argument("--disable_graphbrain", action="store_true",
                        help="Disable GraphBrain semantic hypergraphs (enabled by default)")
    parser.add_argument("--parser_lang", type=str, default="en",
                        help="Language for GraphBrain parser")
    parser.add_argument("--use_legacy_model", action="store_true",
                        help="Use the legacy ethics model (without graph features)")
    
    # Other options
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--inference_only", action="store_true",
                        help="Run inference only (no training)")
    
    args = parser.parse_args()
    main(args)
