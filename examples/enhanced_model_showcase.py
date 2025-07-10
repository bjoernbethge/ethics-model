"""
Ethics Model Enhanced Features Showcase

This script demonstrates the enhanced ethics model with explainability,
uncertainty quantification, and graph-based reasoning capabilities.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from ethics_model.model import create_ethics_model
from ethics_model.ethics_dataset import create_ethics_dataloaders, ETHICSDataset
from ethics_model.explainability import EthicsExplainer, AttentionVisualizer, GraphExplainer
from ethics_model.uncertainty import UncertaintyEthicsModel, UncertaintyVisualizer
from ethics_model.graph_reasoning import GraphReasoningEthicsModel, EthicalRelationExtractor, GraphVisualizer
import argparse
from datetime import datetime
import plotly.io as pio


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"ethics_showcase_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load LLM model and tokenizer
    print(f"Loading {args.llm_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm = AutoModel.from_pretrained(args.llm_model).to(device)
    
    # Load or create a model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            model_state = checkpoint
            config = {}
            
        # Create model with default or loaded config
        model = create_ethics_model({
            'input_dim': config.get('input_dim', llm.config.hidden_size),
            'd_model': config.get('d_model', args.d_model),
            'n_layers': config.get('n_layers', args.n_layers),
            'n_heads': config.get('n_heads', args.n_heads),
            'vocab_size': config.get('vocab_size', tokenizer.vocab_size),
            'max_seq_length': config.get('max_seq_length', args.max_length),
            'activation': config.get('activation', args.activation),
            'use_gnn': True,
            'use_graphbrain': True,
            'parser_lang': args.parser_lang
        })
        
        model.load_state_dict(model_state)
    else:
        print("Creating new model...")
        model = create_ethics_model({
            'input_dim': llm.config.hidden_size,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'vocab_size': tokenizer.vocab_size,
            'max_seq_length': args.max_length,
            'activation': args.activation,
            'use_gnn': True,
            'use_graphbrain': True,
            'parser_lang': args.parser_lang
        })
    
    # Move model to device
    model.to(device)
    
    # Choose enhancement based on args
    if args.enhancement == "uncertainty":
        print("Enhancing model with uncertainty quantification...")
        model = UncertaintyEthicsModel(
            base_model=model,
            mc_dropout_rate=args.dropout_rate,
            num_mc_samples=args.mc_samples,
            uncertainty_method=args.uncertainty_method
        )
    elif args.enhancement == "graph_reasoning":
        print("Enhancing model with advanced graph reasoning...")
        model = GraphReasoningEthicsModel(
            base_model=model,
            d_model=args.d_model,
            gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_output_dim=args.gnn_output_dim,
            gnn_num_layers=args.gnn_layers,
            gnn_conv_type=args.gnn_conv_type,
            parser_lang=args.parser_lang
        )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create example texts
    example_texts = [
        "Companies should prioritize profit over environmental concerns.",
        "We must protect natural resources for future generations.",
        "It's acceptable to lie if it prevents someone from being hurt.",
        "The ends justify the means in politics.",
        "Social media platforms should limit free speech to prevent harm."
    ]
    
    if args.enhancement == "explainability":
        showcase_explainability(model, llm, tokenizer, example_texts, device, output_dir)
    elif args.enhancement == "uncertainty":
        showcase_uncertainty(model, llm, tokenizer, example_texts, device, output_dir)
    elif args.enhancement == "graph_reasoning":
        showcase_graph_reasoning(model, llm, tokenizer, example_texts, device, output_dir)
    else:
        # Showcase all features
        showcase_explainability(model, llm, tokenizer, example_texts, device, output_dir)
        showcase_uncertainty(model, llm, tokenizer, example_texts, device, output_dir)
        showcase_graph_reasoning(model, llm, tokenizer, example_texts, device, output_dir)
    
    print(f"Showcases saved to {output_dir}")


def showcase_explainability(model, llm, tokenizer, example_texts, device, output_dir):
    """Demonstrate explainability features."""
    print("\n=== Showcasing Explainability ===")
    
    # Create explainer
    explainer = EthicsExplainer(model, tokenizer, device, parser_lang="en")
    
    # Process each example
    for i, text in enumerate(example_texts):
        print(f"\nAnalyzing example {i+1}: {text[:50]}...")
        
        # Generate explanation
        explanation = explainer.explain(text, llm)
        
        # Print findings
        print(f"Ethics Score: {explanation['ethics_score']:.2f}")
        print(f"Manipulation Score: {explanation['manipulation_score']:.2f}")
        print(f"Dominant Framework: {explanation['framework_analysis']['dominant_framework']}")
        
        print("\nEthical Entities:")
        for entity_type, entities in explanation['ethical_entities'].items():
            if entities:
                print(f"  {entity_type.title()}: {', '.join(str(e) for e in entities[:3])}" + 
                      (f" + {len(entities) - 3} more" if len(entities) > 3 else ""))
        
        # Save visualizations
        os.makedirs(os.path.join(output_dir, "explainability"), exist_ok=True)
        
        # Save attention visualization
        if 'attention_visualization' in explanation:
            fig_path = os.path.join(output_dir, "explainability", f"attention_{i}.html")
            pio.write_html(explanation['attention_visualization'], fig_path)
            print(f"Attention visualization saved to {fig_path}")
        
        # Save token attribution visualization
        if 'token_attribution_visualization' in explanation:
            fig_path = os.path.join(output_dir, "explainability", f"token_attribution_{i}.html")
            pio.write_html(explanation['token_attribution_visualization'], fig_path)
            print(f"Token attribution visualization saved to {fig_path}")
        
        # Save graph visualization
        if 'graph_visualization' in explanation:
            fig_path = os.path.join(output_dir, "explainability", f"graph_{i}.html")
            pio.write_html(explanation['graph_visualization'], fig_path)
            print(f"Graph visualization saved to {fig_path}")


def showcase_uncertainty(model, llm, tokenizer, example_texts, device, output_dir):
    """Demonstrate uncertainty quantification features."""
    print("\n=== Showcasing Uncertainty Quantification ===")
    
    # Ensure model has uncertainty capabilities
    if not isinstance(model, UncertaintyEthicsModel):
        print("Converting model to UncertaintyEthicsModel...")
        model = UncertaintyEthicsModel(
            base_model=model,
            mc_dropout_rate=0.1,
            num_mc_samples=30,
            uncertainty_method="mc_dropout"
        )
    
    # Create visualizer
    visualizer = UncertaintyVisualizer()
    
    # Process each example
    ethics_means = []
    ethics_vars = []
    manipulation_means = []
    manipulation_vars = []
    
    for i, text in enumerate(example_texts):
        print(f"\nAnalyzing example {i+1} with uncertainty: {text[:50]}...")
        
        # Tokenize text
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generate LLM embeddings
        with torch.no_grad():
            llm_outputs = llm.model.transformer(inputs.input_ids) if hasattr(llm, 'model') else llm.transformer(inputs.input_ids)
            hidden_states = llm_outputs.last_hidden_state
        
        # Run model with uncertainty
        with torch.no_grad():
            outputs = model(
                training=False,
                embeddings=hidden_states,
                attention_mask=inputs.attention_mask,
                texts=[text]
            )
        
        # Extract uncertainty estimates
        ethics_mean = outputs['ethics_score'].item()
        ethics_var = outputs['ethics_uncertainty'].item()
        manipulation_mean = outputs['manipulation_score'].item()
        manipulation_var = outputs['manipulation_uncertainty'].item()
        
        # Store for combined visualization
        ethics_means.append(ethics_mean)
        ethics_vars.append(ethics_var)
        manipulation_means.append(manipulation_mean)
        manipulation_vars.append(manipulation_var)
        
        # Print findings
        print(f"Ethics Score: {ethics_mean:.2f} ± {np.sqrt(ethics_var):.2f}")
        print(f"Manipulation Score: {manipulation_mean:.2f} ± {np.sqrt(manipulation_var):.2f}")
    
    # Create combined visualization
    os.makedirs(os.path.join(output_dir, "uncertainty"), exist_ok=True)
    
    # Create uncertainty distribution plot
    fig = visualizer.plot_uncertainty_distribution(
        ethics_means, ethics_vars,
        manipulation_means, manipulation_vars,
        title="Uncertainty in Ethical Analysis"
    )
    
    fig_path = os.path.join(output_dir, "uncertainty", "distributions.html")
    pio.write_html(fig, fig_path)
    print(f"\nUncertainty distribution visualization saved to {fig_path}")


def showcase_graph_reasoning(model, llm, tokenizer, example_texts, device, output_dir):
    """Demonstrate graph-based reasoning features."""
    print("\n=== Showcasing Graph-Based Reasoning ===")
    
    # Create relation extractor and visualizer
    relation_extractor = EthicalRelationExtractor()
    graph_visualizer = GraphVisualizer()
    
    # Process each example
    for i, text in enumerate(example_texts):
        print(f"\nAnalyzing example {i+1} with graph reasoning: {text[:50]}...")
        
        # Extract ethical relations
        relations = relation_extractor.extract_relations(text)
        
        # Print findings
        print(f"Actors: {', '.join(relations['actors'][:3])}" +
              (f" + {len(relations['actors']) - 3} more" if len(relations['actors']) > 3 else ""))
        
        print(f"Actions: {', '.join(action for action, _ in relations['actions'][:3])}" +
              (f" + {len(relations['actions']) - 3} more" if len(relations['actions']) > 3 else ""))
        
        print(f"Values: {', '.join(value for value, _ in relations['values'][:3])}" +
              (f" + {len(relations['values']) - 3} more" if len(relations['values']) > 3 else ""))
        
        # Create visualizations
        os.makedirs(os.path.join(output_dir, "graph_reasoning"), exist_ok=True)
        
        # Create graph visualization
        fig = graph_visualizer.visualize_ethical_graph(
            relations,
            title=f"Ethical Relationship Graph: Example {i+1}"
        )
        
        fig_path = os.path.join(output_dir, "graph_reasoning", f"graph_{i}.html")
        pio.write_html(fig, fig_path)
        print(f"Graph visualization saved to {fig_path}")
        
        # If model has graph reasoning capabilities
        if isinstance(model, GraphReasoningEthicsModel):
            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            ).to(device)
            
            # Generate LLM embeddings
            with torch.no_grad():
                llm_outputs = llm.model.transformer(inputs.input_ids) if hasattr(llm, 'model') else llm.transformer(inputs.input_ids)
                hidden_states = llm_outputs.last_hidden_state
            
            # Run model with graph reasoning
            with torch.no_grad():
                outputs = model(
                    embeddings=hidden_states,
                    attention_mask=inputs.attention_mask,
                    texts=[text]
                )
            
            # Print model outputs
            print(f"Ethics Score: {outputs['ethics_score'].item():.2f}")
            print(f"Manipulation Score: {outputs['manipulation_score'].item():.2f}")
            
            # If graph reasoning outputs available
            if 'graph_reasoning' in outputs and outputs['graph_reasoning'] is not None:
                print("Graph-enhanced reasoning applied successfully")


if __name__ == "__main__":
    # Import numpy here for simpler dependency handling in functions
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Showcase enhanced ethics model features")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (optional)")
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
    
    # Enhancement parameters
    parser.add_argument("--enhancement", type=str, default="all",
                        choices=["explainability", "uncertainty", "graph_reasoning", "all"],
                        help="Enhancement to showcase")
    
    # Uncertainty parameters
    parser.add_argument("--uncertainty_method", type=str, default="mc_dropout",
                        choices=["mc_dropout", "evidential"],
                        help="Uncertainty estimation method")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout rate for MC Dropout")
    parser.add_argument("--mc_samples", type=int, default=30,
                        help="Number of Monte Carlo samples")
    
    # Graph reasoning parameters
    parser.add_argument("--gnn_hidden_dim", type=int, default=64,
                        help="Hidden dimension for GNN")
    parser.add_argument("--gnn_output_dim", type=int, default=32,
                        help="Output dimension for GNN")
    parser.add_argument("--gnn_layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--gnn_conv_type", type=str, default="gat",
                        choices=["gcn", "gat", "gin", "sage", "transformer"],
                        help="GNN convolution type")
    
    # Other options
    parser.add_argument("--parser_lang", type=str, default="en",
                        help="Language for GraphBrain parser")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)
