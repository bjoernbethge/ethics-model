"""
Ethics Model Training on ETHICS Dataset

This script trains the enhanced ethics model on the ETHICS dataset
from Hendrycks et al. (2021), using CUDA optimizations for efficiency.
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel
from ethics_model.model import create_ethics_model
from ethics_model.ethics_dataset import create_ethics_dataloaders
from ethics_model.cuda_training import train_ethics_model
import argparse
from datetime import datetime


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"ethics_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # Load LLM model and tokenizer
    print(f"Loading {args.llm_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm = AutoModel.from_pretrained(args.llm_model).to(device)
    
    # Create dataloaders
    print(f"Loading ETHICS dataset from {args.data_dir}...")
    dataloaders = create_ethics_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        domains=args.domains.split(",") if args.domains else None,
        use_graphbrain=not args.disable_graphbrain,
        parser_lang=args.parser_lang,
        cache_graphs=True,
        num_workers=args.num_workers
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
        'use_gnn': True,
        'use_graphbrain': not args.disable_graphbrain,
        'parser_lang': args.parser_lang
    }
    
    model = create_ethics_model(model_config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    criterion = nn.BCELoss()
    
    # Set checkpoint path
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    
    # Train model
    print("Starting training...")
    model = train_ethics_model(
        model=model,
        llm=llm,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders.get('test', None),
        test_dataloader=dataloaders.get('test_hard', None),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        checkpoint_path=checkpoint_path,
        writer=writer,
        use_amp=not args.disable_amp,
        use_cuda_graphs=not args.disable_cuda_graphs
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config
    }, final_model_path)
    
    print(f"Training completed. Model saved to {final_model_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Test on ambiguous examples if available
    if 'ambiguous' in dataloaders:
        print("\nEvaluating on ambiguous examples...")
        model.eval()
        total = 0
        uncertain = 0  # Count examples where prediction is close to 0.5
        
        with torch.no_grad():
            for batch in dataloaders['ambiguous']:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                
                # Forward LLM
                llm_outputs = llm.model.transformer(batch['input_ids']) if hasattr(llm, 'model') else llm.transformer(batch['input_ids'])
                hidden_states = llm_outputs.last_hidden_state
                
                # Forward ethics model
                outputs = model(
                    embeddings=hidden_states,
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts'),
                    graph_data={k: v for k, v in batch.items() if k.startswith('graph_')}
                )
                
                ethics_score = outputs['ethics_score']
                
                # Count uncertain predictions (near 0.5)
                uncertain += ((ethics_score > 0.4) & (ethics_score < 0.6)).sum().item()
                total += ethics_score.size(0)
        
        uncertainty_rate = uncertain / total
        print(f"Uncertainty rate on ambiguous examples: {uncertainty_rate:.4f}")
        print(f"Number of ambiguous examples: {total}")
        print(f"Number of uncertain predictions: {uncertain}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ethics model on ETHICS dataset")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the ETHICS dataset")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated list of domains to include (default: all domains)")
    
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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=2,
                        help="Patience for early stopping")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    # CUDA optimizations
    parser.add_argument("--disable_amp", action="store_true",
                        help="Disable automatic mixed precision training")
    parser.add_argument("--disable_cuda_graphs", action="store_true",
                        help="Disable CUDA Graphs for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    # Other options
    parser.add_argument("--disable_graphbrain", action="store_true",
                        help="Disable GraphBrain semantic hypergraphs")
    parser.add_argument("--parser_lang", type=str, default="en",
                        help="Language for GraphBrain parser")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)
