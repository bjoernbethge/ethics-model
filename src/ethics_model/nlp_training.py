"""
spaCy Component Training Pipeline for Ethics Model

This module provides functionality to train custom spaCy components
for the ethics model, following spaCy best practices.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.matcher import Matcher, PhraseMatcher
from typing import List, Dict, Tuple, Any, Optional, Callable, Union, Iterable
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


@Language.factory(
    "ethics_detector",
    default_config={
        "absolutist_terms": ["all", "every", "always", "never", "none", "impossible", 
                           "guaranteed", "certainly", "undoubtedly", "definitely", 
                           "absolutely", "completely", "totally", "universally"],
        "emotional_intensifiers": ["amazing", "wonderful", "terrible", "horrible", "revolutionary",
                                 "breakthrough", "miraculous", "stunning", "shocking", "devastating",
                                 "extraordinary", "incredible", "unbelievable", "outrageous"],
        "hedge_terms": ["may", "might", "could", "possibly", "perhaps", "probably", 
                      "often", "sometimes", "generally", "typically", "usually", 
                      "in most cases", "tends to", "suggests", "indicates"],
        "causal_markers": ["because", "therefore", "thus", "hence", "since", "due to", 
                         "leads to", "results in", "causes", "affects", "influences", 
                         "impacts", "consequently", "accordingly"],
        "weights": {"absolutist": 0.7, "emotional": 0.6, "hedge": -0.3, "causal": 0.1}
    }
)
def create_ethics_detector(nlp: Language, name: str, 
                          absolutist_terms: List[str],
                          emotional_intensifiers: List[str],
                          hedge_terms: List[str],
                          causal_markers: List[str],
                          weights: Dict[str, float]):
    """Factory function for creating EthicsDetector component"""
    return EthicsDetector(nlp, name, absolutist_terms, emotional_intensifiers, 
                         hedge_terms, causal_markers, weights)


class EthicsDetector:
    """
    spaCy component for detecting rhetorical patterns and generating ethics scores.
    Follows spaCy best practices for component architecture.
    """
    
    def __init__(self, 
                nlp: Language, 
                name: str,
                absolutist_terms: List[str],
                emotional_intensifiers: List[str],
                hedge_terms: List[str],
                causal_markers: List[str],
                weights: Dict[str, float]):
        """Initialize the component with configuration."""
        self.name = name
        self.nlp = nlp
        self.vocab = nlp.vocab
        
        # Register extensions
        if not Doc.has_extension("rhetoric_patterns"):
            Doc.set_extension("rhetoric_patterns", default={})
        if not Doc.has_extension("ethics_score"):
            Doc.set_extension("ethics_score", default=0.0)
        if not Doc.has_extension("manipulation_score"):
            Doc.set_extension("manipulation_score", default=0.0)
        if not Token.has_extension("rhetorically_significant"):
            Token.set_extension("rhetorically_significant", default=False)
            
        # Store configuration
        self.absolutist_terms = absolutist_terms
        self.emotional_intensifiers = emotional_intensifiers
        self.hedge_terms = hedge_terms
        self.causal_markers = causal_markers
        self.weights = weights
        
        # Initialize matchers
        self.matcher = Matcher(nlp.vocab)
        
        # Add pattern groups
        self._add_matcher_patterns()
        
    def _add_matcher_patterns(self):
        """Add patterns to the matcher using spaCy pattern syntax."""
        # Absolutist patterns
        absolutist_patterns = [[{"LOWER": {"IN": self.absolutist_terms}}]]
        self.matcher.add("ABSOLUTIST", absolutist_patterns)
        
        # Emotional patterns
        emotional_patterns = [[{"LOWER": {"IN": self.emotional_intensifiers}}]]
        self.matcher.add("EMOTIONAL", emotional_patterns)
        
        # Hedge patterns
        hedge_patterns = [[{"LOWER": {"IN": self.hedge_terms}}]]
        self.matcher.add("HEDGE", hedge_patterns)
        
        # Causal patterns
        causal_patterns = [[{"LOWER": {"IN": self.causal_markers}}]]
        self.matcher.add("CAUSAL", causal_patterns)
        
    def __call__(self, doc: Doc) -> Doc:
        """Process a document using spaCy's pattern matching."""
        # Initialize rhetoric patterns
        doc._.rhetoric_patterns = {
            "absolutist": [],
            "emotional": [],
            "hedge": [],
            "causal": []
        }
        
        # Find matches using the matcher
        matches = self.matcher(doc)
        
        # Process matches
        for match_id, start, end in matches:
            # Get match type
            match_type = self.nlp.vocab.strings[match_id].lower()
            
            # Store match information
            span = doc[start:end]
            doc._.rhetoric_patterns[match_type].append(span.text.lower())
            
            # Mark tokens as rhetorically significant
            for token in span:
                token._.rhetorically_significant = True
        
        # Calculate scores
        rhetoric_counts = {
            k: len(v) for k, v in doc._.rhetoric_patterns.items()
        }
        
        doc_len = len(doc)
        if doc_len > 0:
            manipulation_score = sum(
                count * self.weights.get(pattern_type, 0.0) / doc_len
                for pattern_type, count in rhetoric_counts.items()
            )
            
            # Ensure score is between 0 and 1
            doc._.manipulation_score = max(0.0, min(1.0, manipulation_score))
            
            # Calculate ethics score
            doc._.ethics_score = 1.0 - doc._.manipulation_score
        
        return doc
    
    def to_disk(self, path, exclude=tuple()):
        """Serialize component using spaCy's serialization protocol."""
        data = {
            "weights": self.weights,
            "absolutist_terms": self.absolutist_terms,
            "emotional_intensifiers": self.emotional_intensifiers,
            "hedge_terms": self.hedge_terms,
            "causal_markers": self.causal_markers
        }
        
        path = path if not isinstance(path, str) else Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        
        # Save JSON config
        config_path = path / "cfg.json"
        with config_path.open("w", encoding="utf8") as f:
            f.write(json.dumps(data))
    
    def from_disk(self, path, exclude=tuple()):
        """Load component using spaCy's serialization protocol."""
        path = path if not isinstance(path, str) else Path(path)
        config_path = path / "cfg.json"
        
        with config_path.open("r", encoding="utf8") as f:
            data = json.loads(f.read())
        
        self.weights = data["weights"]
        self.absolutist_terms = data["absolutist_terms"]
        self.emotional_intensifiers = data["emotional_intensifiers"]
        self.hedge_terms = data["hedge_terms"]
        self.causal_markers = data["causal_markers"]
        
        # Rebuild matcher with updated patterns
        self.matcher = Matcher(self.vocab)
        self._add_matcher_patterns()
        
        return self


class EthicsComponentTrainer:
    """
    Trainer for the EthicsDetector spaCy component using proper spaCy training.
    """
    
    def __init__(
        self, 
        model_name: str = "en_core_web_lg",
        component_name: str = "ethics_detector",
        device: str = "cpu"
    ):
        """Initialize the trainer with spaCy model and component."""
        # Initialize spaCy model
        self.nlp = spacy.load(model_name)
        
        # Add component if it doesn't exist
        if component_name not in self.nlp.pipe_names:
            self.nlp.add_pipe(component_name)
        
        self.component_name = component_name
        self.device = device
        
        # Enable GPU if available
        if device == "cuda" and spacy.prefer_gpu():
            spacy.require_gpu()
            self.nlp.to_gpu()
    
    def prepare_examples(
        self,
        texts: List[str],
        manipulation_labels: List[float],
        ethics_labels: Optional[List[float]] = None
    ) -> List[Example]:
        """
        Prepare training examples for spaCy.
        
        Args:
            texts: List of text strings
            manipulation_labels: Manipulation labels (0-1)
            ethics_labels: Optional ethics labels (0-1)
            
        Returns:
            List of spaCy Example objects
        """
        examples = []
        
        for i, (text, manipulation_score) in enumerate(zip(texts, manipulation_labels)):
            # Create the predicted Doc object
            pred = self.nlp.make_doc(text)
            
            # Create the gold Doc object
            gold = self.nlp.make_doc(text)
            
            # Add annotations to the gold Doc
            gold._.manipulation_score = manipulation_score
            
            if ethics_labels:
                gold._.ethics_score = ethics_labels[i]
            else:
                gold._.ethics_score = 1.0 - manipulation_score
            
            # Create example
            examples.append(Example(pred, gold))
        
        return examples
    
    def train(
        self,
        examples: List[Example],
        n_iter: int = 10,
        dropout: float = 0.2,
        init_lr: float = 0.001,
        output_dir: str = "./ethics_model",
    ) -> Dict[str, List[float]]:
        """
        Train the ethics detector component using spaCy's update method.
        
        Args:
            examples: List of spaCy Example objects
            n_iter: Number of training iterations
            dropout: Dropout rate
            init_lr: Initial learning rate
            output_dir: Output directory for saved model
            
        Returns:
            Dictionary of training losses
        """
        # Get only the ethics_detector component
        pipe_exceptions = ["ethics_detector"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
        
        # Track losses
        losses = {"total": []}
        
        # Train with disabled pipes
        with self.nlp.disable_pipes(*other_pipes):
            # Create optimizer
            optimizer = self.nlp.create_optimizer()
            
            # Training loop
            batch_sizes = compounding(4.0, 32.0, 1.001)
            
            # Progress bar for iterations
            for i in tqdm(range(n_iter), desc="Training ethics detector"):
                # Shuffle examples
                random.shuffle(examples)
                
                # Create minibatches
                batches = minibatch(examples, size=batch_sizes)
                
                # Track losses for this iteration
                iteration_loss = 0.0
                batch_count = 0
                
                # Process batches
                for batch in batches:
                    # Update model
                    losses_dict = {}
                    self.nlp.update(
                        batch,
                        drop=dropout,
                        losses=losses_dict,
                        sgd=optimizer
                    )
                    
                    # Accumulate loss
                    if losses_dict:
                        iteration_loss += sum(losses_dict.values())
                        batch_count += 1
                
                # Record mean loss for iteration
                mean_loss = iteration_loss / max(batch_count, 1)
                losses["total"].append(mean_loss)
                
                # Print progress
                print(f"Iteration {i+1}/{n_iter}, Loss: {mean_loss:.4f}")
        
        # Save trained model
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
                
            # Save the full pipeline
            self.nlp.to_disk(output_dir / "spacy_ethics_model")
        
        return losses
    
    def evaluate(
        self,
        test_texts: List[str],
        test_manipulation_labels: List[float],
        test_ethics_labels: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the ethics detector component.
        
        Args:
            test_texts: List of test text strings
            test_manipulation_labels: Test manipulation labels
            test_ethics_labels: Optional test ethics labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Process test texts
        test_docs = list(self.nlp.pipe(test_texts))
        
        # Calculate metrics
        manipulation_mse = 0.0
        ethics_mse = 0.0
        
        for i, doc in enumerate(test_docs):
            manipulation_mse += (doc._.manipulation_score - test_manipulation_labels[i]) ** 2
            
            if test_ethics_labels:
                ethics_mse += (doc._.ethics_score - test_ethics_labels[i]) ** 2
            else:
                ethics_mse += (doc._.ethics_score - (1.0 - test_manipulation_labels[i])) ** 2
        
        manipulation_mse /= len(test_docs)
        ethics_mse /= len(test_docs)
        
        # Calculate correlation
        manipulation_preds = [doc._.manipulation_score for doc in test_docs]
        manipulation_corr = np.corrcoef(manipulation_preds, test_manipulation_labels)[0, 1]
        
        if test_ethics_labels:
            ethics_preds = [doc._.ethics_score for doc in test_docs]
            ethics_corr = np.corrcoef(ethics_preds, test_ethics_labels)[0, 1]
        else:
            ethics_preds = [doc._.ethics_score for doc in test_docs]
            ethics_corr = np.corrcoef(ethics_preds, [1.0 - m for m in test_manipulation_labels])[0, 1]
        
        return {
            "manipulation_mse": manipulation_mse,
            "ethics_mse": ethics_mse,
            "manipulation_correlation": manipulation_corr,
            "ethics_correlation": ethics_corr
        }
    
    def get_predictions(
        self,
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate predictions with proper spaCy processing."""
        # Process texts
        docs = list(self.nlp.pipe(texts))
        
        # Generate predictions
        predictions = []
        
        for doc in docs:
            # Get all matched spans
            spans = []
            for token in doc:
                if token._.rhetorically_significant:
                    spans.append({
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "sent": token.sent.text
                    })
            
            # Collect prediction info
            prediction = {
                "text": doc.text,
                "manipulation_score": doc._.manipulation_score,
                "ethics_score": doc._.ethics_score,
                "rhetoric_patterns": doc._.rhetoric_patterns,
                "highlighted_spans": spans
            }
            
            predictions.append(prediction)
        
        return predictions


# Example usage in a training script
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Train spaCy component for ethics model")
    parser.add_argument("--input-file", required=True, help="Path to input data CSV/TSV")
    parser.add_argument("--text-column", default="text", help="Column name for text data")
    parser.add_argument("--ethics-column", default="ethics_label", help="Column name for ethics labels")
    parser.add_argument("--manipulation-column", default="manipulation_label", 
                       help="Column name for manipulation labels")
    parser.add_argument("--model-name", default="en_core_web_lg", help="spaCy model to use")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-iter", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load data
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.tsv'):
        df = pd.read_csv(args.input_file, sep='\t')
    else:
        raise ValueError("Input file must be CSV or TSV")
    
    # Extract data from dataframe
    texts = df[args.text_column].tolist()
    ethics_labels = df[args.ethics_column].tolist() if args.ethics_column in df.columns else None
    manipulation_labels = df[args.manipulation_column].tolist() if args.manipulation_column in df.columns else None
    
    # Split data into train and test
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    test_texts = texts[train_size:]
    
    train_manipulation_labels = manipulation_labels[:train_size] if manipulation_labels else None
    test_manipulation_labels = manipulation_labels[train_size:] if manipulation_labels else None
    
    train_ethics_labels = ethics_labels[:train_size] if ethics_labels else None
    test_ethics_labels = ethics_labels[train_size:] if ethics_labels else None
    
    # Initialize trainer
    trainer = EthicsComponentTrainer(model_name=args.model_name)
    
    # Prepare training examples
    examples = trainer.prepare_examples(
        train_texts, 
        train_manipulation_labels,
        train_ethics_labels
    )
    
    # Train component
    losses = trainer.train(
        examples,
        n_iter=args.n_iter,
        dropout=args.dropout,
        init_lr=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Save losses
    pd.DataFrame({"loss": losses["total"]}).to_csv(
        Path(args.output_dir) / "training_losses.csv", 
        index=False
    )
    
    # Evaluate on test set
    if test_texts and test_manipulation_labels:
        metrics = trainer.evaluate(
            test_texts,
            test_manipulation_labels,
            test_ethics_labels
        )
        
        # Save metrics
        with open(Path(args.output_dir) / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"Training complete. Model saved to {args.output_dir}")
