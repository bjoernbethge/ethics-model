"""
spaCy Data Preprocessing Pipeline for Ethics Model

This module provides functionality to preprocess text data using spaCy
for the ethics model, following spaCy best practices.
"""

import os
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from tqdm import tqdm

# For compatibility with the existing ethics model dataset
from ethics_model.data import MultiTaskDataset


class SpacyPreprocessor:
    """Preprocesses text data using spaCy for ethics and manipulation detection."""
    
    def __init__(
        self, 
        model_name: str = "en_core_web_lg", 
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize the SpacyPreprocessor.
        
        Args:
            model_name: spaCy model to use
            batch_size: Batch size for processing
            cache_dir: Directory to cache processed documents
            device: Device to use for processing ('cpu' or 'cuda')
        """
        self.nlp = spacy.load(model_name)
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.device = device
        
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Configure spaCy pipeline for GPU if available
        if device == "cuda" and spacy.prefer_gpu():
            spacy.require_gpu()
            self.nlp.to_gpu()
            
        # Initialize matchers for patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_matchers()
        
    def _setup_matchers(self):
        """Setup pattern matchers using spaCy's matcher API."""
        # Absolutist terms patterns
        absolutist_terms = ["all", "every", "always", "never", "none", "impossible", 
                          "guaranteed", "certainly", "undoubtedly", "definitely", 
                          "absolutely", "completely", "totally", "universally"]
        self.matcher.add("ABSOLUTIST", [[{"LOWER": {"IN": absolutist_terms}}]])
        
        # Emotional terms patterns
        emotional_terms = ["amazing", "wonderful", "terrible", "horrible", "revolutionary",
                         "breakthrough", "miraculous", "stunning", "shocking", "devastating",
                         "extraordinary", "incredible", "unbelievable", "outrageous"]
        self.matcher.add("EMOTIONAL", [[{"LOWER": {"IN": emotional_terms}}]])
        
        # Hedge terms patterns
        hedge_terms = ["may", "might", "could", "possibly", "perhaps", "probably", 
                     "often", "sometimes", "generally", "typically", "usually", 
                     "in most cases", "tends to", "suggests", "indicates"]
        self.matcher.add("HEDGE", [[{"LOWER": {"IN": hedge_terms}}]])
        
        # Causal markers patterns
        causal_terms = ["because", "therefore", "thus", "hence", "since", "due to", 
                       "leads to", "results in", "causes", "affects", "influences", 
                       "impacts", "consequently", "accordingly"]
        self.matcher.add("CAUSAL", [[{"LOWER": {"IN": causal_terms}}]])
    
    def preprocess(
        self, 
        texts: List[str], 
        ethics_labels: Optional[List[float]] = None,
        manipulation_labels: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Preprocess texts using spaCy and extract features.
        
        Args:
            texts: List of text strings to process
            ethics_labels: Optional ethics labels
            manipulation_labels: Optional manipulation labels
            
        Returns:
            Dictionary with processed features
        """
        # Process texts with spaCy using pipe for efficiency
        docs = list(tqdm(self.nlp.pipe(texts, batch_size=self.batch_size), 
                        total=len(texts), desc="Processing with spaCy"))
        
        # Extract features
        features = self._extract_features(docs)
        
        # Prepare return dictionary
        result = {
            "texts": texts,
            "features": features,
        }
        
        if ethics_labels is not None:
            result["ethics_labels"] = ethics_labels
        
        if manipulation_labels is not None:
            result["manipulation_labels"] = manipulation_labels
            
        return result
    
    def _extract_features(self, docs: List[spacy.tokens.Doc]) -> List[Dict[str, Any]]:
        """
        Extract features from spaCy docs using proper pattern matching.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            List of feature dictionaries
        """
        features = []
        
        for doc in docs:
            # Basic linguistic features
            feature_dict = {
                # Document statistics
                "doc_length": len(doc),
                "sentence_count": len(list(doc.sents)),
                
                # Part-of-speech features
                "noun_ratio": len([t for t in doc if t.pos_ == "NOUN"]) / max(len(doc), 1),
                "verb_ratio": len([t for t in doc if t.pos_ == "VERB"]) / max(len(doc), 1),
                "adj_ratio": len([t for t in doc if t.pos_ == "ADJ"]) / max(len(doc), 1),
                "adv_ratio": len([t for t in doc if t.pos_ == "ADV"]) / max(len(doc), 1),
                
                # Named entity features
                "ent_count": len(doc.ents),
                "ent_types": {ent.label_: 1 for ent in doc.ents},
                
                # Initialize rhetoric pattern counters
                "absolutist_terms": 0,
                "emotional_terms": 0,
                "causal_markers": 0,
                "hedging_terms": 0,
                
                # Entities mentioned
                "entities": [ent.text for ent in doc.ents]
            }
            
            # Use proper matcher to find pattern matches
            matches = self.matcher(doc)
            
            # Process matches by type
            for match_id, start, end in matches:
                match_type = self.nlp.vocab.strings[match_id].lower()
                
                if match_type == "absolutist":
                    feature_dict["absolutist_terms"] += 1
                elif match_type == "emotional":
                    feature_dict["emotional_terms"] += 1
                elif match_type == "hedge":
                    feature_dict["hedging_terms"] += 1
                elif match_type == "causal":
                    feature_dict["causal_markers"] += 1
            
            # Normalize counts
            if len(doc) > 0:
                feature_dict["absolutist_terms"] /= len(doc)
                feature_dict["emotional_terms"] /= len(doc)
                feature_dict["causal_markers"] /= len(doc)
                feature_dict["hedging_terms"] /= len(doc)
                
            features.append(feature_dict)
        
        return features
    
    def create_dataset(
        self,
        processed_data: Dict[str, Any],
        tokenizer: Any,
        max_length: int = 128,
        augment: bool = False,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[MultiTaskDataset, MultiTaskDataset]:
        """
        Create train and test datasets from processed data.
        
        Args:
            processed_data: Data returned from preprocess method
            tokenizer: Tokenizer for the ethics model
            max_length: Maximum sequence length
            augment: Whether to use data augmentation
            test_size: Size of test split
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Extract data
        texts = processed_data["texts"]
        ethics_labels = processed_data.get("ethics_labels", [0.0] * len(texts))
        manipulation_labels = processed_data.get("manipulation_labels", [0.0] * len(texts))
        
        # Create indices for train/test split
        indices = list(range(len(texts)))
        random.seed(seed)
        random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Create datasets
        train_dataset = MultiTaskDataset(
            texts=[texts[i] for i in train_indices],
            ethics_labels=[ethics_labels[i] for i in train_indices],
            manipulation_labels=[manipulation_labels[i] for i in train_indices],
            tokenizer=tokenizer,
            max_length=max_length,
            augment=augment,
            synonym_augment=self.synonym_augment if augment else None
        )
        
        test_dataset = MultiTaskDataset(
            texts=[texts[i] for i in test_indices],
            ethics_labels=[ethics_labels[i] for i in test_indices],
            manipulation_labels=[manipulation_labels[i] for i in test_indices],
            tokenizer=tokenizer,
            max_length=max_length,
            augment=False
        )
        
        return train_dataset, test_dataset
    
    def synonym_augment(self, text: str) -> str:
        """
        Augment text by replacing words with their synonyms.
        Uses proper spaCy processing for context.
        
        Args:
            text: Text to augment
            
        Returns:
            Augmented text
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        
        # Only attempt to replace nouns, verbs, adjectives, and adverbs
        for i, token in enumerate(doc):
            if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and not token.is_stop and len(token.text) > 3:
                # Find similar words using vectors
                if token.has_vector and token.vector_norm > 0:
                    # 30% chance to replace with a similar word
                    if random.random() < 0.3:
                        # Find most similar words in vocabulary using vectors
                        similar_tokens = []
                        for lex in self.nlp.vocab:
                            # Only consider words with the same POS and sufficient frequency
                            if (lex.is_lower and lex.has_vector and lex.vector_norm > 0 and
                                lex.prob >= -15 and lex.pos_ == token.pos_):
                                similarity = token.similarity(lex)
                                if similarity > 0.7 and similarity < 0.99:  # Not identical but similar
                                    similar_tokens.append((lex.text, similarity))
                        
                        # Get top 5 most similar words
                        similar_tokens.sort(key=lambda x: x[1], reverse=True)
                        top_similar = similar_tokens[:5]
                        
                        if top_similar:
                            # Weight by similarity for selection
                            total_sim = sum(sim for _, sim in top_similar)
                            weights = [sim/total_sim for _, sim in top_similar]
                            selected = random.choices([word for word, _ in top_similar], weights=weights, k=1)[0]
                            tokens[i] = selected
        
        return " ".join(tokens)
    
    def culturally_balanced_augmentation(self, 
                                       texts: List[str], 
                                       ethics_labels: List[float],
                                       manipulation_labels: List[float],
                                       balance_factor: float = 2.0,
                                       seed: int = 42) -> Tuple[List[str], List[float], List[float]]:
        """
        Create a balanced dataset by augmenting underrepresented perspectives.
        
        Args:
            texts: Original text data
            ethics_labels: Ethics labels
            manipulation_labels: Manipulation labels
            balance_factor: How much to oversample minority perspectives
            seed: Random seed
            
        Returns:
            Tuple of (augmented_texts, augmented_ethics_labels, augmented_manipulation_labels)
        """
        # Process texts to identify cultural perspectives (using pipe for efficiency)
        docs = list(self.nlp.pipe(texts, batch_size=self.batch_size))
        
        # Initialize the categorization matcher for cultural perspectives
        cultural_matcher = Matcher(self.nlp.vocab)
        
        # Define patterns for different traditions
        western_patterns = [[{"LOWER": {"IN": ["clinical", "research", "study", "evidence", 
                                            "trial", "fda", "medical"]}}]]
        traditional_patterns = [[{"LOWER": {"IN": ["ayurvedic", "traditional", "chinese", 
                                               "tcm", "holistic", "natural", "herbal", 
                                               "acupuncture"]}}]]
        indigenous_patterns = [[{"LOWER": {"IN": ["indigenous", "native", "tribal", 
                                               "ancestral", "wisdom", "shamanic"]}}]]
        
        # Add patterns to matcher
        cultural_matcher.add("WESTERN", western_patterns)
        cultural_matcher.add("TRADITIONAL", traditional_patterns)
        cultural_matcher.add("INDIGENOUS", indigenous_patterns)
        
        # Identify texts from different cultural traditions
        cultural_indices = {
            "western_medical": [],
            "traditional_medicine": [],
            "indigenous": [],
            "other": []
        }
        
        # Categorize texts based on matcher results
        for i, doc in enumerate(docs):
            matches = cultural_matcher(doc)
            categories = set()
            
            for match_id, _, _ in matches:
                category = self.nlp.vocab.strings[match_id].lower()
                if category == "western":
                    categories.add("western_medical")
                elif category == "traditional":
                    categories.add("traditional_medicine")
                elif category == "indigenous":
                    categories.add("indigenous")
            
            if categories:
                # If multiple categories, choose the first one
                category = next(iter(categories))
                cultural_indices[category].append(i)
            else:
                cultural_indices["other"].append(i)
        
        # Find the majority category
        max_count = max(len(indices) for indices in cultural_indices.values())
        
        # Augment underrepresented categories
        augmented_texts = texts.copy()
        augmented_ethics_labels = ethics_labels.copy()
        augmented_manipulation_labels = manipulation_labels.copy()
        
        random.seed(seed)
        for category, indices in cultural_indices.items():
            if category == "other" or len(indices) >= max_count:
                continue
                
            # Determine how many samples to add
            target_count = int(max_count * balance_factor)
            samples_to_add = target_count - len(indices)
            
            # Don't oversample too much
            samples_to_add = min(samples_to_add, len(indices) * 3)
            
            if samples_to_add <= 0:
                continue
                
            # Sample with replacement and augment
            for _ in range(samples_to_add):
                idx = random.choice(indices)
                augmented_text = self.synonym_augment(texts[idx])
                
                augmented_texts.append(augmented_text)
                augmented_ethics_labels.append(ethics_labels[idx])
                augmented_manipulation_labels.append(manipulation_labels[idx])
        
        return augmented_texts, augmented_ethics_labels, augmented_manipulation_labels
    
    def serialize_features(self, features: List[Dict[str, Any]], output_path: str):
        """
        Serialize extracted features to disk using proper JSON formatting.
        
        Args:
            features: List of feature dictionaries
            output_path: Path to save serialized features
        """
        # Convert to serializable format
        serializable_features = []
        for feature_dict in features:
            serializable_dict = {}
            for k, v in feature_dict.items():
                if isinstance(v, dict):
                    serializable_dict[k] = {str(kk): float(vv) for kk, vv in v.items()}
                elif isinstance(v, list):
                    serializable_dict[k] = [str(item) for item in v]
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    serializable_dict[k] = v
                else:
                    serializable_dict[k] = str(v)
            
            serializable_features.append(serializable_dict)
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_features, f, ensure_ascii=False, indent=2)
            
    def load_features(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load serialized features from disk.
        
        Args:
            input_path: Path to load serialized features from
            
        Returns:
            List of feature dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Example usage in a preprocessing script
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(description="Preprocess data with spaCy for ethics model")
    parser.add_argument("--input-file", required=True, help="Path to input data CSV/TSV")
    parser.add_argument("--text-column", default="text", help="Column name for text data")
    parser.add_argument("--ethics-column", default="ethics_label", help="Column name for ethics labels")
    parser.add_argument("--manipulation-column", default="manipulation_label", 
                       help="Column name for manipulation labels")
    parser.add_argument("--model-name", default="en_core_web_lg", help="spaCy model to use")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="Tokenizer to use")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--balance", action="store_true", help="Apply cultural balancing augmentation")
    
    args = parser.parse_args()
    
    # Load data
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.tsv'):
        df = pd.read_csv(args.input_file, sep='\t')
    else:
        raise ValueError("Input file must be CSV or TSV")
    
    # Initialize preprocessor
    preprocessor = SpacyPreprocessor(model_name=args.model_name, cache_dir=args.output_dir)
    
    # Extract data from dataframe
    texts = df[args.text_column].tolist()
    ethics_labels = df[args.ethics_column].tolist() if args.ethics_column in df.columns else None
    manipulation_labels = df[args.manipulation_column].tolist() if args.manipulation_column in df.columns else None
    
    # Apply cultural balancing if requested
    if args.balance and ethics_labels and manipulation_labels:
        texts, ethics_labels, manipulation_labels = preprocessor.culturally_balanced_augmentation(
            texts, ethics_labels, manipulation_labels
        )
    
    # Preprocess data
    processed_data = preprocessor.preprocess(texts, ethics_labels, manipulation_labels)
    
    # Save features
    os.makedirs(args.output_dir, exist_ok=True)
    preprocessor.serialize_features(
        processed_data["features"], 
        os.path.join(args.output_dir, "spacy_features.json")
    )
    
    # Create datasets for the ethics model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset, test_dataset = preprocessor.create_dataset(
        processed_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        augment=True
    )
    
    # Save dataset statistics
    with open(os.path.join(args.output_dir, "dataset_stats.txt"), 'w') as f:
        f.write(f"Original texts: {len(texts)}\n")
        f.write(f"Train dataset size: {len(train_dataset)}\n")
        f.write(f"Test dataset size: {len(test_dataset)}\n")
    
    print(f"Preprocessing complete. Results saved to {args.output_dir}")
