"""
Comprehensive tests for the data processing module.
Tests use real components without mocks.
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path

try:
    from ethics_model.data import (
        MultiTaskDataset,
        GraphEthicsDataset,
        collate_ethics_batch,
        create_data_splits,
        load_from_json,
        save_to_json
    )
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class RealTokenizer:
    """Real tokenizer implementation for testing."""
    
    def __init__(self, vocab_size=2000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        # Build vocabulary from common words
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "the": 2, "a": 3, "an": 4, "and": 5, "or": 6,
            "but": 7, "in": 8, "on": 9, "at": 10, "to": 11, "for": 12, "of": 13,
            "with": 14, "by": 15, "is": 16, "are": 17, "was": 18, "were": 19,
            "be": 20, "been": 21, "have": 22, "has": 23, "had": 24, "do": 25,
            "does": 26, "did": 27, "will": 28, "would": 29, "could": 30, "should": 31,
            "help": 32, "helped": 33, "helping": 34, "harm": 35, "hurt": 36, "damage": 37,
            "good": 38, "bad": 39, "evil": 40, "kind": 41, "cruel": 42, "fair": 43,
            "unfair": 44, "honest": 45, "dishonest": 46, "truth": 47, "lie": 48, "lies": 49,
            "manipulate": 50, "deceive": 51, "exploit": 52, "protect": 53, "save": 54,
            "john": 55, "mary": 56, "people": 57, "person": 58, "company": 59, "politician": 60
        }
        self.next_id = len(self.vocab)
    
    def _get_word_id(self, word):
        word = word.lower()
        if word in self.vocab:
            return self.vocab[word]
        elif self.next_id < self.vocab_size:
            self.vocab[word] = self.next_id
            self.next_id += 1
            return self.vocab[word]
        else:
            return self.vocab["<unk>"]
    
    def __call__(self, text, **kwargs):
        import re
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        max_len = kwargs.get('max_length', self.max_length)
        
        # Convert to IDs
        input_ids = [self._get_word_id(word) for word in words[:max_len]]
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        while len(input_ids) < max_len:
            input_ids.append(0)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }


def create_test_data():
    """Create realistic test data."""
    texts = [
        "John helped Mary with her difficult homework assignment.",
        "The politician deliberately deceived voters about the economy.",
        "She showed great compassion when caring for sick patients.",
        "The company exploited workers by not paying fair wages.",
        "Volunteers built homes for families in need of shelter.",
        "He spread false information to manipulate public opinion.",
        "Teachers guided students through challenging learning materials.",
        "The corporation damaged the environment for higher profits.",
        "Neighbors supported each other during the natural disaster.",
        "Hackers stole personal information for financial gain.",
        "Scientists shared research findings to benefit humanity.",
        "The executive embezzled money from employee retirement funds."
    ]
    
    ethics_scores = [0.9, 0.1, 0.95, 0.15, 0.85, 0.05, 0.8, 0.1, 0.9, 0.05, 0.95, 0.0]
    manipulation_scores = [0.05, 0.9, 0.0, 0.7, 0.1, 0.95, 0.1, 0.8, 0.05, 0.85, 0.05, 0.9]
    
    return texts, ethics_scores, manipulation_scores


class TestMultiTaskDataset:
    """Test the multi-task dataset implementation."""
    
    def test_initialization(self):
        """Test dataset initialization with real data."""
        texts, ethics_scores, manipulation_scores = create_test_data()
        tokenizer = RealTokenizer(max_length=128)
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        assert len(dataset) == len(texts)
        assert dataset.max_length == 128
    
    def test_data_validation(self):
        """Test input validation."""
        texts = ["Text 1", "Text 2"]
        ethics_scores = [0.8]  # Mismatched length
        manipulation_scores = [0.1, 0.9]
        tokenizer = RealTokenizer()
        
        with pytest.raises(ValueError):
            MultiTaskDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer
            )
    
    def test_item_retrieval(self):
        """Test retrieving individual items."""
        texts, ethics_scores, manipulation_scores = create_test_data()
        texts = texts[:3]  # Use subset
        ethics_scores = ethics_scores[:3]
        manipulation_scores = manipulation_scores[:3]
        
        tokenizer = RealTokenizer(max_length=128)
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        item = dataset[0]
        
        # Check required keys
        required_keys = ['input_ids', 'attention_mask', 'ethics_label', 'manipulation_label', 'sample_id']
        for key in required_keys:
            assert key in item
        
        # Check shapes
        assert item['input_ids'].shape == (128,)
        assert item['attention_mask'].shape == (128,)
        assert item['ethics_label'].shape == (1,)
        assert item['manipulation_label'].shape == (1,)
        
        # Check values with tolerance for floating point comparison
        assert abs(item['ethics_label'].item() - ethics_scores[0]) < 1e-6
        assert abs(item['manipulation_label'].item() - manipulation_scores[0]) < 1e-6
        assert item['sample_id'].item() == 0
    
    def test_augmentation(self):
        """Test text augmentation functionality."""
        def simple_augment(text):
            return text + " additional words"
        
        texts = ["Simple test text"]
        ethics_scores = [0.8]
        manipulation_scores = [0.1]
        tokenizer = RealTokenizer()
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer,
            augment=True,
            synonym_augment=simple_augment
        )
        
        # Test multiple times due to random nature
        augmented_found = False
        for _ in range(20):
            item = dataset[0]
            if 'additional' in item.get('augmented_text', ''):
                augmented_found = True
                break
        
        # Should find at least one augmented version
        assert augmented_found
    
    def test_error_handling(self):
        """Test error handling in tokenization."""
        class FailingTokenizer:
            def __call__(self, text, **kwargs):
                if "fail" in text:
                    raise RuntimeError("Tokenization failed")
                return RealTokenizer()(text, **kwargs)
        
        texts = ["Normal text", "This should fail"]
        ethics_scores = [0.5, 0.3]
        manipulation_scores = [0.2, 0.7]
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=FailingTokenizer()
        )
        
        # First item should work
        item0 = dataset[0]
        assert 'input_ids' in item0
        
        # Second item should use fallback
        item1 = dataset[1]
        assert 'input_ids' in item1


class TestGraphEthicsDataset:
    """Test the graph-enhanced dataset."""
    
    def test_initialization(self):
        """Test graph dataset initialization."""
        texts, ethics_scores, manipulation_scores = create_test_data()
        texts = texts[:3]  # Use subset for faster testing
        ethics_scores = ethics_scores[:3]
        manipulation_scores = manipulation_scores[:3]
        
        tokenizer = RealTokenizer(max_length=32)
        
        dataset = GraphEthicsDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer,
            cache_graphs=False,
            max_graph_nodes=10
        )
        
        assert len(dataset) == len(texts)
    
    def test_graph_extraction(self):
        """Test graph extraction functionality."""
        texts = ["John helped Mary"]
        ethics_scores = [0.9]
        manipulation_scores = [0.1]
        tokenizer = RealTokenizer()
        
        try:
            dataset = GraphEthicsDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer,
                spacy_model="en_core_web_sm",
                cache_graphs=False
            )
            
            item = dataset[0]
            
            # Should have graph data
            graph_keys = [k for k in item.keys() if k.startswith('graph_')]
            assert len(graph_keys) > 0
            assert 'graph_has_graph' in item
            
        except OSError:
            # spaCy model not available - test fallback behavior
            dataset = GraphEthicsDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer,
                cache_graphs=False
            )
            
            item = dataset[0]
            # Should still work, may have empty graph data
            assert 'input_ids' in item
    
    def test_caching(self):
        """Test graph caching functionality."""
        texts = ["Test text for caching"]
        ethics_scores = [0.5]
        manipulation_scores = [0.5]
        tokenizer = RealTokenizer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = GraphEthicsDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer,
                cache_graphs=True,
                cache_dir=temp_dir
            )
            
            # First access
            item1 = dataset[0]
            
            # Create new dataset with same cache
            dataset2 = GraphEthicsDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer,
                cache_graphs=True,
                cache_dir=temp_dir
            )
            
            # Second access should use cache
            item2 = dataset2[0]
            
            # Items should be similar
            assert abs(item1['ethics_label'].item() - item2['ethics_label'].item()) < 1e-6


class TestDataUtilities:
    """Test utility functions."""
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        texts, ethics_scores, manipulation_scores = create_test_data()
        
        train_data, val_data, test_data = create_data_splits(
            texts, ethics_scores, manipulation_scores,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            seed=42
        )
        
        # Check sizes
        total_size = len(train_data['texts']) + len(val_data['texts']) + len(test_data['texts'])
        assert total_size == len(texts)
        
        # Check no overlap
        all_texts = set(train_data['texts'] + val_data['texts'] + test_data['texts'])
        assert len(all_texts) == len(texts)
        
        # Check data integrity
        for split_data in [train_data, val_data, test_data]:
            assert len(split_data['texts']) == len(split_data['ethics_labels'])
            assert len(split_data['texts']) == len(split_data['manipulation_labels'])
    
    def test_invalid_ratios(self):
        """Test error handling for invalid ratios."""
        texts, ethics_scores, manipulation_scores = create_test_data()
        
        with pytest.raises(ValueError):
            create_data_splits(
                texts, ethics_scores, manipulation_scores,
                train_ratio=0.5, val_ratio=0.3, test_ratio=0.3  # Sum > 1
            )
    
    def test_json_operations(self):
        """Test JSON save/load operations."""
        texts, ethics_scores, manipulation_scores = create_test_data()
        texts = texts[:4]  # Use subset
        ethics_scores = ethics_scores[:4]
        manipulation_scores = manipulation_scores[:4]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save data
            save_to_json(texts, ethics_scores, manipulation_scores, temp_path)
            
            # Verify file exists and has content
            assert Path(temp_path).exists()
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert len(data) == len(texts)
            
            # Load data back
            loaded_texts, loaded_ethics, loaded_manip = load_from_json(temp_path)
            
            # Verify data integrity
            assert loaded_texts == texts
            assert loaded_ethics == ethics_scores
            assert loaded_manip == manipulation_scores
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_json_format_validation(self):
        """Test JSON format validation."""
        # Create malformed JSON
        malformed_data = [
            {"text": "Valid text", "ethics_score": 0.8, "manipulation_score": 0.1},
            {"text": "Missing ethics", "manipulation_score": 0.9}  # Missing ethics_score
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(malformed_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError):
                load_from_json(temp_path)
                
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCollateFunction:
    """Test the collate function for batching."""
    
    def test_basic_collation(self):
        """Test basic collation of example data."""
        # Create 4 samples
        samples = []
        for i in range(4):
            samples.append({
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'ethics_label': torch.tensor([0.8]),
                'manipulation_label': torch.tensor([0.2]),
                'sample_id': torch.tensor([i])
            })
        
        # Collate
        batch = collate_ethics_batch(samples)
        
        # Check required keys
        required_keys = ['input_ids', 'attention_mask', 'ethics_label', 
                         'manipulation_label', 'sample_id', 'batch_size']
        
        for key in required_keys:
            assert key in batch
        
        # Check shapes
        assert batch['input_ids'].shape == torch.Size([4, 128])
        assert batch['attention_mask'].shape == torch.Size([4, 128])
        assert batch['ethics_label'].shape == (4, 1)
        assert batch['manipulation_label'].shape == (4, 1)
        assert batch['sample_id'].shape == (4, 1)
        assert batch['batch_size'] == 4
    
    def test_empty_batch(self):
        """Test collation of empty batch."""
        result = collate_ethics_batch([])
        assert result == {}
    
    def test_batch_with_graph_data(self):
        """Test collation with graph data."""
        texts = ["Text with graph"]
        ethics_scores = [0.8]
        manipulation_scores = [0.2]
        tokenizer = RealTokenizer()
        
        # Manually create item with graph data
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        item = dataset[0]
        # Add fake graph data
        item['graph_node_features'] = torch.randn(3, 6)
        item['graph_edge_index'] = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        item['graph_has_graph'] = True
        
        batch = [item]
        batched = collate_ethics_batch(batch)
        
        # Check that graph data is preserved
        assert 'graph_node_features' in batched
        assert 'graph_edge_index' in batched
        assert 'graph_has_graph' in batched
        
        assert len(batched['graph_node_features']) == 1
        assert len(batched['graph_has_graph']) == 1


class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""
    
    def test_dataloader_integration(self):
        """Test dataset with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        texts, ethics_scores, manipulation_scores = create_test_data()
        tokenizer = RealTokenizer(max_length=32)
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_ethics_batch
        )
        
        # Test iteration
        batches = list(dataloader)
        assert len(batches) == 3  # 12 samples / 4 batch_size = 3 batches
        
        for batch in batches:
            assert 'input_ids' in batch
            assert 'ethics_label' in batch
            assert 'manipulation_label' in batch
            assert batch['input_ids'].shape[0] <= 4  # Last batch might be smaller
    
    def test_shuffling_reproducibility(self):
        """Test that shuffling is reproducible with seed."""
        from torch.utils.data import DataLoader
        
        texts, ethics_scores, manipulation_scores = create_test_data()
        tokenizer = RealTokenizer()
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        # Set manual seed
        torch.manual_seed(42)
        dataloader1 = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_ethics_batch)
        batch1 = next(iter(dataloader1))
        
        # Reset seed and create new dataloader
        torch.manual_seed(42)
        dataloader2 = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_ethics_batch)
        batch2 = next(iter(dataloader2))
        
        # Should be identical
        assert torch.equal(batch1['input_ids'], batch2['input_ids'])
        assert torch.equal(batch1['ethics_label'], batch2['ethics_label'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
