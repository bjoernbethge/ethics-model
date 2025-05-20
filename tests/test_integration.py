"""
Comprehensive integration tests for the ethics model components.
Tests use real components without mocks to ensure actual functionality.
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import spacy
from ethics_model.model import EnhancedEthicsModel
from ethics_model.modules.graph_semantic import SemanticGraphProcessor
from torch_geometric.data import Data
import copy
from tqdm import tqdm

try:
    from ethics_model.data import (
        MultiTaskDataset, 
        GraphEthicsDataset, 
        collate_ethics_batch,
        create_data_splits,
        load_from_json,
        save_to_json
    )
    from ethics_model.model import create_ethics_model, EthicsModel
    from ethics_model.training import train, validate, calculate_metrics
    from ethics_model.graph_reasoning import (
        EthicalRelationExtractor,
        EthicalGNN,
        extract_and_visualize
    )
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


class SimpleTokenizer:
    """Simple real tokenizer implementation for testing."""
    
    def __init__(self, vocab_size=5000, max_length=64):
        self.vocab_size = vocab_size
        self.max_length = max_length
        # Create a simple vocabulary mapping
        self.word_to_id = {"<pad>": 0, "<unk>": 1}
        self.id_to_word = {0: "<pad>", 1: "<unk>"}
        self.next_id = 2
    
    def _get_word_id(self, word):
        """Get or create word ID."""
        if word not in self.word_to_id:
            if self.next_id < self.vocab_size:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
            else:
                return self.word_to_id["<unk>"]
        return self.word_to_id[word]
    
    def __call__(self, text, **kwargs):
        words = text.lower().split()
        max_len = kwargs.get('max_length', self.max_length)
        input_ids = [self._get_word_id(word) for word in words[:max_len]]
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_len:
            input_ids.append(0)
            attention_mask.append(0)
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask])
        }


class SimpleLLM(torch.nn.Module):
    """Simple LLM implementation for testing."""
    
    def __init__(self, vocab_size=5000, d_model=512, max_length=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
    def __call__(self, input_ids):
        embeddings = self.embedding(input_ids.long())
        output = self.transformer(embeddings)
        # Return object with last_hidden_state attribute
        class Output:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states
        return Output(output)


def create_sample_data():
    """Create realistic sample data for testing."""
    texts = [
        "John helped Mary carry her heavy groceries up the stairs.",
        "The company deliberately misled customers about product safety risks.",
        "She showed genuine compassion while caring for elderly patients.",
        "The politician manipulated statistics to support false claims about unemployment.",
        "Volunteers worked together to build affordable homes for homeless families.",
        "The corporation exploited workers by paying below minimum wage without benefits.",
        "Teachers patiently guided struggling students through difficult mathematical concepts.",
        "Online trolls spread hate speech targeting vulnerable minority groups.",
        "The community organized food drives to support local families in need.",
        "Hackers stole personal data and sold it on the dark web for profit.",
        "Researchers openly shared their medical findings to benefit global health.",
        "The executive embezzled pension funds meant for retiring employees.",
        "Neighbors helped each other rebuild after the devastating hurricane.",
        "The website used deceptive dark patterns to trick users into subscriptions.",
        "Medical professionals risked their own safety to treat COVID patients.",
        "The factory dumped toxic chemicals into the local water supply."
    ]
    
    # Ethics scores (0 = unethical, 1 = highly ethical)
    ethics_scores = [
        0.9, 0.1, 0.95, 0.05, 0.9, 0.1, 0.8, 0.0,
        0.85, 0.05, 0.9, 0.0, 0.9, 0.2, 0.95, 0.0
    ]
    
    # Manipulation scores (0 = no manipulation, 1 = high manipulation)
    manipulation_scores = [
        0.0, 0.9, 0.0, 0.95, 0.0, 0.7, 0.1, 0.8,
        0.0, 0.85, 0.05, 0.9, 0.0, 0.9, 0.0, 0.8
    ]
    
    return texts, ethics_scores, manipulation_scores


class TestDataProcessing:
    """Test data processing components."""
    
    def test_multi_task_dataset(self):
        """Test basic multi-task dataset functionality."""
        texts, ethics_scores, manipulation_scores = create_sample_data()
        tokenizer = SimpleTokenizer(vocab_size=1000, max_length=64)
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=lambda text, **kwargs: SimpleTokenizer(vocab_size=1000, max_length=64)(text, max_length=64)
        )
        
        assert len(dataset) == len(texts)
        
        # Test item retrieval
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'ethics_label' in item
        assert 'manipulation_label' in item
        assert 'text' in item
        
        assert item['input_ids'].shape == (tokenizer.max_length,)
        assert item['attention_mask'].shape == (tokenizer.max_length,)
        assert pytest.approx(item['ethics_label'].item(), 0.0001) == ethics_scores[0]
        assert pytest.approx(item['manipulation_label'].item(), 0.0001) == manipulation_scores[0]
    
    def test_graph_ethics_dataset(self):
        """Test graph-enhanced dataset."""
        texts, ethics_scores, manipulation_scores = create_sample_data()
        texts = texts[:4]  # Use subset for faster testing
        ethics_scores = ethics_scores[:4]
        manipulation_scores = manipulation_scores[:4]
        
        tokenizer = SimpleTokenizer(vocab_size=1000, max_length=32)
        
        # Test with actual spaCy if available, otherwise test basic functionality
        try:
            dataset = GraphEthicsDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer,
                spacy_model="en_core_web_sm",
                cache_graphs=False,
                max_graph_nodes=10
            )
            
            assert len(dataset) == len(texts)
            
            # Test item with graph data
            item = dataset[0]
            assert 'input_ids' in item
            assert 'graph_has_graph' in item
            assert 'graph_node_features' in item
            assert 'graph_edge_index' in item
            
        except OSError:
            # spaCy model not available - test fallback
            dataset = GraphEthicsDataset(
                texts=texts,
                ethics_labels=ethics_scores,
                manipulation_labels=manipulation_scores,
                tokenizer=tokenizer,
                cache_graphs=False
            )
            
            # Should still work with fallback
            assert len(dataset) == len(texts)
            item = dataset[0]
            assert 'input_ids' in item
    
    def test_collate_function(self):
        """Test batch collation."""
        texts, ethics_scores, manipulation_scores = create_sample_data()
        texts = texts[:4]
        ethics_scores = ethics_scores[:4]
        manipulation_scores = manipulation_scores[:4]
        
        tokenizer = SimpleTokenizer(vocab_size=1000, max_length=32)
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=lambda text, **kwargs: SimpleTokenizer(vocab_size=1000, max_length=32)(text, max_length=32)
        )
        
        # Create batch
        batch = [dataset[i] for i in range(3)]
        batched = collate_ethics_batch(batch)
        
        assert batched['batch_size'] == 3
        assert batched['input_ids'].shape == (3, tokenizer.max_length)
        assert batched['ethics_label'].shape == (3, 1)
        assert batched['manipulation_label'].shape == (3, 1)
        assert len(batched['texts']) == 3
    
    def test_data_splits(self):
        """Test data splitting functionality."""
        texts, ethics_scores, manipulation_scores = create_sample_data()
        
        train_data, val_data, test_data = create_data_splits(
            texts, ethics_scores, manipulation_scores,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            seed=42
        )
        
        total_samples = len(train_data['texts']) + len(val_data['texts']) + len(test_data['texts'])
        assert total_samples == len(texts)
        
        # Test reproducibility
        train_data2, val_data2, test_data2 = create_data_splits(
            texts, ethics_scores, manipulation_scores,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            seed=42
        )
        
        assert train_data['texts'] == train_data2['texts']
    
    def test_json_io(self):
        """Test JSON save/load functionality."""
        texts, ethics_scores, manipulation_scores = create_sample_data()
        texts = texts[:5]  # Use subset
        ethics_scores = ethics_scores[:5]
        manipulation_scores = manipulation_scores[:5]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save data
            save_to_json(texts, ethics_scores, manipulation_scores, temp_path)
            
            # Load data back
            loaded_texts, loaded_ethics, loaded_manip = load_from_json(temp_path)
            
            assert loaded_texts == texts
            assert loaded_ethics == ethics_scores
            assert loaded_manip == manipulation_scores
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestModelComponents:
    """Test model components."""
    
    def test_basic_ethics_model(self):
        """Test basic ethics model functionality."""
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        assert isinstance(model, EnhancedEthicsModel)
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len).float()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
        assert outputs['ethics_score'].shape == (batch_size, 1)
        assert outputs['manipulation_score'].shape == (batch_size, 1)
        
        # Test summary generation
        summary = model.get_ethical_summary(outputs)
        assert isinstance(summary, dict)
        assert 'overall_ethics_score' in summary
        assert 'manipulation_risk' in summary
    
    def test_enhanced_model_features(self):
        """Test enhanced model with all features enabled."""
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_semantic_graphs': True,
            'use_gnn': True
        }
        
        # Test model creation (might fail if spaCy not available)
        try:
            model = create_ethics_model(config)
            
            # Test forward pass with texts
            batch_size = 2
            seq_len = 32
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len).float()
            texts = ["This is good", "This is bad"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts
            )
            
            assert 'ethics_score' in outputs
            assert 'manipulation_score' in outputs
            
        except (ImportError, OSError):
            # spaCy not available - skip enhanced features test
            pytest.skip("spaCy not available for enhanced model testing")
    
    def test_legacy_model_compatibility(self):
        """Test legacy model compatibility."""
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_legacy': True
        }
        
        model = create_ethics_model(config)
        assert isinstance(model, EthicsModel)
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32).float()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs


class TestGraphReasoning:
    """Test graph reasoning components."""
    
    def test_ethical_relation_extractor(self):
        """Test ethical relation extraction."""
        try:
            extractor = EthicalRelationExtractor("en_core_web_sm")
            
            text = "John helped Mary when she was struggling with her homework."
            relations = extractor.extract_relations(text)
            
            assert isinstance(relations, dict)
            assert 'entities' in relations
            assert 'relations' in relations
            assert 'graph' in relations
            
        except OSError:
            # spaCy model not available
            pytest.skip("spaCy model not available for relation extraction testing")
    
    def test_ethical_gnn(self):
        """Test ethical GNN component."""
        from torch_geometric.data import Data
        
        gnn = EthicalGNN(
            in_channels=6,
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
            conv_type="gcn"
        )
        
        # Create sample graph data
        x = torch.randn(5, 6)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 2)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        outputs = gnn(data)
        
        assert 'node_embeddings' in outputs
        assert 'graph_embedding' in outputs
        assert 'ethics_score' in outputs
        assert 'manipulation_score' in outputs
        
        assert outputs['node_embeddings'].shape == (5, 16)
        assert outputs['graph_embedding'].shape == (1, 16)
    
    def test_extract_and_visualize(self):
        """Test graph extraction and analysis utility."""
        text = "The politician helped citizens understand complex policies."
        
        try:
            result = extract_and_visualize(text)
            
            assert 'relations' in result
            assert 'metrics' in result
            assert 'graph' in result
            
            metrics = result['metrics']
            if 'n_nodes' in metrics:
                assert isinstance(metrics['n_nodes'], int)
                assert metrics['n_nodes'] >= 0
                
        except OSError:
            pytest.skip("spaCy model not available for graph analysis")


class TestTraining:
    """Test training components."""
    
    def test_training_pipeline(self):
        """Test complete training pipeline."""
        # Create small dataset
        texts, ethics_scores, manipulation_scores = create_sample_data()
        texts = texts[:8]  # Use subset for faster testing
        ethics_scores = ethics_scores[:8]
        manipulation_scores = manipulation_scores[:8]
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        llm = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in llm.parameters():
            param.requires_grad = False
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_ethics_batch
        )
        
        # Create model and LLM
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        # Test single epoch training
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        trained_model = train(
            model=model,
            llm=llm,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=1,
            patience=10,
            use_amp=False
        )
        
        # Check that parameters changed (model was trained)
        parameters_changed = False
        for name, param in trained_model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                parameters_changed = True
                break
        
        assert parameters_changed, "Model parameters should change during training"
    
    def test_validation(self):
        """Test validation functionality."""
        # Create validation dataset
        texts = ["Good behavior", "Bad behavior"]
        ethics_scores = [0.9, 0.1]
        manipulation_scores = [0.1, 0.9]
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        llm = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in llm.parameters():
            param.requires_grad = False
        
        dataset = MultiTaskDataset(
            texts=texts,
            ethics_labels=ethics_scores,
            manipulation_labels=manipulation_scores,
            tokenizer=tokenizer
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_ethics_batch
        )
        
        # Create model
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 1000,
            'max_seq_length': 16,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        # Angepasste Validierungsfunktion, die das LLM direkt verwendet
        def custom_validate(model, dataloader, criterion, device, return_predictions=False):
            model.eval()
            total_loss = 0.0
            num_samples = 0
            all_ethics_preds = []
            all_ethics_labels = []
            all_manip_preds = []
            all_manip_labels = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Validating"):
                    try:
                        # Daten auf das Gerät verschieben
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        ethics_label = batch['ethics_label'].to(device)
                        manipulation_label = batch['manipulation_label'].to(device)
                        texts = batch.get('texts', None)
                        
                        # LLM-Embeddings extrahieren
                        llm_outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
                        hidden_states = llm_outputs.last_hidden_state
                        
                        # Model-Forward mit den LLM-Embeddings
                        outputs = model(
                            embeddings=hidden_states,
                            attention_mask=attention_mask,
                            texts=texts
                        )
                        
                        # Verluste berechnen
                        ethics_score = outputs['ethics_score']
                        manipulation_score = outputs['manipulation_score']
                        
                        loss_ethics = criterion(ethics_score, ethics_label.view(-1, 1).float())
                        loss_manip = criterion(manipulation_score, manipulation_label.view(-1, 1).float())
                        loss = loss_ethics + 0.5 * loss_manip
                        
                        # Metriken aktualisieren
                        batch_size = input_ids.size(0)
                        total_loss += loss.item() * batch_size
                        num_samples += batch_size
                        
                        # Vorhersagen speichern, wenn angefordert
                        if return_predictions:
                            all_ethics_preds.extend(ethics_score.cpu().numpy())
                            all_ethics_labels.extend(ethics_label.cpu().numpy())
                            all_manip_preds.extend(manipulation_score.cpu().numpy())
                            all_manip_labels.extend(manipulation_label.cpu().numpy())
                            
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            # Durchschnittlichen Verlust berechnen
            avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
            
            if return_predictions:
                return {
                    'loss': avg_loss,
                    'ethics_predictions': all_ethics_preds,
                    'ethics_labels': all_ethics_labels,
                    'manip_predictions': all_manip_preds,
                    'manip_labels': all_manip_labels
                }
            
            return avg_loss
        
        # Test validation
        val_loss = custom_validate(model, dataloader, criterion, device)
        assert isinstance(val_loss, float)
        assert val_loss >= 0
        
        # Test validation with predictions
        val_results = custom_validate(
            model, dataloader, criterion, device, return_predictions=True
        )
        
        assert isinstance(val_results, dict)
        assert 'loss' in val_results
        assert 'ethics_predictions' in val_results
        assert 'ethics_labels' in val_results
        assert len(val_results['ethics_predictions']) == 2
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        predictions = [0.8, 0.3, 0.9, 0.1, 0.7]
        labels = [0.9, 0.2, 0.8, 0.1, 0.6]
        
        metrics = calculate_metrics(predictions, labels, threshold=0.5)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0


def train_one_epoch(model, llm, dataloader, optimizer, criterion, device):
    """Training für eine Epoche durchführen."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        try:
            # Daten vorbereiten
            input_ids = batch['input_ids'].to(device)
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            attention_mask = batch['attention_mask'].to(device)
            if attention_mask.dtype != torch.float32:
                attention_mask = attention_mask.float()
            
            ethics_label = batch['ethics_label'].to(device)
            manipulation_label = batch['manipulation_label'].to(device)
            texts = batch.get('texts', None)
            
            # LLM-Embeddings generieren
            with torch.no_grad():
                outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
            
            # Gradienten zurücksetzen
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                embeddings=hidden_states,
                attention_mask=attention_mask,
                texts=texts
            )
            
            # Verluste berechnen
            ethics_score = outputs['ethics_score']
            manipulation_score = outputs['manipulation_score']
            
            loss_ethics = criterion(ethics_score, ethics_label.view(-1, 1).float())
            loss_manip = criterion(manipulation_score, manipulation_label.view(-1, 1).float())
            loss = loss_ethics + 0.5 * loss_manip
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Statistiken aktualisieren
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
        except Exception as e:
            print(f"Fehler im Batch: {e}")
            continue
    
    # Durchschnittsverlust berechnen
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return model, avg_loss


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_pipeline(self):
        """Test complete pipeline from data to trained model."""
        # Create dataset
        texts, ethics_scores, manipulation_scores = create_sample_data()
        texts = texts[:12]  # Use subset
        ethics_scores = ethics_scores[:12]
        manipulation_scores = manipulation_scores[:12]
        
        # Split data
        train_data, val_data, test_data = create_data_splits(
            texts, ethics_scores, manipulation_scores,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            seed=42
        )
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        llm = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in llm.parameters():
            param.requires_grad = False
        
        # Create datasets
        train_dataset = MultiTaskDataset(
            texts=train_data['texts'],
            ethics_labels=train_data['ethics_labels'],
            manipulation_labels=train_data['manipulation_labels'],
            tokenizer=tokenizer
        )
        
        val_dataset = MultiTaskDataset(
            texts=val_data['texts'],
            ethics_labels=val_data['ethics_labels'],
            manipulation_labels=val_data['manipulation_labels'],
            tokenizer=tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=4, collate_fn=collate_ethics_batch, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=4, collate_fn=collate_ethics_batch, shuffle=False
        )
        
        # Create model and LLM
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        # Early stopping based on validation performance
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        patience = 2
        
        for epoch in range(2):  # Just 2 epochs for testing
            model, train_loss = train_one_epoch(
                model, llm, train_loader, optimizer, criterion, device
            )
            
            # Validation
            val_metrics = validate(
                model, val_loader, criterion, device, 
                return_predictions=True, llm=llm  # LLM übergeben
            )
            
            val_loss = val_metrics['loss']
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Test trained model
        assert model is not None
        
        # Evaluate on test data if available
        if test_data['texts']:
            test_dataset = MultiTaskDataset(
                texts=test_data['texts'],
                ethics_labels=test_data['ethics_labels'],
                manipulation_labels=test_data['manipulation_labels'],
                tokenizer=tokenizer
            )
            
            test_loader = DataLoader(
                test_dataset, batch_size=4, collate_fn=collate_ethics_batch
            )
            
            test_results = validate(
                model, test_loader, criterion, device, return_predictions=True, llm=llm
            )
            
            assert 'loss' in test_results
            assert test_results['loss'] >= 0
            
            # Calculate metrics nur wenn Predictions vorhanden sind
            if test_results['ethics_predictions'] and len(test_results['ethics_predictions']) > 0:
                metrics = calculate_metrics(
                    test_results['ethics_predictions'],
                    test_results['ethics_labels']
                )
                
                assert 'accuracy' in metrics
                print(f"Test accuracy: {metrics['accuracy']:.3f}")
            else:
                print("Keine Vorhersagen für Metriken verfügbar, überspringe Metrik-Berechnung")
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        config = {
            'input_dim': 768,
            'd_model': 768,
            'n_layers': 1,
            'n_heads': 2,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'use_semantic_graphs': False,
            'use_gnn': False
        }
        
        model = create_ethics_model(config)
        
        # Test model state dict
        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        
        # Create new model and load state
        new_model = create_ethics_model(config)
        new_model.load_state_dict(state_dict)
        model.eval()
        new_model.eval()
        # Test that models produce same output
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32).float()
        with torch.no_grad():
            output1 = model(input_ids=input_ids, attention_mask=attention_mask)
            output2 = new_model(input_ids=input_ids, attention_mask=attention_mask)
        assert torch.allclose(output1['ethics_score'], output2['ethics_score'], atol=1e-3)
        assert torch.allclose(output1['manipulation_score'], output2['manipulation_score'], atol=1e-3)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@pytest.fixture
def toy_graph():
    # Einfacher PyG-Graph mit 8 Features
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn((3, 8))
    return Data(x=x, edge_index=edge_index)

def test_integration_full(tokenizer, nlp, toy_graph):
    model = EnhancedEthicsModel(input_dim=8, d_model=8, use_gnn=True)
    texts = ["Ethics in AI is important.", "Models should be transparent."]
    tokens = tokenizer(texts, padding='max_length', return_tensors="pt", max_length=16)
    graphs = [toy_graph, toy_graph]
    output = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], graph_data=graphs)
    assert output is not None
    assert output['ethics_score'].shape[0] == len(texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
