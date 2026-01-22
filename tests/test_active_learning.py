"""
Unit tests for Active Learning NLP components.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from active_learning import ActiveLearningPipeline, ActiveLearningDataset
from data_utils import SyntheticDataGenerator, DataManager
from config import Config


class TestActiveLearningDataset:
    """Test cases for ActiveLearningDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        
        dataset = ActiveLearningDataset(texts, labels)
        
        assert len(dataset) == 3
        assert dataset.texts == texts
        assert dataset.labels == labels
        assert all(dataset.labeled_mask)
    
    def test_dataset_unlabeled_initialization(self):
        """Test dataset initialization with unlabeled data."""
        texts = ["text1", "text2", "text3"]
        
        dataset = ActiveLearningDataset(texts)
        
        assert len(dataset) == 3
        assert dataset.texts == texts
        assert dataset.labels == [-1, -1, -1]
        assert not any(dataset.labeled_mask)
    
    def test_add_labels(self):
        """Test adding labels to dataset."""
        texts = ["text1", "text2", "text3"]
        dataset = ActiveLearningDataset(texts)
        
        dataset.add_labels([0, 2], [1, 0])
        
        assert dataset.labels[0] == 1
        assert dataset.labels[1] == -1
        assert dataset.labels[2] == 0
        assert dataset.labeled_mask[0] == True
        assert dataset.labeled_mask[1] == False
        assert dataset.labeled_mask[2] == True
    
    def test_get_unlabeled_indices(self):
        """Test getting unlabeled indices."""
        texts = ["text1", "text2", "text3", "text4"]
        labels = [0, -1, 1, -1]
        
        dataset = ActiveLearningDataset(texts, labels)
        
        unlabeled_indices = dataset.get_unlabeled_indices()
        assert unlabeled_indices == [1, 3]
    
    def test_get_labeled_data(self):
        """Test getting labeled data."""
        texts = ["text1", "text2", "text3", "text4"]
        labels = [0, -1, 1, -1]
        
        dataset = ActiveLearningDataset(texts, labels)
        
        labeled_texts, labeled_labels = dataset.get_labeled_data()
        assert labeled_texts == ["text1", "text3"]
        assert labeled_labels == [0, 1]


class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(random_seed=42)
        assert generator.random_seed == 42
    
    def test_generate_sentiment_data(self):
        """Test sentiment data generation."""
        generator = SyntheticDataGenerator(random_seed=42)
        texts, labels = generator.generate_sentiment_data(10)
        
        assert len(texts) == 10
        assert len(labels) == 10
        assert all(label in [0, 1] for label in labels)
        assert all(isinstance(text, str) for text in texts)
    
    def test_generate_text_classification_data(self):
        """Test text classification data generation."""
        generator = SyntheticDataGenerator(random_seed=42)
        texts, labels = generator.generate_text_classification_data(
            n_samples=20, n_classes=2
        )
        
        assert len(texts) == 20
        assert len(labels) == 20
        assert all(label in [0, 1] for label in labels)
        assert all(isinstance(text, str) for text in texts)


class TestDataManager:
    """Test cases for DataManager."""
    
    def test_manager_initialization(self):
        """Test data manager initialization."""
        manager = DataManager(random_seed=42)
        assert manager.random_seed == 42
        assert manager.generator is not None
    
    def test_create_active_learning_dataset(self):
        """Test creating active learning dataset."""
        manager = DataManager(random_seed=42)
        
        texts = ["text1", "text2", "text3", "text4", "text5"]
        labels = [0, 1, 0, 1, 0]
        
        texts_out, labels_out, initial_indices = manager.create_active_learning_dataset(
            texts, labels, initial_labeled_size=2
        )
        
        assert len(texts_out) == 5
        assert len(labels_out) == 5
        assert len(initial_indices) == 2
        assert initial_indices == [0, 1]


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = Config()
        assert config.config is not None
        assert config.get('model.name') == "distilbert-base-uncased"
    
    def test_config_get_set(self):
        """Test config get and set methods."""
        config = Config()
        
        # Test get with default
        value = config.get('nonexistent.key', 'default')
        assert value == 'default'
        
        # Test set and get
        config.set('test.key', 'test_value')
        assert config.get('test.key') == 'test_value'
    
    def test_config_update(self):
        """Test config update method."""
        config = Config()
        
        updates = {
            'model': {
                'name': 'bert-base-uncased'
            },
            'training': {
                'epochs_per_iteration': 5
            }
        }
        
        config.update(updates)
        
        assert config.get('model.name') == 'bert-base-uncased'
        assert config.get('training.epochs_per_iteration') == 5


class TestActiveLearningPipeline:
    """Test cases for ActiveLearningPipeline."""
    
    @patch('src.active_learning.AutoTokenizer')
    @patch('src.active_learning.AutoModelForSequenceClassification')
    def test_pipeline_initialization(self, mock_model, mock_tokenizer):
        """Test pipeline initialization."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        pipeline = ActiveLearningPipeline(
            model_name="distilbert-base-uncased",
            num_labels=2,
            random_seed=42
        )
        
        assert pipeline.model_name == "distilbert-base-uncased"
        assert pipeline.num_labels == 2
        assert pipeline.random_seed == 42
    
    def test_tokenize_function(self):
        """Test tokenization function."""
        pipeline = ActiveLearningPipeline(
            model_name="distilbert-base-uncased",
            num_labels=2
        )
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }
        pipeline.tokenizer = mock_tokenizer
        
        examples = {'text': ['test text']}
        result = pipeline.tokenize_function(examples)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result


if __name__ == "__main__":
    pytest.main([__file__])
