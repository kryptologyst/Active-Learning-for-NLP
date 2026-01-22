"""
Data generation and management utilities for active learning experiments.
"""

import logging
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
from datasets import load_dataset, Dataset as HFDataset
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic text classification datasets."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator."""
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def generate_text_classification_data(
        self,
        n_samples: int = 1000,
        n_classes: int = 2,
        n_features: int = 100,
        class_sep: float = 1.0,
        noise: float = 0.1
    ) -> Tuple[List[str], List[int]]:
        """
        Generate synthetic text classification data.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            n_features: Number of features for underlying classification
            class_sep: Class separation (higher = easier classification)
            noise: Amount of noise in the data
            
        Returns:
            Tuple of (texts, labels)
        """
        # Generate underlying classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            class_sep=class_sep,
            random_state=self.random_seed
        )
        
        # Convert features to text-like representations
        texts = []
        for i in range(n_samples):
            # Create text based on feature values
            features = X[i]
            text_parts = []
            
            for j, feature_val in enumerate(features):
                if abs(feature_val) > 0.5:  # Only include significant features
                    if feature_val > 0:
                        text_parts.append(f"feature_{j}_positive")
                    else:
                        text_parts.append(f"feature_{j}_negative")
            
            # Add some noise words
            noise_words = ["common_word", "frequent_term", "general_text"]
            if random.random() < noise:
                text_parts.append(random.choice(noise_words))
            
            # Combine into a sentence
            if text_parts:
                text = " ".join(text_parts)
            else:
                text = "neutral_text"
            
            texts.append(text)
        
        return texts, y.tolist()
    
    def generate_sentiment_data(self, n_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Generate synthetic sentiment analysis data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (texts, labels) where labels are 0 (negative) or 1 (positive)
        """
        positive_templates = [
            "This is amazing and wonderful",
            "I love this product so much",
            "Excellent quality and great service",
            "Fantastic experience overall",
            "Highly recommended to everyone",
            "Outstanding performance and results",
            "Perfect solution for my needs",
            "Incredible value for money",
            "Top-notch quality and design",
            "Absolutely brilliant and innovative"
        ]
        
        negative_templates = [
            "This is terrible and awful",
            "I hate this product completely",
            "Poor quality and bad service",
            "Disappointing experience overall",
            "Not recommended to anyone",
            "Underwhelming performance and results",
            "Wrong solution for my needs",
            "Waste of money and time",
            "Low-quality design and build",
            "Completely useless and broken"
        ]
        
        texts = []
        labels = []
        
        for _ in range(n_samples):
            if random.random() < 0.5:
                # Generate positive sample
                template = random.choice(positive_templates)
                label = 1
            else:
                # Generate negative sample
                template = random.choice(negative_templates)
                label = 0
            
            # Add some variation
            variations = [
                template,
                template + " and very satisfied",
                template + " with great results",
                template + " highly recommended",
                template + " excellent choice"
            ]
            
            text = random.choice(variations)
            texts.append(text)
            labels.append(label)
        
        return texts, labels


class DataManager:
    """Manage datasets for active learning experiments."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data manager."""
        self.random_seed = random_seed
        self.generator = SyntheticDataGenerator(random_seed)
    
    def load_real_dataset(
        self, 
        dataset_name: str = "imdb", 
        subset_size: Optional[int] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Load a real dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset to load
            subset_size: Optional size limit for the dataset
            
        Returns:
            Tuple of (texts, labels)
        """
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")
            
            if subset_size:
                dataset = dataset.select(range(min(subset_size, len(dataset))))
            
            texts = dataset['text']
            labels = dataset['label']
            
            logger.info(f"Loaded {len(texts)} samples from {dataset_name}")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            logger.info("Falling back to synthetic data")
            return self.generator.generate_sentiment_data(1000)
    
    def create_active_learning_dataset(
        self,
        texts: List[str],
        labels: List[int],
        initial_labeled_size: int = 20,
        random_seed: Optional[int] = None
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Create an active learning dataset with initial labeled samples.
        
        Args:
            texts: All text samples
            labels: All labels
            initial_labeled_size: Number of initially labeled samples
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (all_texts, all_labels, initial_labeled_indices)
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Shuffle data
        indices = list(range(len(texts)))
        random.shuffle(indices)
        
        shuffled_texts = [texts[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]
        
        # Select initial labeled samples
        initial_indices = list(range(initial_labeled_size))
        
        logger.info(
            f"Created active learning dataset with {len(texts)} total samples, "
            f"{initial_labeled_size} initially labeled"
        )
        
        return shuffled_texts, shuffled_labels, initial_indices
    
    def save_dataset(
        self, 
        texts: List[str], 
        labels: List[int], 
        filepath: str
    ) -> None:
        """Save dataset to file."""
        import json
        
        data = {
            'texts': texts,
            'labels': labels
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved dataset to {filepath}")
    
    def load_dataset(self, filepath: str) -> Tuple[List[str], List[int]]:
        """Load dataset from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded dataset from {filepath}")
        return data['texts'], data['labels']


def create_demo_dataset() -> Tuple[List[str], List[int], List[int]]:
    """
    Create a demo dataset for active learning experiments.
    
    Returns:
        Tuple of (texts, labels, initial_labeled_indices)
    """
    data_manager = DataManager()
    
    # Try to load a real dataset, fall back to synthetic
    try:
        texts, labels = data_manager.load_real_dataset("imdb", subset_size=500)
    except:
        logger.info("Using synthetic sentiment data")
        texts, labels = data_manager.generate_sentiment_data(500)
    
    # Create active learning setup
    texts, labels, initial_indices = data_manager.create_active_learning_dataset(
        texts, labels, initial_labeled_size=20
    )
    
    return texts, labels, initial_indices
