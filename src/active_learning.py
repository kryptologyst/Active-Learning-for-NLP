"""
Active Learning for NLP using BERT and modern transformers.

This module implements an active learning pipeline for text classification tasks,
using uncertainty sampling to iteratively improve model performance.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearningDataset(Dataset):
    """Custom dataset class for active learning."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: Optional list of labels (None for unlabeled data)
        """
        self.texts = texts
        self.labels = labels if labels is not None else [-1] * len(texts)
        self.labeled_mask = [label != -1 for label in self.labels]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, int]]:
        return {
            'text': self.texts[idx],
            'labels': self.labels[idx],
            'is_labeled': self.labeled_mask[idx]
        }
    
    def add_labels(self, indices: List[int], labels: List[int]) -> None:
        """Add labels to previously unlabeled samples."""
        for idx, label in zip(indices, labels):
            if idx < len(self.labels):
                self.labels[idx] = label
                self.labeled_mask[idx] = True
    
    def get_unlabeled_indices(self) -> List[int]:
        """Get indices of unlabeled samples."""
        return [i for i, labeled in enumerate(self.labeled_mask) if not labeled]
    
    def get_labeled_data(self) -> Tuple[List[str], List[int]]:
        """Get all labeled data."""
        labeled_texts = [self.texts[i] for i, labeled in enumerate(self.labeled_mask) if labeled]
        labeled_labels = [self.labels[i] for i, labeled in enumerate(self.labeled_mask) if labeled]
        return labeled_texts, labeled_labels


class ActiveLearningPipeline:
    """Main active learning pipeline for NLP tasks."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the active learning pipeline.
        
        Args:
            model_name: Hugging Face model name
            num_labels: Number of classification labels
            device: Device to run on ('cuda', 'cpu', or None for auto)
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        # Initialize metrics
        self.metric = evaluate.load("accuracy")
        
        logger.info(f"Initialized ActiveLearningPipeline with {model_name} on {self.device}")
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize text examples."""
        return self.tokenizer(
            examples['text'], 
            padding="max_length", 
            truncation=True,
            max_length=512
        )
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def get_uncertainty_scores(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute uncertainty scores for unlabeled texts.
        
        Args:
            texts: List of unlabeled texts
            batch_size: Batch size for inference
            
        Returns:
            Array of uncertainty scores (higher = more uncertain)
        """
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # Use entropy as uncertainty measure
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
                uncertainties.extend(entropy.cpu().numpy())
        
        return np.array(uncertainties)
    
    def select_uncertain_samples(
        self, 
        dataset: ActiveLearningDataset, 
        num_samples: int = 5
    ) -> List[int]:
        """
        Select the most uncertain unlabeled samples.
        
        Args:
            dataset: Active learning dataset
            num_samples: Number of samples to select
            
        Returns:
            List of indices of selected samples
        """
        unlabeled_indices = dataset.get_unlabeled_indices()
        if not unlabeled_indices:
            logger.warning("No unlabeled samples available")
            return []
        
        unlabeled_texts = [dataset.texts[i] for i in unlabeled_indices]
        uncertainties = self.get_uncertainty_scores(unlabeled_texts)
        
        # Select most uncertain samples
        selected_local_indices = np.argsort(uncertainties)[-num_samples:]
        selected_global_indices = [unlabeled_indices[i] for i in selected_local_indices]
        
        return selected_global_indices
    
    def train_model(
        self, 
        dataset: ActiveLearningDataset,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model on labeled data.
        
        Args:
            dataset: Active learning dataset
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        labeled_texts, labeled_labels = dataset.get_labeled_data()
        
        if len(labeled_texts) < 2:
            logger.warning("Not enough labeled data for training")
            return {}
        
        # Create Hugging Face dataset
        hf_dataset = HFDataset.from_dict({
            'text': labeled_texts,
            'labels': labeled_labels
        })
        
        # Tokenize
        tokenized_dataset = hf_dataset.map(self.tokenize_function, batched=True)
        
        # Split into train/validation
        split_dataset = tokenized_dataset.train_test_split(test_size=validation_split)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/temp',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            seed=self.random_seed,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset['train'],
            eval_dataset=split_dataset['test'],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        logger.info(f"Training completed. F1 score: {eval_results.get('eval_f1', 0):.3f}")
        
        return eval_results
    
    def active_learning_loop(
        self,
        initial_dataset: ActiveLearningDataset,
        num_iterations: int = 5,
        samples_per_iteration: int = 5,
        epochs_per_iteration: int = 3,
        oracle_labels: Optional[List[int]] = None
    ) -> Tuple[ActiveLearningDataset, List[Dict[str, float]]]:
        """
        Run the active learning loop.
        
        Args:
            initial_dataset: Initial dataset with some labeled samples
            num_iterations: Number of active learning iterations
            samples_per_iteration: Number of samples to label per iteration
            epochs_per_iteration: Number of training epochs per iteration
            oracle_labels: Ground truth labels for simulation (optional)
            
        Returns:
            Tuple of (final_dataset, training_history)
        """
        dataset = initial_dataset
        training_history = []
        
        logger.info(f"Starting active learning loop with {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            logger.info(f"Active Learning Iteration {iteration + 1}/{num_iterations}")
            
            # Train model on current labeled data
            if len(dataset.get_labeled_data()[0]) > 0:
                metrics = self.train_model(dataset, epochs=epochs_per_iteration)
                training_history.append(metrics)
            
            # Select uncertain samples
            uncertain_indices = self.select_uncertain_samples(
                dataset, 
                num_samples=samples_per_iteration
            )
            
            if not uncertain_indices:
                logger.info("No more unlabeled samples available")
                break
            
            # Simulate oracle labeling (in practice, this would be human annotation)
            if oracle_labels is not None:
                # Use ground truth labels for simulation
                selected_labels = [oracle_labels[i] for i in uncertain_indices]
            else:
                # Generate random labels for demonstration
                selected_labels = [random.randint(0, self.num_labels - 1) 
                                 for _ in uncertain_indices]
            
            # Add labels to dataset
            dataset.add_labels(uncertain_indices, selected_labels)
            
            labeled_count = sum(dataset.labeled_mask)
            total_count = len(dataset)
            
            logger.info(
                f"Iteration {iteration + 1}: Added {len(uncertain_indices)} samples. "
                f"Total labeled: {labeled_count}/{total_count} "
                f"({labeled_count/total_count*100:.1f}%)"
            )
        
        return dataset, training_history
    
    def evaluate_model(
        self, 
        test_texts: List[str], 
        test_labels: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_texts: Test text samples
            test_labels: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(test_texts), 32):
                batch_texts = test_texts[i:i + 32]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return self.compute_metrics((predictions, test_labels))
