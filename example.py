#!/usr/bin/env python3
"""
Example script demonstrating Active Learning for NLP.

This script shows how to use the active learning pipeline
with different configurations and datasets.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from active_learning import ActiveLearningPipeline, ActiveLearningDataset
from data_utils import DataManager, create_demo_dataset
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_experiment():
    """Run a basic active learning experiment."""
    logger.info("Running basic active learning experiment...")
    
    # Create demo dataset
    texts, labels, initial_indices = create_demo_dataset()
    
    # Initialize pipeline
    pipeline = ActiveLearningPipeline(
        model_name="distilbert-base-uncased",
        num_labels=2,
        random_seed=42
    )
    
    # Create dataset
    dataset = ActiveLearningDataset(texts, labels)
    initial_labels = [labels[i] for i in initial_indices]
    dataset.add_labels(initial_indices, initial_labels)
    
    logger.info(f"Dataset: {len(dataset)} total samples, {sum(dataset.labeled_mask)} initially labeled")
    
    # Run active learning
    final_dataset, training_history = pipeline.active_learning_loop(
        dataset,
        num_iterations=3,
        samples_per_iteration=5,
        epochs_per_iteration=2,
        oracle_labels=labels
    )
    
    # Print results
    if training_history:
        final_metrics = training_history[-1]
        logger.info(f"Final F1 Score: {final_metrics.get('eval_f1', 0):.3f}")
        logger.info(f"Final Accuracy: {final_metrics.get('eval_accuracy', 0):.3f}")
    
    logger.info(f"Final labeled samples: {sum(final_dataset.labeled_mask)}/{len(final_dataset)}")
    
    return final_dataset, training_history


def example_synthetic_data_experiment():
    """Run experiment with synthetic data."""
    logger.info("Running synthetic data experiment...")
    
    # Create data manager
    data_manager = DataManager(random_seed=42)
    
    # Generate synthetic sentiment data
    texts, labels = data_manager.generator.generate_sentiment_data(200)
    
    # Create active learning dataset
    texts, labels, initial_indices = data_manager.create_active_learning_dataset(
        texts, labels, initial_labeled_size=15
    )
    
    # Initialize pipeline
    pipeline = ActiveLearningPipeline(
        model_name="distilbert-base-uncased",
        num_labels=2,
        random_seed=42
    )
    
    # Create dataset
    dataset = ActiveLearningDataset(texts, labels)
    initial_labels = [labels[i] for i in initial_indices]
    dataset.add_labels(initial_indices, initial_labels)
    
    logger.info(f"Synthetic dataset: {len(dataset)} total samples, {sum(dataset.labeled_mask)} initially labeled")
    
    # Run active learning
    final_dataset, training_history = pipeline.active_learning_loop(
        dataset,
        num_iterations=4,
        samples_per_iteration=8,
        epochs_per_iteration=3,
        oracle_labels=labels
    )
    
    # Print results
    if training_history:
        final_metrics = training_history[-1]
        logger.info(f"Synthetic Data - Final F1 Score: {final_metrics.get('eval_f1', 0):.3f}")
        logger.info(f"Synthetic Data - Final Accuracy: {final_metrics.get('eval_accuracy', 0):.3f}")
    
    return final_dataset, training_history


def example_configuration_experiment():
    """Run experiment with custom configuration."""
    logger.info("Running configuration-based experiment...")
    
    # Load configuration
    config = Config()
    
    # Override some settings
    config.set('model.name', 'distilbert-base-uncased')
    config.set('active_learning.num_iterations', 3)
    config.set('active_learning.samples_per_iteration', 6)
    config.set('training.epochs_per_iteration', 2)
    config.set('data.subset_size', 300)
    
    # Create dataset
    data_manager = DataManager(config.get('data.random_seed'))
    
    try:
        texts, labels = data_manager.load_real_dataset(
            config.get('data.dataset_name'),
            config.get('data.subset_size')
        )
    except:
        logger.info("Falling back to synthetic data")
        texts, labels = data_manager.generator.generate_sentiment_data(
            config.get('data.subset_size')
        )
    
    # Create active learning dataset
    texts, labels, initial_indices = data_manager.create_active_learning_dataset(
        texts, labels, config.get('active_learning.initial_labeled_size')
    )
    
    # Initialize pipeline
    pipeline = ActiveLearningPipeline(
        model_name=config.get('model.name'),
        num_labels=config.get('model.num_labels'),
        random_seed=config.get('data.random_seed')
    )
    
    # Create dataset
    dataset = ActiveLearningDataset(texts, labels)
    initial_labels = [labels[i] for i in initial_indices]
    dataset.add_labels(initial_indices, initial_labels)
    
    logger.info(f"Config experiment: {len(dataset)} total samples, {sum(dataset.labeled_mask)} initially labeled")
    
    # Run active learning
    final_dataset, training_history = pipeline.active_learning_loop(
        dataset,
        num_iterations=config.get('active_learning.num_iterations'),
        samples_per_iteration=config.get('active_learning.samples_per_iteration'),
        epochs_per_iteration=config.get('training.epochs_per_iteration'),
        oracle_labels=labels
    )
    
    # Print results
    if training_history:
        final_metrics = training_history[-1]
        logger.info(f"Config Experiment - Final F1 Score: {final_metrics.get('eval_f1', 0):.3f}")
        logger.info(f"Config Experiment - Final Accuracy: {final_metrics.get('eval_accuracy', 0):.3f}")
    
    return final_dataset, training_history


def main():
    """Run example experiments."""
    logger.info("Starting Active Learning NLP Examples")
    logger.info("=" * 50)
    
    try:
        # Run basic experiment
        logger.info("\n1. Basic Experiment")
        logger.info("-" * 20)
        example_basic_experiment()
        
        # Run synthetic data experiment
        logger.info("\n2. Synthetic Data Experiment")
        logger.info("-" * 30)
        example_synthetic_data_experiment()
        
        # Run configuration experiment
        logger.info("\n3. Configuration Experiment")
        logger.info("-" * 30)
        example_configuration_experiment()
        
        logger.info("\n" + "=" * 50)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
