#!/usr/bin/env python3
"""
Command-line interface for Active Learning NLP experiments.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from active_learning import ActiveLearningPipeline, ActiveLearningDataset
from data_utils import DataManager, create_demo_dataset
from config import Config, create_default_config_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_experiment(args):
    """Set up the active learning experiment."""
    logger.info("Setting up active learning experiment...")
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Override config with command line arguments
    if args.model:
        config.set('model.name', args.model)
    if args.iterations:
        config.set('active_learning.num_iterations', args.iterations)
    if args.samples_per_iteration:
        config.set('active_learning.samples_per_iteration', args.samples_per_iteration)
    if args.epochs:
        config.set('training.epochs_per_iteration', args.epochs)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.dataset_size:
        config.set('data.subset_size', args.dataset_size)
    if args.use_synthetic:
        config.set('data.use_synthetic', True)
    
    # Initialize pipeline
    pipeline = ActiveLearningPipeline(
        model_name=config.get('model.name'),
        num_labels=config.get('model.num_labels'),
        random_seed=config.get('data.random_seed')
    )
    
    # Load dataset
    data_manager = DataManager(config.get('data.random_seed'))
    
    if config.get('data.use_synthetic'):
        logger.info("Using synthetic dataset")
        texts, labels = data_manager.generator.generate_sentiment_data(
            config.get('data.subset_size')
        )
    else:
        logger.info(f"Loading dataset: {config.get('data.dataset_name')}")
        texts, labels = data_manager.load_real_dataset(
            config.get('data.dataset_name'),
            config.get('data.subset_size')
        )
    
    # Create active learning dataset
    texts, labels, initial_indices = data_manager.create_active_learning_dataset(
        texts, labels, config.get('active_learning.initial_labeled_size')
    )
    
    dataset = ActiveLearningDataset(texts, labels)
    initial_labels = [labels[i] for i in initial_indices]
    dataset.add_labels(initial_indices, initial_labels)
    
    logger.info(f"Dataset created: {len(dataset)} total samples, {sum(dataset.labeled_mask)} initially labeled")
    
    return pipeline, dataset, labels, config


def run_experiment(pipeline, dataset, oracle_labels, config, args):
    """Run the active learning experiment."""
    logger.info("Starting active learning experiment...")
    
    start_time = time.time()
    
    # Run active learning loop
    final_dataset, training_history = pipeline.active_learning_loop(
        dataset,
        num_iterations=config.get('active_learning.num_iterations'),
        samples_per_iteration=config.get('active_learning.samples_per_iteration'),
        epochs_per_iteration=config.get('training.epochs_per_iteration'),
        oracle_labels=oracle_labels
    )
    
    end_time = time.time()
    experiment_time = end_time - start_time
    
    logger.info(f"Experiment completed in {experiment_time:.2f} seconds")
    
    return final_dataset, training_history, experiment_time


def save_results(final_dataset, training_history, experiment_time, config, args):
    """Save experiment results."""
    results_dir = Path(args.output_dir) if args.output_dir else Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history
    history_file = results_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save final metrics
    final_metrics = training_history[-1] if training_history else {}
    metrics_file = results_dir / "final_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'final_metrics': final_metrics,
            'experiment_time': experiment_time,
            'config': config.to_dict()
        }, f, indent=2)
    
    # Save dataset info
    dataset_info = {
        'total_samples': len(final_dataset),
        'labeled_samples': sum(final_dataset.labeled_mask),
        'labeling_percentage': sum(final_dataset.labeled_mask) / len(final_dataset) * 100
    }
    
    dataset_file = results_dir / "dataset_info.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")
    
    return results_dir


def print_results(training_history, experiment_time):
    """Print experiment results to console."""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"Total experiment time: {experiment_time:.2f} seconds")
    print(f"Number of iterations: {len(training_history)}")
    
    if training_history:
        print("\nTraining Progress:")
        print("-" * 40)
        
        for i, metrics in enumerate(training_history):
            print(f"Iteration {i+1}:")
            print(f"  F1 Score: {metrics.get('eval_f1', 0):.3f}")
            print(f"  Accuracy: {metrics.get('eval_accuracy', 0):.3f}")
            print(f"  Precision: {metrics.get('eval_precision', 0):.3f}")
            print(f"  Recall: {metrics.get('eval_recall', 0):.3f}")
            print()
        
        # Final results
        final_metrics = training_history[-1]
        print("Final Results:")
        print("-" * 20)
        print(f"F1 Score: {final_metrics.get('eval_f1', 0):.3f}")
        print(f"Accuracy: {final_metrics.get('eval_accuracy', 0):.3f}")
        print(f"Precision: {final_metrics.get('eval_precision', 0):.3f}")
        print(f"Recall: {final_metrics.get('eval_recall', 0):.3f}")
    
    print("="*60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Active Learning for NLP - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python cli.py

  # Run with custom model and iterations
  python cli.py --model bert-base-uncased --iterations 10

  # Run with synthetic data
  python cli.py --use-synthetic --dataset-size 1000

  # Run with custom configuration file
  python cli.py --config config/custom.yaml

  # Create default configuration file
  python cli.py --create-config
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    # Model options
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base'],
        help='Model to use for classification'
    )
    
    # Training options
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of epochs per iteration'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate for training'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for training'
    )
    
    # Active learning options
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        help='Number of active learning iterations'
    )
    
    parser.add_argument(
        '--samples-per-iteration', '-s',
        type=int,
        help='Number of samples to label per iteration'
    )
    
    # Data options
    parser.add_argument(
        '--dataset-size', '-d',
        type=int,
        help='Size of the dataset to use'
    )
    
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic data instead of real dataset'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle create-config option
    if args.create_config:
        config_path = args.config or "config/default.yaml"
        create_default_config_file(config_path)
        print(f"Default configuration file created: {config_path}")
        return
    
    try:
        # Setup experiment
        pipeline, dataset, oracle_labels, config = setup_experiment(args)
        
        # Run experiment
        final_dataset, training_history, experiment_time = run_experiment(
            pipeline, dataset, oracle_labels, config, args
        )
        
        # Save results
        results_dir = save_results(final_dataset, training_history, experiment_time, config, args)
        
        # Print results
        print_results(training_history, experiment_time)
        
        print(f"\nResults saved to: {results_dir}")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
