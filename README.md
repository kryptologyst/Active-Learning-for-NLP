# Active Learning for NLP

A comprehensive implementation of active learning for natural language processing tasks, featuring uncertainty sampling, state-of-the-art transformer models, and multiple interfaces for easy experimentation.

## Features

- **Modern Architecture**: Built with Hugging Face Transformers, PyTorch, and scikit-learn
- **Multiple Models**: Support for BERT, DistilBERT, RoBERTa, and other transformer models
- **Uncertainty Sampling**: Entropy-based uncertainty estimation for sample selection
- **Flexible Interfaces**: Web UI (Streamlit), CLI, and Python API
- **Real & Synthetic Data**: Support for real datasets (IMDB) and synthetic data generation
- **Comprehensive Logging**: Detailed experiment tracking and metrics
- **Configuration Management**: YAML/JSON configuration files
- **Visualization**: Interactive plots and progress tracking
- **Type Safety**: Full type hints and modern Python practices

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── active_learning.py  # Main active learning pipeline
│   ├── data_utils.py       # Data generation and management
│   └── config.py           # Configuration management
├── web_app/                # Streamlit web interface
│   └── app.py             # Main web application
├── config/                 # Configuration files
├── data/                   # Data storage
├── models/                 # Model checkpoints
├── results/                # Experiment results
├── tests/                  # Unit tests
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Active-Learning-for-NLP.git
   cd Active-Learning-for-NLP
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python cli.py --help
   ```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` and follow the interactive interface.

### Command Line Interface

Run a basic experiment:

```bash
python cli.py --iterations 5 --samples-per-iteration 5
```

Run with custom settings:

```bash
python cli.py \
  --model bert-base-uncased \
  --iterations 10 \
  --samples-per-iteration 3 \
  --epochs 5 \
  --dataset-size 1000
```

### Python API

```python
from src.active_learning import ActiveLearningPipeline, ActiveLearningDataset
from src.data_utils import create_demo_dataset

# Create dataset
texts, labels, initial_indices = create_demo_dataset()

# Initialize pipeline
pipeline = ActiveLearningPipeline(
    model_name="distilbert-base-uncased",
    num_labels=2
)

# Create dataset
dataset = ActiveLearningDataset(texts, labels)
dataset.add_labels(initial_indices, [labels[i] for i in initial_indices])

# Run active learning
final_dataset, training_history = pipeline.active_learning_loop(
    dataset,
    num_iterations=5,
    samples_per_iteration=5,
    oracle_labels=labels
)

print(f"Final F1 score: {training_history[-1]['eval_f1']:.3f}")
```

## Configuration

### Configuration Files

Create a configuration file (`config/custom.yaml`):

```yaml
model:
  name: "distilbert-base-uncased"
  num_labels: 2
  max_length: 512

training:
  epochs_per_iteration: 3
  learning_rate: 2e-5
  batch_size: 16
  validation_split: 0.2

active_learning:
  num_iterations: 5
  samples_per_iteration: 5
  initial_labeled_size: 20

data:
  dataset_name: "imdb"
  subset_size: 500
  use_synthetic: false
  random_seed: 42
```

Use the configuration:

```bash
python cli.py --config config/custom.yaml
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model to use | `distilbert-base-uncased` |
| `--iterations` | Number of AL iterations | `5` |
| `--samples-per-iteration` | Samples to label per iteration | `5` |
| `--epochs` | Training epochs per iteration | `3` |
| `--learning-rate` | Learning rate | `2e-5` |
| `--batch-size` | Batch size | `16` |
| `--dataset-size` | Dataset size | `500` |
| `--use-synthetic` | Use synthetic data | `False` |
| `--output-dir` | Results directory | `results/` |

## Active Learning Process

The active learning pipeline follows these steps:

1. **Initialization**: Start with a small set of labeled data
2. **Model Training**: Train the model on currently labeled data
3. **Uncertainty Estimation**: Compute uncertainty scores for unlabeled samples
4. **Sample Selection**: Select the most uncertain samples for labeling
5. **Labeling**: Simulate human annotation (oracle labels)
6. **Update**: Add newly labeled samples to training set
7. **Repeat**: Continue until stopping criteria met

### Uncertainty Sampling

The system uses **entropy-based uncertainty sampling**:

```python
# For each unlabeled sample, compute prediction probabilities
probabilities = model.predict_proba(text)

# Compute entropy (uncertainty)
entropy = -sum(p * log(p) for p in probabilities)

# Select samples with highest entropy (most uncertain)
```

## Experiments

### Basic Experiment

```bash
python cli.py --iterations 5 --samples-per-iteration 5
```

### Advanced Experiment

```bash
python cli.py \
  --model bert-base-uncased \
  --iterations 10 \
  --samples-per-iteration 3 \
  --epochs 5 \
  --learning-rate 1e-5 \
  --batch-size 32 \
  --dataset-size 1000 \
  --output-dir results/bert_experiment
```

### Synthetic Data Experiment

```bash
python cli.py \
  --use-synthetic \
  --dataset-size 2000 \
  --iterations 8 \
  --samples-per-iteration 10
```

## Results and Metrics

The system tracks comprehensive metrics:

- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **Labeling Progress**: Percentage of data labeled over time

### Output Files

- `training_history.json`: Metrics for each iteration
- `final_metrics.json`: Final performance metrics
- `dataset_info.json`: Dataset statistics
- `config.json`: Experiment configuration

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ web_app/ cli.py
flake8 src/ web_app/ cli.py
mypy src/
```

### Adding New Features

1. **New Models**: Extend `ActiveLearningPipeline` to support additional models
2. **New Sampling Strategies**: Implement different uncertainty sampling methods
3. **New Datasets**: Add dataset loaders in `data_utils.py`
4. **New Metrics**: Extend the metrics computation in the pipeline

## API Reference

### ActiveLearningPipeline

Main class for running active learning experiments.

```python
pipeline = ActiveLearningPipeline(
    model_name="distilbert-base-uncased",
    num_labels=2,
    device="auto",
    random_seed=42
)
```

**Methods**:
- `active_learning_loop()`: Run the complete active learning process
- `get_uncertainty_scores()`: Compute uncertainty for unlabeled samples
- `train_model()`: Train the model on labeled data
- `evaluate_model()`: Evaluate model performance

### ActiveLearningDataset

Custom dataset class for managing labeled and unlabeled data.

```python
dataset = ActiveLearningDataset(texts, labels)
```

**Methods**:
- `add_labels()`: Add labels to previously unlabeled samples
- `get_unlabeled_indices()`: Get indices of unlabeled samples
- `get_labeled_data()`: Get all labeled data

### DataManager

Utility class for data loading and management.

```python
data_manager = DataManager()
texts, labels = data_manager.load_real_dataset("imdb", subset_size=500)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The active learning research community for foundational algorithms

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation
- Review the example configurations
# Active-Learning-for-NLP
