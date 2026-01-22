"""
Streamlit web interface for Active Learning NLP experiments.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from active_learning import ActiveLearningPipeline, ActiveLearningDataset
from data_utils import DataManager, create_demo_dataset
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Active Learning for NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'config' not in st.session_state:
        st.session_state.config = Config()

def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model configuration
    st.sidebar.subheader("🤖 Model Settings")
    model_name = st.sidebar.selectbox(
        "Model",
        ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
        index=0
    )
    
    # Training configuration
    st.sidebar.subheader("🎯 Training Settings")
    epochs = st.sidebar.slider("Epochs per iteration", 1, 10, 3)
    learning_rate = st.sidebar.selectbox(
        "Learning Rate",
        [1e-5, 2e-5, 5e-5, 1e-4],
        index=1,
        format_func=lambda x: f"{x:.0e}"
    )
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32], index=1)
    
    # Active learning configuration
    st.sidebar.subheader("🔄 Active Learning Settings")
    num_iterations = st.sidebar.slider("Number of iterations", 1, 10, 5)
    samples_per_iteration = st.sidebar.slider("Samples per iteration", 1, 20, 5)
    initial_labeled_size = st.sidebar.slider("Initial labeled samples", 10, 100, 20)
    
    # Data configuration
    st.sidebar.subheader("📊 Data Settings")
    dataset_size = st.sidebar.slider("Dataset size", 100, 1000, 500)
    use_synthetic = st.sidebar.checkbox("Use synthetic data", value=False)
    
    return {
        'model_name': model_name,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'samples_per_iteration': samples_per_iteration,
        'initial_labeled_size': initial_labeled_size,
        'dataset_size': dataset_size,
        'use_synthetic': use_synthetic
    }

def load_dataset(config):
    """Load and prepare the dataset."""
    with st.spinner("Loading dataset..."):
        data_manager = DataManager()
        
        if config['use_synthetic']:
            texts, labels = data_manager.generator.generate_sentiment_data(config['dataset_size'])
        else:
            try:
                texts, labels = data_manager.load_real_dataset("imdb", config['dataset_size'])
            except:
                st.warning("Failed to load IMDB dataset, using synthetic data instead")
                texts, labels = data_manager.generator.generate_sentiment_data(config['dataset_size'])
        
        # Create active learning dataset
        texts, labels, initial_indices = data_manager.create_active_learning_dataset(
            texts, labels, config['initial_labeled_size']
        )
        
        # Create dataset object
        dataset = ActiveLearningDataset(texts, labels)
        
        # Mark initial samples as labeled
        initial_labels = [labels[i] for i in initial_indices]
        dataset.add_labels(initial_indices, initial_labels)
        
        return dataset, labels

def display_dataset_info(dataset):
    """Display dataset information."""
    labeled_count = sum(dataset.labeled_mask)
    total_count = len(dataset)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", total_count)
    
    with col2:
        st.metric("Labeled Samples", labeled_count)
    
    with col3:
        st.metric("Labeling Progress", f"{labeled_count/total_count*100:.1f}%")
    
    # Progress bar
    progress = labeled_count / total_count
    st.progress(progress)

def display_sample_data(dataset, num_samples=5):
    """Display sample data."""
    st.subheader("📝 Sample Data")
    
    # Get labeled samples
    labeled_texts, labeled_labels = dataset.get_labeled_data()
    
    if labeled_texts:
        sample_data = []
        for i, (text, label) in enumerate(zip(labeled_texts[:num_samples], labeled_labels[:num_samples])):
            sample_data.append({
                'Index': i,
                'Text': text[:100] + "..." if len(text) > 100 else text,
                'Label': 'Positive' if label == 1 else 'Negative',
                'Full Text': text
            })
        
        df = pd.DataFrame(sample_data)
        
        # Display table
        st.dataframe(df[['Index', 'Text', 'Label']], use_container_width=True)
        
        # Show full text for selected sample
        if len(sample_data) > 0:
            selected_idx = st.selectbox("Select sample to view full text:", range(len(sample_data)))
            with st.expander("Full Text"):
                st.text(sample_data[selected_idx]['Full Text'])

def run_active_learning(pipeline, dataset, config, oracle_labels):
    """Run the active learning loop."""
    with st.spinner("Running active learning..."):
        # Update pipeline configuration
        pipeline.random_seed = 42
        
        # Run active learning
        final_dataset, training_history = pipeline.active_learning_loop(
            dataset,
            num_iterations=config['num_iterations'],
            samples_per_iteration=config['samples_per_iteration'],
            epochs_per_iteration=config['epochs'],
            oracle_labels=oracle_labels
        )
        
        return final_dataset, training_history

def plot_training_history(training_history):
    """Plot training history."""
    if not training_history:
        st.warning("No training history available")
        return
    
    st.subheader("📈 Training Progress")
    
    # Extract metrics
    iterations = list(range(1, len(training_history) + 1))
    f1_scores = [h.get('eval_f1', 0) for h in training_history]
    accuracies = [h.get('eval_accuracy', 0) for h in training_history]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('F1 Score', 'Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # F1 Score plot
    fig.add_trace(
        go.Scatter(x=iterations, y=f1_scores, mode='lines+markers', name='F1 Score'),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=iterations, y=accuracies, mode='lines+markers', name='Accuracy'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Score", range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)

def plot_labeling_progress(dataset_history):
    """Plot labeling progress over iterations."""
    if not dataset_history:
        return
    
    st.subheader("📊 Labeling Progress")
    
    iterations = list(range(len(dataset_history)))
    labeled_counts = [sum(d.labeled_mask) for d in dataset_history]
    total_counts = [len(d) for d in dataset_history]
    percentages = [labeled/total*100 for labeled, total in zip(labeled_counts, total_counts)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=labeled_counts,
        mode='lines+markers',
        name='Labeled Samples',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=total_counts,
        mode='lines+markers',
        name='Total Samples',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    fig.update_layout(
        title="Labeled vs Total Samples Over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Number of Samples",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Active Learning for NLP</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application demonstrates active learning for text classification tasks. 
    The system iteratively selects the most uncertain samples for labeling, 
    improving model performance with minimal human annotation effort.
    """)
    
    # Sidebar configuration
    config = create_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Setup", "📊 Dataset", "🔄 Active Learning", "📈 Results"])
    
    with tab1:
        st.header("Setup and Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Configuration")
            config_df = pd.DataFrame([
                ["Model", config['model_name']],
                ["Epochs per iteration", config['epochs']],
                ["Learning rate", f"{config['learning_rate']:.0e}"],
                ["Batch size", config['batch_size']],
                ["Number of iterations", config['num_iterations']],
                ["Samples per iteration", config['samples_per_iteration']],
                ["Initial labeled samples", config['initial_labeled_size']],
                ["Dataset size", config['dataset_size']],
                ["Data type", "Synthetic" if config['use_synthetic'] else "Real (IMDB)"]
            ], columns=["Parameter", "Value"])
            
            st.dataframe(config_df, use_container_width=True)
        
        with col2:
            st.subheader("Actions")
            
            if st.button("🔄 Initialize Pipeline", type="primary"):
                with st.spinner("Initializing pipeline..."):
                    st.session_state.pipeline = ActiveLearningPipeline(
                        model_name=config['model_name'],
                        num_labels=2,
                        random_seed=42
                    )
                    st.success("Pipeline initialized successfully!")
            
            if st.button("📊 Load Dataset"):
                st.session_state.dataset, oracle_labels = load_dataset(config)
                st.session_state.oracle_labels = oracle_labels
                st.success("Dataset loaded successfully!")
    
    with tab2:
        st.header("Dataset Information")
        
        if st.session_state.dataset is not None:
            display_dataset_info(st.session_state.dataset)
            st.divider()
            display_sample_data(st.session_state.dataset)
        else:
            st.info("Please load a dataset first in the Setup tab.")
    
    with tab3:
        st.header("Active Learning Experiment")
        
        if st.session_state.pipeline is None:
            st.warning("Please initialize the pipeline first in the Setup tab.")
        elif st.session_state.dataset is None:
            st.warning("Please load a dataset first in the Setup tab.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Run Active Learning")
                st.write("Click the button below to start the active learning process.")
                
                if st.button("🚀 Start Active Learning", type="primary"):
                    final_dataset, training_history = run_active_learning(
                        st.session_state.pipeline,
                        st.session_state.dataset,
                        config,
                        st.session_state.oracle_labels
                    )
                    
                    st.session_state.final_dataset = final_dataset
                    st.session_state.training_history = training_history
                    
                    st.success("Active learning completed successfully!")
            
            with col2:
                st.subheader("Quick Stats")
                if st.session_state.dataset is not None:
                    labeled_count = sum(st.session_state.dataset.labeled_mask)
                    total_count = len(st.session_state.dataset)
                    st.metric("Current Labeled", f"{labeled_count}/{total_count}")
                    st.metric("Progress", f"{labeled_count/total_count*100:.1f}%")
    
    with tab4:
        st.header("Results and Analysis")
        
        if hasattr(st.session_state, 'training_history') and st.session_state.training_history:
            plot_training_history(st.session_state.training_history)
            st.divider()
            
            # Final metrics
            st.subheader("📊 Final Results")
            final_metrics = st.session_state.training_history[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final F1 Score", f"{final_metrics.get('eval_f1', 0):.3f}")
            
            with col2:
                st.metric("Final Accuracy", f"{final_metrics.get('eval_accuracy', 0):.3f}")
            
            with col3:
                st.metric("Final Precision", f"{final_metrics.get('eval_precision', 0):.3f}")
            
            with col4:
                st.metric("Final Recall", f"{final_metrics.get('eval_recall', 0):.3f}")
            
            # Detailed metrics table
            st.subheader("📋 Detailed Metrics")
            metrics_df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(metrics_df, use_container_width=True)
            
        else:
            st.info("No results available. Please run the active learning experiment first.")

if __name__ == "__main__":
    main()
