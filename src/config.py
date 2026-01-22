"""
Configuration management for active learning experiments.
"""

import logging
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for active learning experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_default_config()
        
        if config_path:
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "model": {
                "name": "distilbert-base-uncased",
                "num_labels": 2,
                "max_length": 512,
                "device": "auto"
            },
            "training": {
                "epochs_per_iteration": 3,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "validation_split": 0.2,
                "early_stopping_patience": 3,
                "weight_decay": 0.01,
                "warmup_steps": 100
            },
            "active_learning": {
                "num_iterations": 5,
                "samples_per_iteration": 5,
                "initial_labeled_size": 20,
                "uncertainty_method": "entropy"
            },
            "data": {
                "dataset_name": "imdb",
                "subset_size": 500,
                "use_synthetic": False,
                "random_seed": 42
            },
            "logging": {
                "level": "INFO",
                "log_dir": "./logs",
                "log_steps": 10
            },
            "paths": {
                "models_dir": "./models",
                "data_dir": "./data",
                "results_dir": "./results"
            }
        }
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return
            
            # Merge with default config
            self._merge_config(file_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Recursively merge new configuration with existing."""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> None:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, new_config)
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.name')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._merge_config(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()


def create_default_config_file(config_path: str = "config/default.yaml") -> None:
    """Create a default configuration file."""
    config = Config()
    config.save_to_file(config_path)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)
