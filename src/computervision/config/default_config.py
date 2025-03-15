"""
Default configuration settings for the computer vision package.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
DEFAULT_MODEL_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "device": "cuda",  # or "cpu"
}

# Data configurations
DATA_CONFIG = {
    "train_test_split": 0.2,
    "random_seed": 42,
    "img_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
} 