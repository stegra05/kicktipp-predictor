"""Kicktipp Predictor package."""

__version__ = "3.0.0a1"

# Import main classes for convenience
from .config import Config, get_config
from .data import DataLoader
from .predictor import GoalDifferencePredictor

__all__ = [
    "get_config",
    "Config",
    "DataLoader",
    "GoalDifferencePredictor",
]
