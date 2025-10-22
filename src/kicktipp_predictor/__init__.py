"""Kicktipp Predictor package."""

__version__ = "2.0.0"

# Import main classes for convenience
from .config import Config, get_config
from .data import DataLoader
from .predictor import MatchPredictor

__all__ = [
    "get_config",
    "Config",
    "DataLoader",
    "MatchPredictor",
]
