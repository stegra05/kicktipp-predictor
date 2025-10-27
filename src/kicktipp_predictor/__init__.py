"""Kicktipp Predictor package.

Version: 4.0.0a1 (alpha)

Changes in this release:
- Switch to V4 cascaded predictor (Draw → Win classifiers).
- CLI and evaluation updated to reflect cascaded architecture.
- V3 goal-difference regressor deprecated; see BLUEPRINT.md for migration notes.
"""

__version__ = "4.0.0a1"

# Import main classes for convenience
from .config import Config, get_config
from .data import DataLoader
from .predictor import CascadedPredictor

__all__ = [
    "get_config",
    "Config",
    "DataLoader",
    "CascadedPredictor",
]
