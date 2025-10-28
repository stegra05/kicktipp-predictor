"""Kicktipp Predictor package."""

__version__ = "4.0.0a2"

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

# Optional web app factory re-export
try:  # pragma: no cover - optional import
    from .web import create_app as create_flask_app  # type: ignore
    __all__.append("create_flask_app")
except Exception:  # pragma: no cover
    pass
