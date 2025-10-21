"""Configuration management for kicktipp predictor.

This module centralizes all configuration parameters including paths,
model hyperparameters, and API settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class PathConfig:
    """File system paths configuration."""

    # Data directories
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = data_dir / "models"
    cache_dir: Path = data_dir / "cache"

    # Configuration files
    config_dir: Path = PROJECT_ROOT / "config"

    def __post_init__(self):
        """Ensure all directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def outcome_model_path(self) -> Path:
        """Path to the outcome classifier model."""
        return self.models_dir / "outcome_classifier.joblib"

    @property
    def home_goals_model_path(self) -> Path:
        """Path to the home goals regressor model."""
        return self.models_dir / "home_goals_regressor.joblib"

    @property
    def away_goals_model_path(self) -> Path:
        """Path to the away goals regressor model."""
        return self.models_dir / "away_goals_regressor.joblib"


@dataclass
class APIConfig:
    """API and data fetching configuration."""

    base_url: str = "https://api.openligadb.de"
    league_code: str = "bl3"  # 3. Liga
    cache_ttl: int = 3600  # Cache time-to-live in seconds (1 hour)
    request_timeout: int = 10  # HTTP request timeout in seconds


@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration."""

    # XGBoost Outcome Classifier (H/D/A)
    outcome_n_estimators: int = 200
    outcome_max_depth: int = 6
    outcome_learning_rate: float = 0.1
    outcome_subsample: float = 0.8
    outcome_colsample_bytree: float = 0.8

    # XGBoost Goal Regressors
    goals_n_estimators: int = 200
    goals_max_depth: int = 6
    goals_learning_rate: float = 0.1
    goals_subsample: float = 0.8
    goals_colsample_bytree: float = 0.8

    # Poisson grid for scoreline selection
    max_goals: int = 8
    min_lambda: float = 0.2  # Minimum expected goals to avoid degenerate predictions

    # Training
    random_state: int = 42
    test_size: float = 0.2
    min_training_matches: int = 50

    # Threading
    n_jobs: int = field(default_factory=lambda: max(1, int(os.getenv("OMP_NUM_THREADS", "0")) or os.cpu_count() or 1))

    # Class weights for outcome classifier
    # Boost draw class since it's underrepresented
    draw_boost: float = 1.5


@dataclass
class Config:
    """Main configuration container."""

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "Config":
        """Load configuration from file if available.

        Args:
            config_file: Path to config YAML file. If None, uses default location.

        Returns:
            Config instance with loaded or default values.
        """
        config = cls()

        if config_file is None:
            config_file = config.paths.config_dir / "best_params.yaml"

        # Try to load from YAML if available
        if yaml is not None and config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    params = yaml.safe_load(f)

                if isinstance(params, dict):
                    # Load model parameters if present
                    if "max_goals" in params:
                        config.model.max_goals = int(params["max_goals"])
                    if "min_lambda" in params:
                        config.model.min_lambda = float(params["min_lambda"])
                    if "draw_boost" in params:
                        config.model.draw_boost = float(params["draw_boost"])

                    print(f"[Config] Loaded from {config_file}")
            except Exception as e:
                print(f"[Config] Warning: Could not load {config_file}: {e}")
                print("[Config] Using default configuration")

        return config

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  models_dir={self.paths.models_dir}\n"
            f"  cache_dir={self.paths.cache_dir}\n"
            f"  league={self.api.league_code}\n"
            f"  max_goals={self.model.max_goals}\n"
            f"  min_lambda={self.model.min_lambda}\n"
            f"  n_jobs={self.model.n_jobs}\n"
            f")"
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance.

    Returns:
        The global Config instance.
    """
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config():
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None
