"""Configuration management for kicktipp predictor.

This module centralizes all configuration parameters including paths,
model hyperparameters, and API settings.
"""

# === Imports ===
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

# === Project Root ===
PROJECT_ROOT = Path(__file__).parent.parent.parent

# === Utilities ===
def _resolve_config_file(paths: "PathConfig", override: str | None) -> Path:
    """Resolve configuration file path.

    Prefers the `override` path when provided and exists; otherwise uses the
    default file in `paths.config_dir`.

    Args:
        paths: Path configuration.
        override: Optional path from environment.

    Returns:
        Path to the configuration file to use.
    """
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate
    return paths.config_dir / "best_params.yaml"


def _read_yaml_params(config_file: Path) -> dict[str, Any] | None:
    """Read YAML parameters from a file if possible.

    Provides graceful fallbacks: returns None when YAML support is missing
    or the file does not exist. Emits a warning on read/parse failures.

    Args:
        config_file: Path to the YAML file.

    Returns:
        A dictionary of parameters if loaded, else None.
    """
    if yaml is None:
        warnings.warn("YAML support not available; using defaults.")
        return None
    if not config_file.exists():
        return None
    try:
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # pragma: no cover - robust error path
        warnings.warn(f"Failed to read config file '{config_file}': {exc}")
        return None


def _apply_model_params_from_dict(config: "Config", params: dict[str, Any]) -> None:
    """Apply model-related parameters to `config.model` from a dict.

    Handles type conversion with simple casting functions and skips invalid
    values with a warning, preserving existing defaults.

    Args:
        config: The configuration container to mutate.
        params: Parameter dictionary loaded from YAML.
    """
    casts: dict[str, Callable[[Any], Any]] = {
        # General knobs
        "max_goals": int,
        "draw_boost": float,
        "use_time_decay": bool,
        "time_decay_half_life_days": float,
        "form_last_n": int,
        "hybrid_weight": float,
        "val_fraction": float,
        # Outcome hyperparameters
        "outcome_n_estimators": int,
        "outcome_max_depth": int,
        "outcome_learning_rate": float,
        "outcome_subsample": float,
        "outcome_reg_lambda": float,
        "outcome_min_child_weight": float,
        # Goal regressors hyperparameters
        "goals_n_estimators": int,
        "goals_max_depth": int,
        "goals_learning_rate": float,
        "goals_subsample": float,
        "goals_reg_lambda": float,
        "goals_min_child_weight": float,
    }

    for key, caster in casts.items():
        if key in params:
            try:
                value = caster(params[key])
                setattr(config.model, key, value)
            except Exception as exc:  # pragma: no cover - robust error path
                warnings.warn(f"Invalid value for '{key}': {exc}")

# === Paths ===
@dataclass
class PathConfig:
    """File system paths configuration.

    Creates and manages directories used by the project.
    """

    # Data directories
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = data_dir / "models"
    cache_dir: Path = data_dir / "cache"

    # Configuration files
    config_dir: Path = PROJECT_ROOT / "config"

    def __post_init__(self) -> None:
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

# === API Settings ===
@dataclass
class APIConfig:
    """API and data fetching configuration."""

    base_url: str = "https://api.openligadb.de"
    league_code: str = "bl3"  # 3. Liga
    cache_ttl: int = 3600  # Cache time-to-live in seconds (1 hour)
    request_timeout: int = 10  # HTTP request timeout in seconds

# === Model Hyperparameters ===
# Defaults updated per final tuning run (2025-10-24)
# See README Key Parameters for rationale.
@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration."""

    # XGBoost Outcome Classifier (H/D/A)
    outcome_n_estimators: int = 1150
    outcome_max_depth: int = 6
    outcome_learning_rate: float = 0.1
    outcome_subsample: float = 0.8
    # Regularization and constraints
    outcome_reg_lambda: float = 1.0
    outcome_min_child_weight: float = 0.1587

    # XGBoost Goal Regressors
    goals_n_estimators: int = 800
    goals_max_depth: int = 6
    goals_learning_rate: float = 0.1
    goals_subsample: float = 0.8
    # Regularization and constraints
    goals_reg_lambda: float = 1.0
    goals_min_child_weight: float = 1.6919

    # Poisson grid for scoreline selection
    max_goals: int = 8

    # Training
    random_state: int = 42
    min_training_matches: int = 50
    val_fraction: float = 0.1

    # Time-decay weighting (recency)
    use_time_decay: bool = True
    time_decay_half_life_days: float = 330.0

    # Feature engineering knobs
    form_last_n: int = 5

    # Threading
    n_jobs: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("OMP_NUM_THREADS", "0")) or os.cpu_count() or 1
        )
    )

    # Class weights for outcome classifier
    # Boost draw class since it's underrepresented
    draw_boost: float = 1.7

    # Outcome probability post-processing
    hybrid_weight: float = 0.0525
    # Blend with empirical prior from training window

    selected_features_file: str = "kept_features.yaml"

    @property
    def goals_params(self) -> dict[str, int | float]:
        """Return XGBoost parameters for goal regressors as a dictionary."""
        return {
            "n_estimators": self.goals_n_estimators,
            "max_depth": self.goals_max_depth,
            "learning_rate": self.goals_learning_rate,
            "subsample": self.goals_subsample,
            "reg_lambda": self.goals_reg_lambda,
            "min_child_weight": self.goals_min_child_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    @property
    def outcome_params(self) -> dict[str, int | float]:
        """Return XGBoost parameters for outcome classifier as a dictionary."""
        return {
            "n_estimators": self.outcome_n_estimators,
            "max_depth": self.outcome_max_depth,
            "learning_rate": self.outcome_learning_rate,
            "subsample": self.outcome_subsample,
            "reg_lambda": self.outcome_reg_lambda,
            "min_child_weight": self.outcome_min_child_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

# === Configuration Container ===
@dataclass
class Config:
    """Main configuration container.

    Aggregates path, API, and model configuration sections and provides a
    loader with YAML override support and a readable string summary.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load(cls, config_file: Path | None = None) -> "Config":
        """Load configuration from file if available.

        Resolution order:
        1. Explicit `config_file` argument when provided.
        2. `KTP_CONFIG_FILE` environment variable when it points to a file.
        3. Default `best_params.yaml` in the project config directory.

        Args:
            config_file: Path to config YAML file. If None, uses default.

        Returns:
            Config instance with loaded or default values.
        """
        config = cls()

        # Choose file path
        chosen_file = (
            config_file
            if config_file is not None
            else _resolve_config_file(
                config.paths, os.getenv("KTP_CONFIG_FILE")
            )
        )

        # Load parameters and apply
        params = _read_yaml_params(chosen_file)
        if params:
            _apply_model_params_from_dict(config, params)

        return config

    def __str__(self) -> str:  # pragma: no cover - human-friendly output
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  models_dir={self.paths.models_dir}\n"
            f"  cache_dir={self.paths.cache_dir}\n"
            f"  league={self.api.league_code}\n"
            f"  max_goals={self.model.max_goals}\n"
            f"  n_jobs={self.model.n_jobs}\n"
            f")"
        )

# === Global Access ===
# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global configuration instance.

    Returns:
        The global Config instance.
    """
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None
