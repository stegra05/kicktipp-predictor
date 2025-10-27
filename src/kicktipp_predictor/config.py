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

    Prefers the `override` path when provided and exists; otherwise checks the
    package config directory (`src/kicktipp_predictor/config`) for `best_params.yaml`.
    Falls back to the project-level `paths.config_dir`.

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
    # Prefer packaged config file under src for easy distribution
    pkg_config_file = Path(__file__).parent / "config" / "best_params.yaml"
    if pkg_config_file.exists():
        return pkg_config_file
    # Fallback to project config directory
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
        "use_time_decay": bool,
        "time_decay_half_life_days": float,
        "form_last_n": int,
        "val_fraction": float,
        "random_state": int,
        "n_jobs": int,
        "selected_features_file": str,
        # --- V4 Cascaded Model: Draw Classifier ---
        "draw_n_estimators": int,
        "draw_max_depth": int,
        "draw_learning_rate": float,
        "draw_subsample": float,
        "draw_colsample_bytree": float,
        "draw_scale_pos_weight": float,
        # --- V4 Cascaded Model: Win Classifier (H vs A) ---
        "win_n_estimators": int,
        "win_max_depth": int,
        "win_learning_rate": float,
        "win_subsample": float,
        "win_colsample_bytree": float,
        # Additional general knobs
        "draw_margin": float,
        "avg_total_goals": float,
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

    Manages directories and file locations used by the project.

    Directories:
    - data_dir: Root directory for cached data and artifacts.
    - models_dir: Directory where trained models are stored.
    - cache_dir: Directory for HTTP/data caches.
    - config_dir: Directory containing YAML and configuration files.

    Model artifacts:
    - gd_model_path: GoalDifferenceRegressor model path.
    """

    # Data directories
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = data_dir / "models"
    cache_dir: Path = data_dir / "cache"

    # Configuration files
    config_dir: Path = PROJECT_ROOT / "src" / "kicktipp_predictor" / "config"

    def __post_init__(self) -> None:
        """Ensure all directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)




    @property
    def gd_model_path(self) -> Path:
        """Path to the goal difference regressor model."""
        return self.models_dir / "goal_diff_regressor.joblib"

# === API Settings ===
@dataclass
class APIConfig:
    """API and data fetching configuration.

    Parameters:
    - base_url (str): Root URL for OpenLigaDB or alternate provider.
    - league_code (str): League identifier (e.g., 'bl3'). Use provider's codes.
    - cache_ttl (int): Cache TTL in seconds. Range 0–86400.
    - request_timeout (int): HTTP timeout seconds. Range 1–60.
    - auth_token (str | None): Optional bearer/API token for authenticated endpoints.
    - rate_limit_per_minute (int): Max requests per minute when self-throttling.
    - retry_count (int): Number of retries on transient failures. Range 0–10.
    - backoff_initial (float): Initial backoff seconds. Range 0.0–10.0.
    - backoff_max (float): Maximum backoff seconds. Range 0.0–60.0.
    - endpoints (dict[str, str]): Relative endpoint templates used by the data loader.
      Example keys: 'season_matches', 'matchday', 'team', 'table'.
    """

    base_url: str = "https://api.openligadb.de"
    league_code: str = "bl3"  # 3. Liga
    cache_ttl: int = 3600  # Cache time-to-live in seconds (1 hour)
    request_timeout: int = 10  # HTTP request timeout in seconds

    # Authentication
    auth_token: str | None = None

    # Rate limiting & retries
    rate_limit_per_minute: int = 60
    retry_count: int = 3
    backoff_initial: float = 0.5
    backoff_max: float = 8.0

    # Endpoint templates
    endpoints: dict[str, str] = field(
        default_factory=lambda: {
            "season_matches": "getmatchdata/{league_code}/{season}",
            "matchday": "getmatchdata/{league_code}/{season}/{matchday}",
            "table": "getbltable/{league_code}/{season}",
            "team": "getteam/{team_id}",
        }
    )

# === Model Hyperparameters ===
# V4 cascaded model configuration: draw classifier + win classifier
@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration.

    Defines two cascaded classifiers for V4 architecture:
    - Draw Classifier: predicts draw vs non-draw.
    - Win Classifier: predicts home win vs away win given non-draw.

    Also includes general training and feature engineering knobs.
    """

    # --- V4 Cascaded Model: Draw Classifier ---
    draw_n_estimators: int = 400
    draw_max_depth: int = 5
    draw_learning_rate: float = 0.05
    draw_subsample: float = 0.7
    draw_colsample_bytree: float = 0.7
    draw_scale_pos_weight: float = 3.0  # Ratio of non-draw to draw samples

    # --- V4 Cascaded Model: Win Classifier (H vs A) ---
    win_n_estimators: int = 800
    win_max_depth: int = 6
    win_learning_rate: float = 0.1
    win_subsample: float = 0.8
    win_colsample_bytree: float = 0.8

    # General knobs and probabilistic helpers
    draw_margin: float = 0.5
    max_goals: int = 8
    avg_total_goals: float = 2.6

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

    # Feature selection file (optional)
    selected_features_file: str = "kept_features.yaml"

    # --- Parameter dictionaries ---
    @property
    def draw_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.draw_n_estimators,
            "max_depth": self.draw_max_depth,
            "learning_rate": self.draw_learning_rate,
            "subsample": self.draw_subsample,
            "colsample_bytree": self.draw_colsample_bytree,
            "scale_pos_weight": self.draw_scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    @property
    def win_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.win_n_estimators,
            "max_depth": self.win_max_depth,
            "learning_rate": self.win_learning_rate,
            "subsample": self.win_subsample,
            "colsample_bytree": self.win_colsample_bytree,
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
