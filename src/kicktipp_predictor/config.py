"""Configuration management for kicktipp predictor.

This module centralizes all configuration parameters including paths,
model hyperparameters, and API settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

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

    # Early stopping
    goals_early_stopping_rounds: int = 25

    # Poisson grid for scoreline selection
    max_goals: int = 8

    # Training
    random_state: int = 42
    test_size: float = 0.2
    min_training_matches: int = 50

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
    # Temperature < 1 sharpens, > 1 softens
    proba_temperature: float = 1.0
    # Blend with empirical prior from training window


    # Outcome probability source for evaluation and reporting
    # One of: 'classifier' (default), 'poisson', 'hybrid'
    prob_source: str = "hybrid"
    # When prob_source='hybrid', weight of Poisson-derived probabilities in [0,1]
    # Note: only fixed-weight blending is supported.
    hybrid_poisson_weight: float = 0.0525
    # Max goals for Poisson probability grid used to derive P(H/D/A) (separate from scoreline grid)
    proba_grid_max_goals: int = 12

    # --- New: EP scoreline selection toggle ---
    use_ep_selection: bool = True

    selected_features_file: str = "kept_features.yaml"


    @property
    def goals_params(self) -> dict:
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
    def outcome_params(self) -> dict:
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


@dataclass
class Config:
    """Main configuration container."""

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load(cls, config_file: Path | None = None) -> "Config":
        """Load configuration from file if available.

        Args:
            config_file: Path to config YAML file. If None, uses default location.

        Returns:
            Config instance with loaded or default values.
        """
        config = cls()

        if config_file is None:
            # Allow override via environment variable for tuning processes
            env_cfg = os.getenv("KTP_CONFIG_FILE")
            if env_cfg and Path(env_cfg).exists():
                config_file = Path(env_cfg)
            else:
                config_file = config.paths.config_dir / "best_params.yaml"

        # Try to load from YAML if available
        if yaml is not None and config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    params = yaml.safe_load(f)

                if isinstance(params, dict):
                    # Load model parameters if present
                    if "max_goals" in params:
                        config.model.max_goals = int(params["max_goals"])

                    if "draw_boost" in params:
                        config.model.draw_boost = float(params["draw_boost"])
                    if "proba_temperature" in params:
                        config.model.proba_temperature = float(
                            params["proba_temperature"]
                        )

                    if "prob_source" in params:
                        config.model.prob_source = (
                            str(params["prob_source"]).strip().lower()
                        )
                    if "hybrid_poisson_weight" in params:
                        config.model.hybrid_poisson_weight = float(
                            params["hybrid_poisson_weight"]
                        )
                    if "proba_grid_max_goals" in params:
                        config.model.proba_grid_max_goals = int(
                            params["proba_grid_max_goals"]
                        )
                    # New: EP selection toggle from YAML
                    if "use_ep_selection" in params:
                        config.model.use_ep_selection = bool(params["use_ep_selection"])

                    # Time-decay weighting and feature knobs
                    if "use_time_decay" in params:
                        config.model.use_time_decay = bool(params["use_time_decay"])
                    if "time_decay_half_life_days" in params:
                        config.model.time_decay_half_life_days = float(
                            params["time_decay_half_life_days"]
                        )
                    if "form_last_n" in params:
                        config.model.form_last_n = int(params["form_last_n"])

                    # Outcome classifier hyperparameters
                    if "outcome_n_estimators" in params:
                        config.model.outcome_n_estimators = int(
                            params["outcome_n_estimators"]
                        )
                    if "outcome_max_depth" in params:
                        config.model.outcome_max_depth = int(
                            params["outcome_max_depth"]
                        )
                    if "outcome_learning_rate" in params:
                        config.model.outcome_learning_rate = float(
                            params["outcome_learning_rate"]
                        )
                    if "outcome_subsample" in params:
                        config.model.outcome_subsample = float(
                            params["outcome_subsample"]
                        )
                    if "outcome_reg_lambda" in params:
                        config.model.outcome_reg_lambda = float(
                            params["outcome_reg_lambda"]
                        )
                    if "outcome_min_child_weight" in params:
                        config.model.outcome_min_child_weight = float(
                            params["outcome_min_child_weight"]
                        )

                    # Goal regressors hyperparameters
                    if "goals_n_estimators" in params:
                        config.model.goals_n_estimators = int(
                            params["goals_n_estimators"]
                        )
                    if "goals_max_depth" in params:
                        config.model.goals_max_depth = int(params["goals_max_depth"])
                    if "goals_learning_rate" in params:
                        config.model.goals_learning_rate = float(
                            params["goals_learning_rate"]
                        )
                    if "goals_subsample" in params:
                        config.model.goals_subsample = float(params["goals_subsample"])
                    if "goals_reg_lambda" in params:
                        config.model.goals_reg_lambda = float(
                            params["goals_reg_lambda"]
                        )
                    if "goals_min_child_weight" in params:
                        config.model.goals_min_child_weight = float(
                            params["goals_min_child_weight"]
                        )


            except Exception:
                pass

        return config

    def __str__(self) -> str:
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


def reset_config():
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None
