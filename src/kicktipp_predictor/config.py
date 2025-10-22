"""Configuration management for the Kicktipp predictor.

This module centralizes all configuration parameters, including paths,
model hyperparameters, and API settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class PathConfig:
    """Configuration for file system paths.

    Attributes:
        data_dir: The directory where data is stored.
        models_dir: The directory where models are stored.
        cache_dir: The directory where cached data is stored.
        config_dir: The directory where configuration files are stored.
    """

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = data_dir / "models"
    cache_dir: Path = data_dir / "cache"
    config_dir: Path = PROJECT_ROOT / "config"

    def __post_init__(self):
        """Create the data, models, and cache directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def outcome_model_path(self) -> Path:
        """The path to the outcome classifier model."""
        return self.models_dir / "outcome_classifier.joblib"

    @property
    def home_goals_model_path(self) -> Path:
        """The path to the home goals regressor model."""
        return self.models_dir / "home_goals_regressor.joblib"

    @property
    def away_goals_model_path(self) -> Path:
        """The path to the away goals regressor model."""
        return self.models_dir / "away_goals_regressor.joblib"


@dataclass
class APIConfig:
    """Configuration for the API and data fetching.

    Attributes:
        base_url: The base URL of the API.
        league_code: The league code to fetch data for.
        cache_ttl: The time-to-live for the cache in seconds.
        request_timeout: The timeout for HTTP requests in seconds.
    """

    base_url: str = "https://api.openligadb.de"
    league_code: str = "bl3"
    cache_ttl: int = 3600
    request_timeout: int = 10


@dataclass
class ModelConfig:
    """Configuration for the model hyperparameters and training.

    Attributes:
        outcome_n_estimators: The number of estimators for the outcome
            classifier.
        outcome_max_depth: The maximum depth of the outcome classifier.
        outcome_learning_rate: The learning rate of the outcome classifier.
        outcome_subsample: The subsample ratio of the outcome classifier.
        outcome_colsample_bytree: The colsample_bytree ratio of the outcome
            classifier.
        outcome_reg_alpha: The L1 regularization term of the outcome
            classifier.
        outcome_reg_lambda: The L2 regularization term of the outcome
            classifier.
        outcome_gamma: The gamma value of the outcome classifier.
        outcome_min_child_weight: The minimum child weight of the outcome
            classifier.
        goals_n_estimators: The number of estimators for the goal regressors.
        goals_max_depth: The maximum depth of the goal regressors.
        goals_learning_rate: The learning rate of the goal regressors.
        goals_subsample: The subsample ratio of the goal regressors.
        goals_colsample_bytree: The colsample_bytree ratio of the goal
            regressors.
        goals_reg_alpha: The L1 regularization term of the goal regressors.
        goals_reg_lambda: The L2 regularization term of the goal regressors.
        goals_gamma: The gamma value of the goal regressors.
        goals_min_child_weight: The minimum child weight of the goal
            regressors.
        goals_early_stopping_rounds: The number of early stopping rounds for
            the goal regressors.
        max_goals: The maximum number of goals for the Poisson grid.
        min_lambda: The minimum lambda value for the Poisson grid.
        random_state: The random state for the models.
        test_size: The test size for the train-test split.
        min_training_matches: The minimum number of matches required for
            training.
        use_time_decay: Whether to use time decay for the training data.
        time_decay_half_life_days: The half-life for the time decay in days.
        form_last_n: The number of last matches to use for the form features.
        momentum_decay: The decay for the momentum features.
        n_jobs: The number of jobs to use for the models.
        draw_boost: The boost for the draw class in the outcome classifier.
        proba_temperature: The temperature for the outcome probabilities.
        prior_blend_alpha: The alpha for the prior blend.
        prob_source: The source of the outcome probabilities.
        hybrid_poisson_weight: The weight of the Poisson probabilities when
            using the hybrid probability source.
        proba_grid_max_goals: The maximum number of goals for the Poisson
            probability grid.
        poisson_draw_rho: The diagonal bump for draws in the Poisson
            probabilities.
    """

    outcome_n_estimators: int = 800
    outcome_max_depth: int = 6
    outcome_learning_rate: float = 0.1
    outcome_subsample: float = 0.8
    outcome_colsample_bytree: float = 0.8
    outcome_reg_alpha: float = 0.0
    outcome_reg_lambda: float = 1.0
    outcome_gamma: float = 0.0
    outcome_min_child_weight: float = 1.0
    goals_n_estimators: int = 800
    goals_max_depth: int = 6
    goals_learning_rate: float = 0.1
    goals_subsample: float = 0.8
    goals_colsample_bytree: float = 0.8
    goals_reg_alpha: float = 0.0
    goals_reg_lambda: float = 1.0
    goals_gamma: float = 0.0
    goals_min_child_weight: float = 1.0
    goals_early_stopping_rounds: int = 25
    max_goals: int = 8
    min_lambda: float = 0.2
    random_state: int = 42
    test_size: float = 0.2
    min_training_matches: int = 50
    use_time_decay: bool = True
    time_decay_half_life_days: float = 90.0
    form_last_n: int = 5
    momentum_decay: float = 0.9
    n_jobs: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("OMP_NUM_THREADS", "0")) or os.cpu_count() or 1
        )
    )
    draw_boost: float = 1.1
    proba_temperature: float = 1.0
    prior_blend_alpha: float = 0.0
    prob_source: str = "classifier"
    hybrid_poisson_weight: float = 0.5
    proba_grid_max_goals: int = 12
    poisson_draw_rho: float = 0.0


@dataclass
class Config:
    """The main configuration container.

    Attributes:
        paths: The path configuration.
        api: The API configuration.
        model: The model configuration.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load(cls, config_file: Path | None = None) -> "Config":
        """Load the configuration from a YAML file.

        Args:
            config_file: The path to the configuration file.

        Returns:
            The configuration.
        """
        config = cls()

        if config_file is None:
            env_cfg = os.getenv("KTP_CONFIG_FILE")
            if env_cfg and Path(env_cfg).exists():
                config_file = Path(env_cfg)
            else:
                config_file = config.paths.config_dir / "best_params.yaml"

        if yaml is not None and config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    params = yaml.safe_load(f)

                if isinstance(params, dict):
                    if "max_goals" in params:
                        config.model.max_goals = int(params["max_goals"])
                    if "min_lambda" in params:
                        config.model.min_lambda = float(params["min_lambda"])
                    if "draw_boost" in params:
                        config.model.draw_boost = float(params["draw_boost"])
                    if "proba_temperature" in params:
                        config.model.proba_temperature = float(
                            params["proba_temperature"]
                        )
                    if "prior_blend_alpha" in params:
                        config.model.prior_blend_alpha = float(
                            params["prior_blend_alpha"]
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
                    if "poisson_draw_rho" in params:
                        config.model.poisson_draw_rho = float(
                            params["poisson_draw_rho"]
                        )
                    if "use_time_decay" in params:
                        config.model.use_time_decay = bool(params["use_time_decay"])
                    if "time_decay_half_life_days" in params:
                        config.model.time_decay_half_life_days = float(
                            params["time_decay_half_life_days"]
                        )
                    if "form_last_n" in params:
                        config.model.form_last_n = int(params["form_last_n"])
                    if "momentum_decay" in params:
                        config.model.momentum_decay = float(params["momentum_decay"])
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
                    if "outcome_colsample_bytree" in params:
                        config.model.outcome_colsample_bytree = float(
                            params["outcome_colsample_bytree"]
                        )
                    if "outcome_reg_alpha" in params:
                        config.model.outcome_reg_alpha = float(
                            params["outcome_reg_alpha"]
                        )
                    if "outcome_reg_lambda" in params:
                        config.model.outcome_reg_lambda = float(
                            params["outcome_reg_lambda"]
                        )
                    if "outcome_gamma" in params:
                        config.model.outcome_gamma = float(params["outcome_gamma"])
                    if "outcome_min_child_weight" in params:
                        config.model.outcome_min_child_weight = float(
                            params["outcome_min_child_weight"]
                        )
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
                    if "goals_colsample_bytree" in params:
                        config.model.goals_colsample_bytree = float(
                            params["goals_colsample_bytree"]
                        )
                    if "goals_reg_alpha" in params:
                        config.model.goals_reg_alpha = float(params["goals_reg_alpha"])
                    if "goals_reg_lambda" in params:
                        config.model.goals_reg_lambda = float(
                            params["goals_reg_lambda"]
                        )
                    if "goals_gamma" in params:
                        config.model.goals_gamma = float(params["goals_gamma"])
                    if "goals_min_child_weight" in params:
                        config.model.goals_min_child_weight = float(
                            params["goals_min_child_weight"]
                        )
                    if "goals_early_stopping_rounds" in params:
                        config.model.goals_early_stopping_rounds = int(
                            params["goals_early_stopping_rounds"]
                        )

                    if os.getenv("KTP_VERBOSE") == "1":
                        print(f"[Config] Loaded from {config_file}")
            except Exception as e:
                if os.getenv("KTP_VERBOSE") == "1":
                    print(f"[Config] Warning: Could not load {config_file}: {e}")
                    print("[Config] Using default configuration")

        return config

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
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


_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration.

    Returns:
        The global configuration.
    """
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config():
    """Reset the global configuration."""
    global _config
    _config = None
