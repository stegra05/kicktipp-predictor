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


@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration."""

    # XGBoost Outcome Classifier (H/D/A)
    outcome_n_estimators: int = 800
    outcome_max_depth: int = 6
    outcome_learning_rate: float = 0.1
    outcome_subsample: float = 0.8
    outcome_colsample_bytree: float = 0.8
    # Regularization and constraints
    outcome_reg_alpha: float = 0.0
    outcome_reg_lambda: float = 1.0
    outcome_gamma: float = 0.0
    outcome_min_child_weight: float = 1.0

    # XGBoost Goal Regressors
    goals_n_estimators: int = 800
    goals_max_depth: int = 6
    goals_learning_rate: float = 0.1
    goals_subsample: float = 0.8
    goals_colsample_bytree: float = 0.8
    # Regularization and constraints
    goals_reg_alpha: float = 0.0
    goals_reg_lambda: float = 1.0
    goals_gamma: float = 0.0
    goals_min_child_weight: float = 1.0

    # Early stopping
    goals_early_stopping_rounds: int = 25

    # Poisson grid for scoreline selection
    max_goals: int = 8
    min_lambda: float = 0.2  # Minimum expected goals to avoid degenerate predictions

    # Training
    random_state: int = 42
    test_size: float = 0.2
    min_training_matches: int = 50

    # Time-decay weighting (recency)
    use_time_decay: bool = True
    time_decay_half_life_days: float = 90.0

    # Feature engineering knobs
    form_last_n: int = 5
    momentum_decay: float = 0.9

    # Threading
    n_jobs: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("OMP_NUM_THREADS", "0")) or os.cpu_count() or 1
        )
    )

    # Class weights for outcome classifier
    # Boost draw class since it's underrepresented
    draw_boost: float = 1.1

    # Outcome probability post-processing
    # Temperature < 1 sharpens, > 1 softens
    proba_temperature: float = 1.0
    # Blend with empirical prior from training window
    prior_blend_alpha: float = 0.0

    # Outcome probability source for evaluation and reporting
    # One of: 'classifier' (default), 'poisson', 'hybrid'
    prob_source: str = "hybrid"
    # When prob_source='hybrid', weight of Poisson-derived probabilities in [0,1]
    hybrid_poisson_weight: float = 0.5
    # Hybrid weighting scheme: 'fixed' uses hybrid_poisson_weight, 'entropy' adapts per-match
    hybrid_scheme: str = "entropy"
    # Entropy-based hybrid weight bounds (in [0,1])
    hybrid_entropy_w_min: float = 0.2
    hybrid_entropy_w_max: float = 1.0
    # Max goals for Poisson probability grid used to derive P(H/D/A) (separate from scoreline grid)
    proba_grid_max_goals: int = 12
    # Draw bump for Poisson-derived probabilities: multiply diagonal cells by exp(rho) before normalization
    poisson_draw_rho: float = 0.0
    # Joint model for Poisson score grid: 'independent' or 'dixon_coles'
    poisson_joint: str = "dixon_coles"
    # Dixon–Coles low-score correlation parameter (small magnitude, e.g., -0.05..0.05)
    dixon_coles_rho: float = 0.0

    # Decision logic enhancements
    # Aggressive value weighting multipliers for 2/3/4-point components
    value_weight_2pt: float = 1.0
    value_weight_3pt: float = 1.2
    value_weight_4pt: float = 1.3
    # Confidence-based scoreline shift (for 'H' outcome only)
    confidence_shift_threshold: float = 0.15
    confidence_shift_prob_ratio: float = 0.5
    # Entropy-guided draw forcing
    force_draw_enabled: bool = True
    force_draw_entropy_threshold: float = 0.95

    # Feature selection list filename under config/ (default kept_features.yaml)
    selected_features_file: str = "kept_features.yaml"

    # Calibration of final blended probabilities
    calibrator_enabled: bool = True
    calibrator_method: str = (
        "dirichlet"  # or 'multinomial_logistic' fallback if external lib unavailable
    )
    calibrator_C: float = 1.0
    calibrator_cv_folds: int = 3
    # Post-calibration class-prior anchoring
    prior_anchor_enabled: bool = False
    prior_anchor_strength: float = 0.15
    # Entropy-based hybrid tuning candidates
    hybrid_entropy_tune: bool = False
    hybrid_entropy_w_min_candidates: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3]
    )
    hybrid_entropy_w_max_candidates: list[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.9, 1.0]
    )

    @property
    def goals_params(self) -> dict:
        """Return XGBoost parameters for goal regressors as a dictionary."""
        return {
            "n_estimators": self.goals_n_estimators,
            "max_depth": self.goals_max_depth,
            "learning_rate": self.goals_learning_rate,
            "subsample": self.goals_subsample,
            "colsample_bytree": self.goals_colsample_bytree,
            "reg_alpha": self.goals_reg_alpha,
            "reg_lambda": self.goals_reg_lambda,
            "gamma": self.goals_gamma,
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
            "colsample_bytree": self.outcome_colsample_bytree,
            "reg_alpha": self.outcome_reg_alpha,
            "reg_lambda": self.outcome_reg_lambda,
            "gamma": self.outcome_gamma,
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
                    if "hybrid_scheme" in params:
                        config.model.hybrid_scheme = (
                            str(params["hybrid_scheme"]).strip().lower()
                        )
                    if "hybrid_entropy_w_min" in params:
                        config.model.hybrid_entropy_w_min = float(
                            params["hybrid_entropy_w_min"]
                        )
                    if "hybrid_entropy_w_max" in params:
                        config.model.hybrid_entropy_w_max = float(
                            params["hybrid_entropy_w_max"]
                        )
                    if "proba_grid_max_goals" in params:
                        config.model.proba_grid_max_goals = int(
                            params["proba_grid_max_goals"]
                        )
                    if "poisson_draw_rho" in params:
                        config.model.poisson_draw_rho = float(
                            params["poisson_draw_rho"]
                        )
                    if "poisson_joint" in params:
                        config.model.poisson_joint = (
                            str(params["poisson_joint"]).strip().lower()
                        )
                    if "dixon_coles_rho" in params:
                        config.model.dixon_coles_rho = float(params["dixon_coles_rho"])

                    # New decision logic knobs
                    if "value_weight_2pt" in params:
                        config.model.value_weight_2pt = float(
                            params["value_weight_2pt"]
                        )
                    if "value_weight_3pt" in params:
                        config.model.value_weight_3pt = float(
                            params["value_weight_3pt"]
                        )
                    if "value_weight_4pt" in params:
                        config.model.value_weight_4pt = float(
                            params["value_weight_4pt"]
                        )
                    if "confidence_shift_threshold" in params:
                        config.model.confidence_shift_threshold = float(
                            params["confidence_shift_threshold"]
                        )
                    if "confidence_shift_prob_ratio" in params:
                        config.model.confidence_shift_prob_ratio = float(
                            params["confidence_shift_prob_ratio"]
                        )
                    if "force_draw_enabled" in params:
                        config.model.force_draw_enabled = bool(
                            params["force_draw_enabled"]
                        )
                    if "force_draw_entropy_threshold" in params:
                        config.model.force_draw_entropy_threshold = float(
                            params["force_draw_entropy_threshold"]
                        )

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

                    # Calibration-related knobs
                    if "calibrator_enabled" in params:
                        config.model.calibrator_enabled = bool(
                            params["calibrator_enabled"]
                        )
                    if "calibrator_method" in params:
                        config.model.calibrator_method = (
                            str(params["calibrator_method"]).strip().lower()
                        )
                    if "calibrator_C" in params:
                        config.model.calibrator_C = float(params["calibrator_C"])
                    if "calibrator_cv_folds" in params:
                        config.model.calibrator_cv_folds = int(
                            params["calibrator_cv_folds"]
                        )
                    if "prior_anchor_enabled" in params:
                        config.model.prior_anchor_enabled = bool(
                            params["prior_anchor_enabled"]
                        )
                    if "prior_anchor_strength" in params:
                        config.model.prior_anchor_strength = float(
                            params["prior_anchor_strength"]
                        )

                    # Optional: hybrid entropy tuning controls
                    if "hybrid_entropy_tune" in params:
                        config.model.hybrid_entropy_tune = bool(
                            params["hybrid_entropy_tune"]
                        )
                    if "hybrid_entropy_w_min_candidates" in params:
                        try:
                            config.model.hybrid_entropy_w_min_candidates = [
                                float(x)
                                for x in params["hybrid_entropy_w_min_candidates"]
                            ]
                        except Exception:
                            pass
                    if "hybrid_entropy_w_max_candidates" in params:
                        try:
                            config.model.hybrid_entropy_w_max_candidates = [
                                float(x)
                                for x in params["hybrid_entropy_w_max_candidates"]
                            ]
                        except Exception:
                            pass

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
            f"  min_lambda={self.model.min_lambda}\n"
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
