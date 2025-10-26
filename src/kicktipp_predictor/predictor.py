"""
Goal difference predictor for the Kicktipp Predictor V3 architecture.
"""

from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import norm

from .config import Config, get_config

class GoalDifferencePredictor:
    """
    Predicts match goal differences using an XGBoost regressor.

    This class encapsulates the model lifecycle, including training,
    probabilistic prediction, and persistence, all driven by a central
    configuration object.
    """

    def __init__(self, config: Config | None = None):
        """Initializes the predictor with its configuration."""
        self.config = config or get_config()
        self.model: XGBRegressor | None = None
        self.feature_columns: list[str] = []

    def train(self, matches_df: pd.DataFrame) -> None:
        """
        Trains the XGBRegressor on the provided match data.

        Args:
            matches_df: A DataFrame containing features and the 'goal_difference' target.
        """
        # --- Feature and Target Preparation ---
        # (Implementation to be added)
        # 1. Determine feature_columns from matches_df and config
        # 2. X = matches_df[self.feature_columns]
        # 3. y = matches_df["goal_difference"]
        
        # --- Model Training ---
        # (Implementation to be added)
        # 1. self.model = XGBRegressor(**self.config.model.gd_params)
        # 2. self.model.fit(X, y, sample_weight=...)
        raise NotImplementedError("Training logic not implemented yet.")

    def predict(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Makes predictions and derives H/D/A probabilities.

        Args:
            features_df: DataFrame with features for upcoming matches.

        Returns:
            A list of prediction dictionaries, including goal differences
            and outcome probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded. Call train() or load_model() first.")

        # --- Prediction ---
        # (Implementation to be added)
        # 1. X = features_df[self.feature_columns]
        # 2. pred_gd = self.model.predict(X)

        # --- Probabilistic Bridge ---
        # (Implementation to be added)
        # 1. stddev = self.config.model.gd_uncertainty_stddev
        # 2. p_home = 1 - norm.cdf(0.5, loc=pred_gd, scale=stddev)
        # 3. p_away = norm.cdf(-0.5, loc=pred_gd, scale=stddev)
        # 4. p_draw = 1 - p_home - p_away

        # --- Formatting ---
        # (Implementation to be added)
        # 1. Format output into list of dicts with all required keys.
        raise NotImplementedError("Prediction logic not implemented yet.")

    def save_model(self) -> None:
        """Saves the trained model and metadata to the path specified in the config."""
        if self.model is None:
            raise RuntimeError("No model to save. Must train first.")
        
        # (Implementation to be added)
        # 1. metadata = {"feature_columns": self.feature_columns}
        # 2. joblib.dump(self.model, self.config.paths.gd_model_path)
        # 3. joblib.dump(metadata, self.config.paths.gd_model_path.with_name("metadata.joblib"))
        raise NotImplementedError("Model saving not implemented yet.")

    def load_model(self) -> None:
        """Loads the model and metadata from the path specified in the config."""
        # (Implementation to be added)
        # 1. self.model = joblib.load(self.config.paths.gd_model_path)
        # 2. metadata = joblib.load(self.config.paths.gd_model_path.with_name("metadata.joblib"))
        # 3. self.feature_columns = metadata["feature_columns"]
        raise NotImplementedError("Model loading not implemented yet.")