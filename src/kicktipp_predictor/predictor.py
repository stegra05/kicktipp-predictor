"""
Goal difference predictor for the Kicktipp Predictor V3 architecture.
"""

from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import norm
import numpy as np

from .config import Config, get_config

class GoalDifferencePredictor:
    """Predicts match goal differences using an XGBoost regressor.

    This class encapsulates the entire model lifecycle, including training,
    prediction, and persistence. It uses an XGBoost model to predict the goal
    difference between two teams and then translates this prediction into
    probabilistic outcomes (home win, draw, away win).

    The behavior of the predictor is controlled by a configuration object,
    which specifies model hyperparameters, feature sets, and other settings.
    """

    def __init__(self, config: Config | None = None):
        """Initializes the GoalDifferencePredictor.

        Args:
            config: An optional configuration object. If not provided, the
                global configuration is loaded.
        """
        self.config = config or get_config()
        self.model: XGBRegressor | None = None
        self.feature_columns: list[str] = []
        # Store simple training/validation metrics for persistence
        self.last_metrics: dict[str, float] | None = None

    def train(self, matches_df: pd.DataFrame) -> None:
        """Trains the goal difference predictor on the provided match data.

        This method takes a DataFrame of historical matches, selects the relevant
        features, and fits the XGBoost regressor to the goal difference target.
        It also supports time-decay weighting to give more importance to recent
        matches.

        Args:
            matches_df: A pandas DataFrame containing the training data, including
                features and the 'goal_difference' target column.

        Raises:
            ValueError: If the training DataFrame is empty, lacks the target
                column, or has insufficient samples.
        """
        # Validate input
        if matches_df is None or len(matches_df) == 0:
            raise ValueError("Empty training DataFrame provided.")
        if "goal_difference" not in matches_df.columns:
            raise ValueError("Training DataFrame must include 'goal_difference' target column.")

        # Enforce minimum training size from config
        min_n = int(self.config.model.min_training_matches)
        if len(matches_df) < min_n:
            raise ValueError(f"Insufficient training samples: {len(matches_df)} < {min_n}")

        # --- Feature selection: use all relevant features from matches_df ---
        # Exclude meta/target columns; keep numeric features only
        exclude = {
            "match_id",
            "date",
            "home_team",
            "away_team",
            "is_finished",
            "home_score",
            "away_score",
            "goal_difference",
            "result",
        }
        numeric_cols = matches_df.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.feature_columns = [c for c in numeric_cols if c not in exclude]
        # ELO hard guard removed: allow all ELO-related features
        if not self.feature_columns:
            raise ValueError("No usable feature columns found for training.")

        X_all = matches_df[self.feature_columns].copy()
        y_all = matches_df["goal_difference"].astype(float).copy()

        # --- Time-decay weighting (recency) ---
        sample_weight_all = None
        try:
            use_decay = bool(self.config.model.use_time_decay)
            half_life = float(self.config.model.time_decay_half_life_days)
            if use_decay and "date" in matches_df.columns and half_life > 0:
                dates = pd.to_datetime(matches_df["date"], errors="coerce")
                max_date = pd.to_datetime(dates.max())
                delta_days = (max_date - dates).dt.days.fillna(0).astype(float)
                # Weight halves every `half_life` days
                sample_weight_all = np.power(0.5, delta_days / half_life)
            else:
                sample_weight_all = None
        except Exception:
            # Graceful fallback: unweighted
            sample_weight_all = None

        # --- Train/Validation split (chronological when date available) ---
        val_fraction = float(self.config.model.val_fraction)
        n = len(X_all)
        val_n = int(n * val_fraction) if 0.0 < val_fraction < 0.5 else 0
        if val_n > 0:
            if "date" in matches_df.columns:
                order = np.argsort(pd.to_datetime(matches_df["date"], errors="coerce").values)
            else:
                rng = np.random.RandomState(int(self.config.model.random_state))
                order = rng.permutation(n)
            idx_train = order[: n - val_n]
            idx_val = order[n - val_n :]
        else:
            idx_train = np.arange(n)
            idx_val = np.array([], dtype=int)

        X_train = X_all.iloc[idx_train]
        y_train = y_all.iloc[idx_train]
        sw_train = sample_weight_all[idx_train] if sample_weight_all is not None else None

        # --- Model initialization from gd_* params in config ---
        try:
            params = dict(self.config.model.gd_params)
        except Exception as exc:
            raise RuntimeError(f"Failed to load gd_params from config: {exc}")
        if not params:
            raise RuntimeError("Missing gd_* parameters in config ModelConfig (gd_params empty).")

        self.model = XGBRegressor(**params)
        # --- Fit with optional early stopping ---
        X_val = X_all.iloc[idx_val] if len(idx_val) > 0 else None
        y_val = y_all.iloc[idx_val] if len(idx_val) > 0 else None
        es_rounds = int(self.config.model.gd_early_stopping_rounds)
        if X_val is not None and y_val is not None and es_rounds > 0:
            self.model.fit(
                X_train,
                y_train,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                early_stopping_rounds=es_rounds,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, sample_weight=sw_train)

        # --- Validation metrics ---
        metrics: dict[str, float] = {}
        if len(idx_val) > 0:
            X_val = X_all.iloc[idx_val]
            y_val = y_all.iloc[idx_val]
            sw_val = sample_weight_all[idx_val] if sample_weight_all is not None else None
            y_pred = self.model.predict(X_val)
            # Basic numeric metrics (weighted where applicable)
            diff = y_pred - y_val.values
            if sw_val is not None:
                w = sw_val
                mae = float(np.sum(np.abs(diff) * w) / np.sum(w))
                rmse = float(np.sqrt(np.sum((diff ** 2) * w) / np.sum(w)))
            else:
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
            # Naive R^2 (unweighted)
            var = float(np.var(y_val.values))
            r2 = float(1.0 - (float(np.mean(diff ** 2)) / var)) if var > 0 else float("nan")
            metrics = {"val_mae": mae, "val_rmse": rmse, "val_r2": r2, "n_val": float(len(idx_val))}
        else:
            metrics = {"val_mae": float("nan"), "val_rmse": float("nan"), "val_r2": float("nan"), "n_val": 0.0}

        self.last_metrics = metrics

    def predict(self, features_df: pd.DataFrame) -> list[dict]:
        """Makes predictions for upcoming matches.

        This method takes a DataFrame of features for upcoming matches, predicts
        the goal difference, and then derives the probabilities for home win,
        draw, and away win.

        Args:
            features_df: A pandas DataFrame with the same features used for
                training.

        Returns:
            A list of prediction dictionaries, where each dictionary contains
            the predicted goal difference, score, and outcome probabilities.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
            ValueError: If the features DataFrame is empty.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded. Call train() or load_model() first.")
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            raise ValueError("features_df must be a non-empty pandas DataFrame.")
        # Align features and predict
        if not self.feature_columns:
            exclude = {
                "match_id",
                "date",
                "home_team",
                "away_team",
                "is_finished",
                "home_score",
                "away_score",
                "goal_difference",
                "result",
            }
            numeric_cols = features_df.select_dtypes(include=["number", "bool"]).columns.tolist()
            self.feature_columns = [c for c in numeric_cols if c not in exclude]
        X = features_df.reindex(columns=self.feature_columns).fillna(0.0)
        pred_gd = self.model.predict(X)
        # Probabilistic bridge using Normal around predicted GD (dynamic stddev)

        base = float(getattr(self.config.model, "gd_uncertainty_base_stddev", self.config.model.gd_uncertainty_stddev))
        scale = float(getattr(self.config.model, "gd_uncertainty_scale", 0.0))
        min_std = float(getattr(self.config.model, "gd_uncertainty_min_stddev", 0.2))
        max_std = float(getattr(self.config.model, "gd_uncertainty_max_stddev", 4.0))
        # Validate parameters
        if not np.isfinite(base) or base <= 0:
            base = max(1e-6, float(self.config.model.gd_uncertainty_stddev))
        if not np.isfinite(scale) or scale < 0:
            scale = 0.0
        if not np.isfinite(min_std) or min_std <= 0:
            min_std = 1e-6
        if not np.isfinite(max_std) or max_std <= min_std:
            max_std = max(min_std * 2.0, 1.0)
        # Dynamic stddev by predicted goal difference magnitude
        dynamic_stddev = base + scale * np.abs(pred_gd)
        # Clamp bounds for numerical stability
        dynamic_stddev = np.clip(dynamic_stddev, min_std, max_std)
        # Draw margin parameterization with validation
        draw_margin = float(getattr(self.config.model, "draw_margin", 0.5))
        if not np.isfinite(draw_margin):
            draw_margin = 0.5
        draw_margin = float(np.clip(draw_margin, 0.1, 1.0))
        p_home = 1 - norm.cdf(draw_margin, loc=pred_gd, scale=dynamic_stddev)
        p_away = norm.cdf(-draw_margin, loc=pred_gd, scale=dynamic_stddev)
        p_draw = norm.cdf(draw_margin, loc=pred_gd, scale=dynamic_stddev) - p_away
        probs = np.vstack([p_home, p_draw, p_away]).T
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        # Golden Rule: scoreline will be computed per match to be consistent with argmax outcome
        # Monitoring: uncertainty correlation and bounds
        try:
            abs_gd = np.abs(pred_gd)
            if len(abs_gd) > 1 and np.std(abs_gd) > 0 and np.std(dynamic_stddev) > 0:
                corr = float(np.corrcoef(abs_gd, dynamic_stddev)[0, 1])
            else:
                corr = float("nan")
            self.last_metrics = {
                "uncertainty_corr_abs_gd": corr,
                "stddev_mean": float(np.mean(dynamic_stddev)),
                "stddev_min": float(np.min(dynamic_stddev)),
                "stddev_max": float(np.max(dynamic_stddev)),
                "draw_margin": draw_margin,
            }
        except Exception:
            pass
        # Format predictions
        predictions: list[dict] = []
        for i in range(len(features_df)):
            row = features_df.iloc[i]
            probs_i = probs[i]
            outcome_idx = int(np.argmax(probs_i))
            predicted_outcome = ["H", "D", "A"][outcome_idx]
            pred_gd_i = float(pred_gd[i])

            # Outcome-consistent tiered scoreline heuristic
            if predicted_outcome == "H":
                if pred_gd_i > 1.5:
                    score = (2, 0)
                else:
                    score = (2, 1)
            elif predicted_outcome == "A":
                if pred_gd_i < -1.5:
                    score = (0, 2)
                else:
                    score = (1, 2)
            else:  # Draw
                score = (1, 1)

            pred = {
                "match_id": row.get("match_id"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "predicted_goal_difference": pred_gd_i,
                "predicted_home_score": int(score[0]),
                "predicted_away_score": int(score[1]),
                "home_win_probability": float(probs_i[0]),
                "draw_probability": float(probs_i[1]),
                "away_win_probability": float(probs_i[2]),
                "uncertainty_stddev": float(dynamic_stddev[i]),
                "draw_margin_used": float(draw_margin),
                "predicted_result": predicted_outcome,
            }
            if "matchday" in features_df.columns:
                try:
                    pred["matchday"] = int(row.get("matchday"))
                except Exception:
                    pred["matchday"] = row.get("matchday")
            predictions.append(pred)
        return predictions

    def save_model(self) -> None:
        """Saves the trained model and its metadata to disk.

        The model and its associated metadata (feature columns, hyperparameters)
        are saved to the paths specified in the configuration.

        Raises:
            RuntimeError: If there is no model to save.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Must train first.")
        # Persist model and minimal metadata
        metadata = {
            "feature_columns": list(self.feature_columns),
            "gd_params": dict(self.config.model.gd_params),
            "metrics": self.last_metrics or {},
        }
        try:
            joblib.dump(self.model, self.config.paths.gd_model_path)
            joblib.dump(metadata, self.config.paths.gd_model_path.with_name("metadata.joblib"))
        except Exception as exc:
            raise RuntimeError(f"Failed to save model artifacts: {exc}")

    def load_model(self) -> None:
        """Loads a trained model and its metadata from disk.

        This method loads the XGBoost model and its metadata from the paths
        specified in the configuration. It also performs a dimensionality check
        to ensure that the loaded model is compatible with the feature columns.

        Raises:
            RuntimeError: If the model or its metadata cannot be loaded, or if
                there is a feature dimension mismatch.
        """
        try:
            self.model = joblib.load(self.config.paths.gd_model_path)
        except FileNotFoundError as exc:
            raise exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}")
        try:
            metadata = joblib.load(self.config.paths.gd_model_path.with_name("metadata.joblib"))
            if not isinstance(metadata, dict):
                raise RuntimeError("Invalid metadata format.")
            cols = metadata.get("feature_columns")
            if not isinstance(cols, list) or len(cols) == 0:
                raise RuntimeError("Missing feature_columns in metadata.")
            self.feature_columns = [str(c) for c in cols]
            # Dimensionality check
            n_in = getattr(self.model, "n_features_in_", None)
            if n_in is not None and int(n_in) != len(self.feature_columns):
                raise RuntimeError("Feature dimension mismatch between model and metadata.")
        except FileNotFoundError as exc:
            raise RuntimeError("Model metadata not found; retrain required.") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load model metadata: {exc}")