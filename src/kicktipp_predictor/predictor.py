"""
Predictor implementations for the Kicktipp Predictor project.

Includes:
- GoalDifferencePredictor (legacy V3 regressor)
- CascadedPredictor (new V4 cascaded classifiers: draw + win)
"""

from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional

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
        # Store simple training/validation metrics for persistence
        self.last_metrics: dict[str, float] | None = None

    def train(self, matches_df: pd.DataFrame) -> None:
        """
        Trains the XGBRegressor on the provided match data.

        Args:
            matches_df: A DataFrame containing features and the 'goal_difference' target.
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
        # Legacy path retained: if gd_params exist, use them; otherwise raise
        if not hasattr(self.config.model, "gd_params"):
            raise RuntimeError("Legacy gd_params not available in ModelConfig. Use CascadedPredictor for V4.")
        params = dict(self.config.model.gd_params)
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
        # Smoother scoreline heuristic
        predicted_scores: list[tuple[int, int]] = []
        avg_goals = float(self.config.model.avg_total_goals)
        alpha = float(self.config.model.gd_score_alpha)
        for gd in pred_gd:
            T = avg_goals + alpha * float(abs(gd))
            home = max(0, int(round((T + gd) / 2.0)))
            away = max(0, int(round((T - gd) / 2.0)))
            predicted_scores.append((home, away))
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
            pred = {
                "match_id": row.get("match_id"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "predicted_goal_difference": float(pred_gd[i]),
                "predicted_home_score": int(predicted_scores[i][0]),
                "predicted_away_score": int(predicted_scores[i][1]),
                "home_win_probability": float(probs[i, 0]),
                "draw_probability": float(probs[i, 1]),
                "away_win_probability": float(probs[i, 2]),
                "uncertainty_stddev": float(dynamic_stddev[i]),
                "draw_margin_used": float(draw_margin),
            }
            # Add derived predicted result label for CLI display
            try:
                outcome_idx = int(np.argmax(probs[i]))
                pred["predicted_result"] = ["H", "D", "A"][outcome_idx]
            except Exception:
                pred["predicted_result"] = None
            if "matchday" in features_df.columns:
                try:
                    pred["matchday"] = int(row.get("matchday"))
                except Exception:
                    pred["matchday"] = row.get("matchday")
            predictions.append(pred)
        return predictions

    def save_model(self) -> None:
        """Saves the trained model and metadata to the path specified in the config."""
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
        """Loads the model and metadata from the path specified in the config."""
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


class CascadedPredictor:
    """Cascaded classifier predictor for match outcomes (V4).

    This predictor uses two binary classifiers in a cascade:
    - Draw Classifier: predicts Draw vs NotDraw.
    - Win Classifier: predicts HomeWin vs AwayWin conditioned on NotDraw.

    Attributes:
    - config: Configuration object; defaults to global `get_config()` when not provided.
    - draw_model: XGBClassifier for draw prediction; initialized to None until trained/loaded.
    - win_model: XGBClassifier for win prediction; initialized to None until trained/loaded.
    - feature_columns: List of feature names used during training/prediction.
    - draw_label_encoder: LabelEncoder for draw outcomes (0=NotDraw, 1=Draw).
    - win_label_encoder: LabelEncoder for win outcomes (0=AwayWin, 1=HomeWin).
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the cascaded predictor.

        Args:
            config: Optional configuration object. If not provided, uses `get_config()`.
        """
        self.config = config or get_config()
        self.draw_model: Optional[XGBClassifier] = None
        self.win_model: Optional[XGBClassifier] = None
        self.feature_columns: list[str] = []

        # Initialize label encoders
        self.draw_label_encoder = LabelEncoder()
        self.win_label_encoder = LabelEncoder()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from input DataFrame.

        Selects numeric/bool columns and excludes typical metadata columns.

        Args:
            df: Input DataFrame containing features and possibly metadata columns.

        Returns:
            DataFrame with selected feature columns, aligned to `self.feature_columns` when set.
        """
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
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        if not self.feature_columns:
            self.feature_columns = [c for c in numeric_cols if c not in exclude]
            if not self.feature_columns:
                raise ValueError("No usable feature columns found.")
        X = df.reindex(columns=self.feature_columns).fillna(0.0)
        return X

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Train both draw and win classifiers with preprocessing.

        Expected input:
        - `X_train`: DataFrame of features (numeric/bool). Non-feature columns are ignored.
        - `y_train`: DataFrame with either:
          - column `result` containing values 'H', 'D', 'A'; or
          - columns `is_draw` (bool/int) and `win_outcome` ('HomeWin'/'AwayWin' or 1/0).

        Process:
        - Fit the draw classifier on all samples (labels 0=NotDraw, 1=Draw).
        - Fit the win classifier on non-draw samples (labels 0=AwayWin, 1=HomeWin).

        Raises:
        - ValueError for invalid inputs or insufficient samples.
        """
        if not isinstance(X_train, pd.DataFrame) or X_train.empty:
            raise ValueError("X_train must be a non-empty pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame) or y_train.empty:
            raise ValueError("y_train must be a non-empty pandas DataFrame.")

        # Derive labels from y_train
        if "result" in y_train.columns:
            res = y_train["result"].astype(str)
            y_draw = (res == "D").astype(int)
            # For win classifier, drop draw rows
            non_draw_mask = res.isin(["H", "A"]) & (~res.isna())
            y_win = (res[non_draw_mask] == "H").astype(int)
        else:
            if "is_draw" not in y_train.columns:
                raise ValueError("y_train must include 'result' or 'is_draw' column.")
            y_draw = y_train["is_draw"].astype(int)
            # win_outcome may be bool/int or string
            if "win_outcome" not in y_train.columns:
                raise ValueError("y_train must include 'win_outcome' column when 'is_draw' is provided.")
            wo = y_train["win_outcome"]
            if wo.dtype == bool:
                y_win = wo.astype(int)
                non_draw_mask = (y_draw == 0)
            else:
                # Accept 0/1 or 'HomeWin'/'AwayWin'
                if wo.dtype.kind in {"i", "u"}:
                    y_win = wo.astype(int)
                else:
                    y_win = wo.astype(str).map({"AwayWin": 0, "HomeWin": 1})
                non_draw_mask = (y_draw == 0)

        # Enforce minimum training size
        min_n = int(self.config.model.min_training_matches)
        if len(X_train) < min_n:
            raise ValueError(f"Insufficient training samples: {len(X_train)} < {min_n}")

        # Prepare features
        X_all = self._prepare_features(X_train)

        # Time-decay weighting (optional)
        sample_weight_all = None
        try:
            use_decay = bool(self.config.model.use_time_decay)
            half_life = float(self.config.model.time_decay_half_life_days)
            if use_decay and "date" in X_train.columns and half_life > 0:
                dates = pd.to_datetime(X_train["date"], errors="coerce")
                max_date = pd.to_datetime(dates.max())
                delta_days = (max_date - dates).dt.days.fillna(0).astype(float)
                sample_weight_all = np.power(0.5, delta_days / half_life)
        except Exception:
            sample_weight_all = None

        # Fit label encoders with explicit class order
        self.draw_label_encoder.fit([0, 1])
        self.win_label_encoder.fit([0, 1])

        # Initialize and fit draw classifier
        draw_params = dict(self.config.model.draw_params)
        self.draw_model = XGBClassifier(**draw_params)
        self.draw_model.fit(
            X_all,
            self.draw_label_encoder.transform(y_draw.tolist()),
            sample_weight=sample_weight_all,
            verbose=False,
        )

        # Fit win classifier on non-draw subset
        idx_nd = np.where(non_draw_mask.values if hasattr(non_draw_mask, "values") else non_draw_mask)[0]
        if idx_nd.size == 0:
            raise ValueError("No non-draw samples available to train win classifier.")
        X_win = X_all.iloc[idx_nd]
        y_win_enc = self.win_label_encoder.transform(y_win.tolist())
        y_win_enc = y_win_enc[: len(X_win)]  # align length if needed

        self.win_model = XGBClassifier(**dict(self.config.model.win_params))
        self.win_model.fit(
            X_win,
            y_win_enc,
            verbose=False,
        )

    def predict(self, X_test: pd.DataFrame) -> list[dict]:
        """Make predictions using cascaded approach.

        Args:
            X_test: DataFrame of features for upcoming matches.

        Returns:
            A list of dictionaries per match with keys:
            - 'p_draw', 'p_home', 'p_away': outcome probabilities.
            - 'predicted_outcome': one of 'H', 'D', 'A'.
        Raises:
            RuntimeError if models are not loaded/trained.
        """
        if self.draw_model is None or self.win_model is None:
            raise RuntimeError("Models are not trained/loaded. Call train() or load_models() first.")
        if not isinstance(X_test, pd.DataFrame) or X_test.empty:
            raise ValueError("X_test must be a non-empty pandas DataFrame.")

        X = self._prepare_features(X_test)

        # Draw probabilities
        draw_proba = self.draw_model.predict_proba(X)
        # Identify positive class (1) index
        draw_pos_idx = int(np.where(self.draw_model.classes_ == 1)[0][0])
        p_draw = draw_proba[:, draw_pos_idx]

        # Win probabilities on all rows (model trained on non-draw; but predict for all)
        win_proba = self.win_model.predict_proba(X)
        win_pos_idx = int(np.where(self.win_model.classes_ == 1)[0][0])
        p_home = (1.0 - p_draw) * win_proba[:, win_pos_idx]
        p_away = (1.0 - p_draw) * (1.0 - win_proba[:, win_pos_idx])

        # Determine predicted outcome
        outcomes = np.where(
            p_draw >= np.maximum(p_home, p_away), "D", np.where(p_home >= p_away, "H", "A")
        )

        results: list[dict] = []
        for i in range(len(X)):
            row = X_test.iloc[i]
            pred: dict[str, object] = {
                "home_win_probability": float(p_home[i]),
                "draw_probability": float(p_draw[i]),
                "away_win_probability": float(p_away[i]),
                "predicted_outcome": str(outcomes[i]),
            }
            # Include optional identifiers if present
            for key in ("match_id", "home_team", "away_team", "matchday"):
                if key in X_test.columns:
                    pred[key] = row.get(key)
            results.append(pred)

        # --- Deterministic scoreline heuristic (production-ready baseline) ---
        # Maps predicted outcome to a plausible scoreline deterministically.
        scoreline_map: dict[str, tuple[int, int]] = {
            "H": (2, 1),
            "D": (1, 1),
            "A": (1, 2),
        }
        for r in results:
            outcome = str(r.get("predicted_outcome", "D"))
            home_score, away_score = scoreline_map.get(outcome, (1, 1))
            r["predicted_home_score"] = int(home_score)
            r["predicted_away_score"] = int(away_score)

        return results

    def save_models(self, output_dir: str) -> None:
        """Save trained models and encoders to specified directory.

        Artifacts:
        - 'draw_classifier.joblib': draw XGBClassifier
        - 'win_classifier.joblib': win XGBClassifier
        - 'encoders.joblib': dict with label encoders
        - 'cascaded_metadata.joblib': dict with feature_columns

        Args:
            output_dir: Directory path to store the model artifacts.

        Raises:
            RuntimeError if models are not trained.
        """
        if self.draw_model is None or self.win_model is None:
            raise RuntimeError("No models to save. Must train first.")
        out_path = self._ensure_dir(output_dir)

        joblib.dump(self.draw_model, out_path / "draw_classifier.joblib")
        joblib.dump(self.win_model, out_path / "win_classifier.joblib")
        joblib.dump(
            {
                "draw": self.draw_label_encoder,
                "win": self.win_label_encoder,
            },
            out_path / "encoders.joblib",
        )
        joblib.dump(
            {"feature_columns": list(self.feature_columns)},
            out_path / "cascaded_metadata.joblib",
        )

    def load_models(self, input_dir: str) -> None:
        """Load models and encoders from specified directory.

        Args:
            input_dir: Directory path containing saved artifacts.

        Raises:
            FileNotFoundError if artifacts are missing.
            RuntimeError for invalid metadata or loading errors.
        """
        in_path = self._ensure_dir(input_dir)
        try:
            self.draw_model = joblib.load(in_path / "draw_classifier.joblib")
            self.win_model = joblib.load(in_path / "win_classifier.joblib")
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load models: {exc}")

        try:
            enc = joblib.load(in_path / "encoders.joblib")
            if not isinstance(enc, dict) or "draw" not in enc or "win" not in enc:
                raise RuntimeError("Invalid encoders artifact.")
            self.draw_label_encoder = enc["draw"]
            self.win_label_encoder = enc["win"]
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load encoders: {exc}")

        try:
            meta = joblib.load(in_path / "cascaded_metadata.joblib")
            cols = meta.get("feature_columns") if isinstance(meta, dict) else None
            if not isinstance(cols, list) or len(cols) == 0:
                raise RuntimeError("Missing feature_columns in metadata.")
            self.feature_columns = [str(c) for c in cols]
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load model metadata: {exc}")

    @staticmethod
    def _ensure_dir(path_str: str) -> "Path":
        """Ensure a directory exists and return its Path.

        Args:
            path_str: Directory path as string.

        Returns:
            Path to the directory, created if it did not exist.
        """
        from pathlib import Path

        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        return p