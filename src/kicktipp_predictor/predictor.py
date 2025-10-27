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
from sklearn.model_selection import StratifiedKFold
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
        # Training metrics container
        self.training_metrics: dict[str, dict] = {}

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

    def train(self, matches_df: pd.DataFrame) -> None:
        """Train the two-stage cascaded classifiers with in-method data preparation.

        Data preparation steps (encapsulated):
        - Create 'is_draw' target (1 for draw, 0 otherwise).
        - Filter non-draw matches and create 'is_home_win' (1=home win, 0=away win).
        - Initialize label encoders: draw with [0, 1]; win with ['A', 'H'].
        - Prepare feature matrices for all matches and non-draw subset.

        Defensive programming:
        - Validates non-empty input, presence of 'result' values, and feature columns.
        - Handles missing results by dropping such rows; errors if none remain.
        - Errors when no non-draw samples are available for the win classifier.

        Args:
            matches_df: DataFrame containing features and a 'result' column ('H', 'D', 'A').
        """
        # --- Input validation ---
        if not isinstance(matches_df, pd.DataFrame) or matches_df.empty:
            raise ValueError("Training DataFrame must be a non-empty pandas DataFrame.")
        if "result" not in matches_df.columns:
            raise ValueError("Training DataFrame must include 'result' column with values 'H', 'D', 'A'.")

        # Drop rows with missing result values
        df = matches_df.copy()
        df["result"] = df["result"].astype(str)
        valid_mask = df["result"].isin(["H", "D", "A"]) & (~df["result"].isna())
        df = df.loc[valid_mask]
        if df.empty:
            raise ValueError("No valid rows with 'result' values present for training.")

        # Enforce minimum training size
        min_n = int(self.config.model.min_training_matches)
        if len(df) < min_n:
            raise ValueError(f"Insufficient training samples: {len(df)} < {min_n}")

        # --- Target preparation ---
        # 1) Draw target
        df["is_draw"] = (df["result"] == "D").astype(int)
        # Reflect new target column in the original input (non-destructive for other rows)
        try:
            matches_df.loc[df.index, "is_draw"] = df["is_draw"].values
        except Exception:
            pass
        y_draw = df["is_draw"].astype(int)

        # 2) Win target on non-draw subset
        non_draw_mask = df["is_draw"] == 0
        df_nd = df.loc[non_draw_mask]
        if df_nd.empty:
            raise ValueError("No non-draw samples available to train win classifier.")
        df_nd["is_home_win"] = (df_nd["result"] == "H").astype(int)
        # Reflect new target column in the original input on non-draw rows
        try:
            matches_df.loc[df_nd.index, "is_home_win"] = df_nd["is_home_win"].values
        except Exception:
            pass
        # Map to label strings for encoder ['A','H']
        y_win = np.where(df_nd["is_home_win"].astype(int) == 1, "H", "A")

        # --- Feature preparation ---
        X_all = self._prepare_features(df)
        X_non_draw = self._prepare_features(df_nd)
        if len(X_all) != len(y_draw):
            raise ValueError("Feature matrix and y_draw length mismatch.")
        if len(X_non_draw) != len(y_win):
            raise ValueError("Feature matrix (non-draw) and y_win length mismatch.")

        # --- Time-decay weighting (optional) ---
        sample_weight_all = None
        try:
            use_decay = bool(self.config.model.use_time_decay)
            half_life = float(self.config.model.time_decay_half_life_days)
            if use_decay and "date" in df.columns and half_life > 0:
                dates = pd.to_datetime(df["date"], errors="coerce")
                max_date = pd.to_datetime(dates.max())
                delta_days = (max_date - dates).dt.days.fillna(0).astype(float)
                sample_weight_all = np.power(0.5, delta_days / half_life)
        except Exception:
            sample_weight_all = None

        # --- Label encoders ---
        # Draw: explicit [0,1]
        self.draw_label_encoder.fit([0, 1])
        # Win: explicit ['A','H'] for away/home win classes
        self.win_label_encoder.fit(["A", "H"])

        # --- Initialize classifiers ---
        draw_params = dict(self.config.model.draw_params)
        win_params = dict(self.config.model.win_params)

        # --- Prepare simple validation split for early stopping ---
        val_fraction = float(self.config.model.val_fraction)
        # Draw split (stratified if both classes present)
        y_draw_arr = y_draw.to_numpy()
        if 0.0 < val_fraction < 0.5 and len(X_all) >= 10 and len(np.unique(y_draw_arr)) > 1:
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(self.config.model.random_state))
                # Take the first split as validation for simplicity
                train_idx, val_idx = next(skf.split(X_all, y_draw_arr))
            except Exception:
                n = len(X_all)
                val_n = int(n * val_fraction)
                train_idx = np.arange(n - val_n)
                val_idx = np.arange(n - val_n, n)
        else:
            train_idx = np.arange(len(X_all))
            val_idx = np.array([], dtype=int)

        Xd_train = X_all.iloc[train_idx]
        yd_train = self.draw_label_encoder.transform(y_draw.iloc[train_idx].tolist())
        Xd_val = X_all.iloc[val_idx] if len(val_idx) else None
        yd_val = self.draw_label_encoder.transform(y_draw.iloc[val_idx].tolist()) if len(val_idx) else None

        # --- Draw model training ---
        print("Training Draw Classifier (Gatekeeper)...")
        try:
            self.draw_model = XGBClassifier(**draw_params)
            if Xd_val is not None:
                self.draw_model.fit(
                    Xd_train,
                    yd_train,
                    sample_weight=sample_weight_all[train_idx] if sample_weight_all is not None else None,
                    eval_set=[(Xd_train, yd_train), (Xd_val, yd_val)],
                    eval_metric="logloss",
                    early_stopping_rounds=20,
                    verbose=True,
                )
            else:
                self.draw_model.fit(
                    X_all,
                    self.draw_label_encoder.transform(y_draw.tolist()),
                    sample_weight=sample_weight_all,
                    eval_set=[(X_all, self.draw_label_encoder.transform(y_draw.tolist()))],
                    eval_metric="logloss",
                    verbose=True,
                )
            # Save training metrics
            self.training_metrics["draw_model"] = {
                "train_score": float(self.draw_model.score(X_all, self.draw_label_encoder.transform(y_draw.tolist()))),
                "feature_importances": self.draw_model.feature_importances_.tolist() if hasattr(self.draw_model, "feature_importances_") else None,
                "class_counts": {
                    "draw": int((y_draw_arr == 1).sum()),
                    "not_draw": int((y_draw_arr == 0).sum()),
                },
            }
        except Exception as e:
            print(f"Error training Draw Model: {str(e)}")
            raise

        # --- Win model training ---
        print("Training Win Classifier (Finisher)...")
        try:
            self.win_model = XGBClassifier(**win_params)
            # Validation split for non-draw
            y_win_arr = np.array(list(y_win))
            # Encode for fitting (numeric expected)
            y_win_enc = self.win_label_encoder.transform(list(y_win))
            if 0.0 < val_fraction < 0.5 and len(X_non_draw) >= 10 and len(np.unique(y_win_enc)) > 1:
                try:
                    skf_w = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(self.config.model.random_state))
                    w_train_idx, w_val_idx = next(skf_w.split(X_non_draw, y_win_enc))
                except Exception:
                    n_w = len(X_non_draw)
                    val_n_w = int(n_w * val_fraction)
                    w_train_idx = np.arange(n_w - val_n_w)
                    w_val_idx = np.arange(n_w - val_n_w, n_w)
            else:
                w_train_idx = np.arange(len(X_non_draw))
                w_val_idx = np.array([], dtype=int)

            Xw_train = X_non_draw.iloc[w_train_idx]
            yw_train = y_win_enc[w_train_idx]
            Xw_val = X_non_draw.iloc[w_val_idx] if len(w_val_idx) else None
            yw_val = y_win_enc[w_val_idx] if len(w_val_idx) else None

            if Xw_val is not None:
                self.win_model.fit(
                    Xw_train,
                    yw_train,
                    eval_set=[(Xw_train, yw_train), (Xw_val, yw_val)],
                    eval_metric="logloss",
                    early_stopping_rounds=20,
                    verbose=True,
                )
            else:
                self.win_model.fit(
                    X_non_draw,
                    y_win_enc,
                    eval_set=[(X_non_draw, y_win_enc)],
                    eval_metric="logloss",
                    verbose=True,
                )
            self.training_metrics["win_model"] = {
                "train_score": float(self.win_model.score(X_non_draw, y_win_enc)),
                "feature_importances": self.win_model.feature_importances_.tolist() if hasattr(self.win_model, "feature_importances_") else None,
                "class_counts": {
                    "home": int((y_win_arr == "H").sum()),
                    "away": int((y_win_arr == "A").sum()),
                },
            }
        except Exception as e:
            print(f"Error training Win Model: {str(e)}")
            raise

        # --- Cross-validation summaries (quick sanity checks) ---
        try:
            cv_draw_scores: list[float] = []
            if len(np.unique(y_draw_arr)) > 1:
                skf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(self.config.model.random_state))
                for tr_idx, te_idx in skf_cv.split(X_all, y_draw_arr):
                    mdl = XGBClassifier(**draw_params)
                    mdl.fit(X_all.iloc[tr_idx], self.draw_label_encoder.transform(y_draw.iloc[tr_idx].tolist()), verbose=False)
                    s = mdl.score(X_all.iloc[te_idx], self.draw_label_encoder.transform(y_draw.iloc[te_idx].tolist()))
                    cv_draw_scores.append(float(s))
            self.training_metrics.setdefault("draw_model", {})["cv_accuracy"] = {
                "mean": float(np.mean(cv_draw_scores)) if cv_draw_scores else None,
                "std": float(np.std(cv_draw_scores)) if cv_draw_scores else None,
                "n_splits": int(len(cv_draw_scores)),
            }
        except Exception:
            # Non-fatal; skip CV on errors
            self.training_metrics.setdefault("draw_model", {})["cv_accuracy"] = None

        try:
            cv_win_scores: list[float] = []
            y_win_enc_full = self.win_label_encoder.transform(list(y_win))
            if len(np.unique(y_win_enc_full)) > 1 and len(X_non_draw) >= 6:
                skf_cv_w = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(self.config.model.random_state))
                for tr_idx, te_idx in skf_cv_w.split(X_non_draw, y_win_enc_full):
                    mdl = XGBClassifier(**win_params)
                    mdl.fit(X_non_draw.iloc[tr_idx], y_win_enc_full[tr_idx], verbose=False)
                    s = mdl.score(X_non_draw.iloc[te_idx], y_win_enc_full[te_idx])
                    cv_win_scores.append(float(s))
            self.training_metrics.setdefault("win_model", {})["cv_accuracy"] = {
                "mean": float(np.mean(cv_win_scores)) if cv_win_scores else None,
                "std": float(np.std(cv_win_scores)) if cv_win_scores else None,
                "n_splits": int(len(cv_win_scores)),
            }
        except Exception:
            self.training_metrics.setdefault("win_model", {})["cv_accuracy"] = None

        # --- Post-training summary ---
        summary = {
            "n_samples_all": int(len(X_all)),
            "n_samples_non_draw": int(len(X_non_draw)),
            "feature_count": int(len(self.feature_columns)),
            "label_classes": {
                "draw": list(map(int, self.draw_label_encoder.classes_.tolist() if hasattr(self.draw_label_encoder, "classes_") else [0, 1])),
                "win": [str(c) for c in (self.win_label_encoder.classes_.tolist() if hasattr(self.win_label_encoder, "classes_") else ["A", "H"])],
            },
        }
        self.training_metrics["summary"] = summary
        print("Training complete. Summary:")
        print({
            "features": summary["feature_count"],
            "samples": {
                "all": summary["n_samples_all"],
                "non_draw": summary["n_samples_non_draw"],
            },
            "cv": {
                "draw": self.training_metrics.get("draw_model", {}).get("cv_accuracy"),
                "win": self.training_metrics.get("win_model", {}).get("cv_accuracy"),
            },
        })

    def predict(self, X_test: pd.DataFrame) -> list[dict]:
        """Make predictions using two-stage probability combination.

        Probability combination:
        - P(NotDraw) = 1 - P(Draw)
        - P(Home) = P(NotDraw) * P(Home | NotDraw)
        - P(Away) = P(NotDraw) * (1 - P(Home | NotDraw))
        - P(Draw) unchanged

        Args:
            X_test: DataFrame of features for upcoming matches.

        Returns:
            A list of dictionaries per match including probabilities and labels.
        Raises:
            RuntimeError if models are not loaded/trained.
            ValueError for invalid inputs or feature mismatches.
        """
        if self.draw_model is None or self.win_model is None:
            raise RuntimeError("Models are not trained/loaded. Call train() or load_models() first.")
        if not isinstance(X_test, pd.DataFrame) or X_test.empty:
            raise ValueError("X_test must be a non-empty pandas DataFrame.")

        # Ensure feature columns are available and match model expectations
        if not self.feature_columns:
            raise ValueError("Feature columns are not set. Train or load models before predicting.")

        X = self._prepare_features(X_test)
        # Defensive check against model's expected input dimension
        for mdl_name, mdl in (("draw", self.draw_model), ("win", self.win_model)):
            n_in = getattr(mdl, "n_features_in_", None)
            if n_in is not None and int(n_in) != X.shape[1]:
                raise ValueError(f"Feature dimension mismatch for {mdl_name} model: {X.shape[1]} != {int(n_in)}")

        # 2. Draw probabilities
        print("Calculating Draw probabilities...")
        draw_proba = self.draw_model.predict_proba(X)
        # Map encoder label to index within model classes_
        try:
            draw_label = int(self.draw_label_encoder.transform([1])[0])
            draw_class_index = int(np.where(self.draw_model.classes_ == draw_label)[0][0])
        except Exception:
            # Fallback to assuming label '1' is positive class
            draw_class_index = int(np.where(self.draw_model.classes_ == 1)[0][0])
        p_draw = draw_proba[:, draw_class_index]

        # 3. Win probabilities
        print("Calculating conditional Win probabilities...")
        win_proba = self.win_model.predict_proba(X)
        # Identify 'H' class index via encoder mapping
        try:
            h_label = int(self.win_label_encoder.transform(["H"])[0])
            home_class_index = int(np.where(self.win_model.classes_ == h_label)[0][0])
        except Exception:
            home_class_index = int(np.where(self.win_model.classes_ == 1)[0][0])
        p_home_given_not_draw = win_proba[:, home_class_index]

        # 4. Combine probabilities via law of total probability
        print("Combining probabilities using law of total probability...")
        p_not_draw = 1.0 - p_draw
        final_p_home = p_not_draw * p_home_given_not_draw
        final_p_away = p_not_draw * (1.0 - p_home_given_not_draw)
        final_p_draw = p_draw

        # 5. Assemble final matrix and normalize rows
        final_probs = np.vstack([final_p_home, final_p_draw, final_p_away]).T
        # Correct minor floating errors by normalization
        row_sums = final_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero in degenerate cases
        row_sums[row_sums == 0.0] = 1.0
        final_probs = final_probs / row_sums

        # Determine predicted outcome from normalized probs
        outcomes = np.where(
            final_probs[:, 1] >= np.maximum(final_probs[:, 0], final_probs[:, 2]),
            "D",
            np.where(final_probs[:, 0] >= final_probs[:, 2], "H", "A"),
        )

        results: list[dict] = []
        for i in range(len(X)):
            row = X_test.iloc[i]
            pred: dict[str, object] = {
                "home_win_probability": float(final_probs[i, 0]),
                "draw_probability": float(final_probs[i, 1]),
                "away_win_probability": float(final_probs[i, 2]),
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

    def save_models(self) -> None:
        """Save both classification models and their metadata to disk.

        Uses `config.paths.models_dir` as the target directory and stores:
        - 'draw_classifier.joblib'
        - 'win_classifier.joblib'
        - 'metadata_v4.joblib' (consolidated metadata with encoders)

        Raises:
            ValueError: If models are not initialized.
            RuntimeError: If file operations fail.
        """
        # Validate models exist
        if not hasattr(self, "draw_model") or not hasattr(self, "win_model"):
            raise ValueError("Models not initialized")
        if self.draw_model is None or self.win_model is None:
            raise ValueError("Models not initialized")

        # Ensure target directory exists
        out_path = self.config.paths.models_dir
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to ensure models directory: {exc}")

        # Save models
        try:
            joblib.dump(self.draw_model, out_path / "draw_classifier.joblib")
            joblib.dump(self.win_model, out_path / "win_classifier.joblib")
        except Exception as exc:
            raise RuntimeError(f"Failed to save models: {exc}")

        # Save shared metadata (consolidated, versioned)
        metadata = {
            "version": "v4",
            "feature_columns": list(self.feature_columns),
            "draw_label_encoder": self.draw_label_encoder,
            "win_label_encoder": self.win_label_encoder,
        }
        try:
            joblib.dump(metadata, out_path / "metadata_v4.joblib")
        except Exception as exc:
            raise RuntimeError(f"Failed to save metadata: {exc}")

    def load_models(self) -> None:
        """Load both models and their metadata from disk.

        Uses `config.paths.models_dir` and supports backward compatibility
        with pre-v4 artifacts.

        Raises:
            FileNotFoundError: If required artifacts are missing.
            RuntimeError: For invalid metadata or loading errors.
        """
        in_path = self.config.paths.models_dir
        try:
            in_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to ensure models directory: {exc}")

        # Load draw and win models (existence check first)
        draw_path = in_path / "draw_classifier.joblib"
        win_path = in_path / "win_classifier.joblib"
        if not draw_path.exists():
            raise FileNotFoundError(f"Missing draw model: {draw_path}")
        if not win_path.exists():
            raise FileNotFoundError(f"Missing win model: {win_path}")
        try:
            self.draw_model = joblib.load(draw_path)
            self.win_model = joblib.load(win_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load models: {exc}")

        # Load consolidated v4 metadata if present; otherwise fallback to legacy artifacts
        meta_v4_path = in_path / "metadata_v4.joblib"
        if meta_v4_path.exists():
            try:
                metadata = joblib.load(meta_v4_path)
                if not isinstance(metadata, dict):
                    raise RuntimeError("Invalid metadata format.")
                cols = metadata.get("feature_columns")
                if not isinstance(cols, list) or len(cols) == 0:
                    raise RuntimeError("Missing feature_columns in metadata.")
                self.feature_columns = [str(c) for c in cols]
                dle = metadata.get("draw_label_encoder")
                wle = metadata.get("win_label_encoder")
                if dle is None or wle is None:
                    raise RuntimeError("Missing label encoders in metadata.")
                self.draw_label_encoder = dle
                self.win_label_encoder = wle
            except Exception as exc:
                raise RuntimeError(f"Failed to load v4 metadata: {exc}")
        else:
            # Backward compatibility: legacy separate artifacts
            try:
                enc_path = in_path / "encoders.joblib"
                meta_path = in_path / "cascaded_metadata.joblib"
                if not enc_path.exists() or not meta_path.exists():
                    raise FileNotFoundError(
                        "Legacy artifacts missing: encoders.joblib and/or cascaded_metadata.joblib"
                    )
                enc = joblib.load(enc_path)
                if not isinstance(enc, dict) or "draw" not in enc or "win" not in enc:
                    raise RuntimeError("Invalid encoders artifact.")
                self.draw_label_encoder = enc["draw"]
                self.win_label_encoder = enc["win"]

                meta = joblib.load(meta_path)
                cols = meta.get("feature_columns") if isinstance(meta, dict) else None
                if not isinstance(cols, list) or len(cols) == 0:
                    raise RuntimeError("Missing feature_columns in metadata.")
                self.feature_columns = [str(c) for c in cols]
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Failed to load legacy artifacts: {exc}")

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