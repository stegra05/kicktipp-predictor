"""
Predictor implementations for the Kicktipp Predictor project.

Includes:
- CascadedPredictor (V4 cascaded classifiers: draw + win)
"""

from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBClassifier
# EarlyStopping callback is not used due to current xgboost .fit signature
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from typing import Optional

from .config import Config, get_config


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
        # Deterministic mapping for predict_proba columns
        # {'draw_positive': idx, 'win_home': idx, 'win_away': idx}
        self.class_index_map: dict[str, int] = {}

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

    def _prepare_targets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Prepare target variables for cascaded training.

        Validates input and constructs:
        - 'is_draw' for all rows (1 if result == 'D')
        - 'is_home_win' for non-draw rows (1 if result == 'H')

        Returns (df_with_targets, y_draw_series, y_win_array_of_str).
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame.")
        if "result" not in df.columns:
            raise ValueError("DataFrame must include a 'result' column.")

        df = df.copy()
        res = df["result"].astype(str).str.upper()
        valid = {"H", "D", "A"}
        if (~res.isin(valid)).any():
            raise ValueError("Invalid values in 'result'; expected only 'H','D','A'.")

        df["is_draw"] = (res == "D").astype(int)
        y_draw = df["is_draw"].astype(int)

        non_draw_mask = df["is_draw"] == 0
        df_nd = df.loc[non_draw_mask].copy()
        if df_nd.empty:
            raise ValueError("No non-draw samples available to train win classifier.")
        df_nd.loc[:, "is_home_win"] = (df_nd["result"] == "H").astype(int)
        y_win = np.where(df_nd["is_home_win"].astype(int) == 1, "H", "A")

        # Merge back is_home_win for completeness
        df.loc[df_nd.index, "is_home_win"] = df_nd["is_home_win"].values

        return df, y_draw, y_win

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
        df, y_draw, y_win = self._prepare_targets(df)
        # Reflect targets back to original dataframe view (best-effort)
        try:
            matches_df.loc[df.index, "is_draw"] = df["is_draw"].values
            if "is_home_win" in df.columns:
                matches_df.loc[df.index, "is_home_win"] = df["is_home_win"].values
        except Exception:
            pass

        # --- Feature preparation ---
        X_all = self._prepare_features(df)
        # Define non-draw subset locally (aligns with y_win created in _prepare_targets)
        non_draw_mask = df["is_draw"] == 0
        df_nd = df.loc[non_draw_mask].copy()
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
        # Explicit ordering ensures deterministic mapping (positive class index = 1)
        self.draw_label_encoder.fit([0, 1])
        self.win_label_encoder.fit(["A", "H"])
        # Predefine class index mapping (avoid runtime reliance on model.classes_)
        self.class_index_map = {"draw_positive": 1, "win_home": 1, "win_away": 0}

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
            # Set evaluation metric and logging verbosity
            self.draw_model = XGBClassifier(
                eval_metric=self.config.model.eval_metric,
                verbosity=0 if not self.config.model.fit_verbose else 1,
                **draw_params,
            )
            if Xd_val is not None:
                # Try early stopping if supported by installed xgboost
                try:
                    self.draw_model.fit(
                        Xd_train,
                        yd_train,
                        sample_weight=sample_weight_all[train_idx] if sample_weight_all is not None else None,
                        eval_set=[(Xd_train, yd_train), (Xd_val, yd_val)],
                        early_stopping_rounds=int(self.config.model.early_stopping_rounds),
                        verbose=self.config.model.fit_verbose,
                    )
                except TypeError:
                    self.draw_model.fit(
                        Xd_train,
                        yd_train,
                        sample_weight=sample_weight_all[train_idx] if sample_weight_all is not None else None,
                        eval_set=[(Xd_train, yd_train), (Xd_val, yd_val)],
                        verbose=self.config.model.fit_verbose,
                    )
            else:
                self.draw_model.fit(
                    X_all,
                    self.draw_label_encoder.transform(y_draw.tolist()),
                    sample_weight=sample_weight_all,
                    eval_set=[(X_all, self.draw_label_encoder.transform(y_draw.tolist()))],
                    verbose=self.config.model.fit_verbose,
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
            # Set evaluation metric and logging verbosity
            self.win_model = XGBClassifier(
                eval_metric=self.config.model.eval_metric,
                verbosity=0 if not self.config.model.fit_verbose else 1,
                **win_params,
            )
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
                try:
                    self.win_model.fit(
                        Xw_train,
                        yw_train,
                        eval_set=[(Xw_train, yw_train), (Xw_val, yw_val)],
                        early_stopping_rounds=int(self.config.model.early_stopping_rounds),
                        verbose=self.config.model.fit_verbose,
                    )
                except TypeError:
                    self.win_model.fit(
                        Xw_train,
                        yw_train,
                        eval_set=[(Xw_train, yw_train), (Xw_val, yw_val)],
                        verbose=self.config.model.fit_verbose,
                    )
            else:
                self.win_model.fit(
                    X_non_draw,
                    y_win_enc,
                    eval_set=[(X_non_draw, y_win_enc)],
                    verbose=self.config.model.fit_verbose,
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
        draw_class_index = int(self.class_index_map.get("draw_positive", 1))
        if draw_class_index < 0 or draw_class_index >= draw_proba.shape[1]:
            # Fallback to encoder/classes_ mapping if available
            try:
                draw_label = int(self.draw_label_encoder.transform([1])[0])
                draw_class_index = int(np.where(getattr(self.draw_model, "classes_", np.array([0, 1])) == draw_label)[0][0])
            except Exception:
                draw_class_index = 1
        p_draw = draw_proba[:, draw_class_index]

        # 3. Win probabilities
        print("Calculating conditional Win probabilities...")
        win_proba = self.win_model.predict_proba(X)
        home_class_index = int(self.class_index_map.get("win_home", 1))
        if home_class_index < 0 or home_class_index >= win_proba.shape[1]:
            try:
                h_label = int(self.win_label_encoder.transform(["H"])[0])
                home_class_index = int(np.where(getattr(self.win_model, "classes_", np.array([0, 1])) == h_label)[0][0])
            except Exception:
                home_class_index = 1
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
            "class_index_map": dict(self.class_index_map),
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
                cim = metadata.get("class_index_map")
                if isinstance(cim, dict):
                    self.class_index_map = {k: int(v) for k, v in cim.items() if k in ("draw_positive", "win_home", "win_away")}
                else:
                    # Default deterministic mapping
                    self.class_index_map = {"draw_positive": 1, "win_home": 1, "win_away": 0}
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