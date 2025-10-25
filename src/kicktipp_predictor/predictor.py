# Version 3.0: Simplified architecture (2025-10-24)
"""Implements a two-stage match predictor for football games.

This module defines the `MatchPredictor`, a class that uses a predictor-selector
architecture to forecast match outcomes and scorelines. The process involves:
1.  **Outcome Prediction (Selector):** An XGBoost classifier predicts the
    match result (Home Win, Draw, or Away Win).
2.  **Scoreline Selection (Predictor):** Two XGBoost regressors estimate the
    expected goals for each team. These lambda values are then used with a
    Poisson distribution to find the most probable scoreline that matches the
    predicted outcome.
"""

# === Imports ===
import time

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

# Optional YAML import for feature selection file
try:
    import yaml
except Exception:
    yaml = None

from .config import get_config
from .scoring import compute_ep_scoreline

# === Scoreline Selection ===


def compute_ep_scoreline(
    home_lambda: float,
    away_lambda: float,
    max_goals: int,
) -> tuple[int, int]:
    """Delegate to scoring.compute_ep_scoreline for EP-max scoreline selection."""
    from .scoring import compute_ep_scoreline as _compute
    return _compute(home_lambda, away_lambda, max_goals)


# === MatchPredictor ===
class MatchPredictor:
    """A two-step predictor for match outcomes and scorelines.

    This class trains and manages three models:
    - An XGBClassifier for predicting match outcomes (H/D/A).
    - Two XGBRegressors for predicting expected goals for home and away teams.

    Predictions are made by first selecting an outcome and then finding the most
    likely scoreline that aligns with that outcome, using a Poisson distribution.

    Attributes:
        config: Configuration object.
        quiet: A flag to suppress log output.
        outcome_model: The trained outcome classifier.
        home_goals_model: The trained home goals regressor.
        away_goals_model: The trained away goals regressor.
        feature_columns: A list of feature names used for training.
        label_encoder: A LabelEncoder for the outcome variable.
    """

    HYBRID_WEIGHT: float = 0.0525

    def __init__(self, quiet: bool = False):
        """Initializes the MatchPredictor with its configuration."""
        self.config = get_config()
        self.quiet = quiet
        self.outcome_model: XGBClassifier | None = None
        self.home_goals_model: XGBRegressor | None = None
        self.away_goals_model: XGBRegressor | None = None
        self.feature_columns: list[str] = []
        self.label_encoder = LabelEncoder()

        self.console = Console()

    def _log(self, *args, **kwargs) -> None:
        if not self.quiet:
            self.console.log(*args, **kwargs)

    def _fit_xgb_regressor(
        self,
        model: XGBRegressor,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Fit a single XGBRegressor with optional sample weights.

        Args:
            model: The regressor instance to fit.
            X_tr: Training features.
            y_tr: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            sample_weights: Optional per-sample weights.
        """
        fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weights is not None:
            fit_params["sample_weight"] = sample_weights
        model.fit(X_tr, y_tr, **fit_params)

    def _load_selected_features(self) -> list[str] | None:
        """Load selected feature names from YAML (config/selected_features_file)."""
        try:
            sel_filename = getattr(
                self.config.model, "selected_features_file", "kept_features.yaml"
            )
            sel_path = self.config.paths.config_dir / sel_filename
            if not sel_path.exists():
                return None
            if yaml is not None:
                with open(sel_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, list):
                    return [str(c) for c in loaded]
                if isinstance(loaded, dict) and "features" in loaded:
                    val = loaded.get("features")
                    if isinstance(val, list):
                        return [str(c) for c in val]
                    if isinstance(val, str):
                        return [s.strip() for s in val.splitlines() if s.strip()]
                if isinstance(loaded, str):
                    return [s.strip() for s in loaded.splitlines() if s.strip()]
            else:
                txt_path = sel_path.with_suffix(".txt")
                if txt_path.exists():
                    return [
                        line.strip()
                        for line in txt_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
        except Exception:
            return None
        return None

    def train(self, matches_df: pd.DataFrame):
        """Trains the outcome classifier and goal regressors.

        Args:
            matches_df: A DataFrame containing match features and results.
        """
        training_data = matches_df[matches_df["home_score"].notna()].copy()

        if len(training_data) < self.config.model.min_training_matches:
            self._log(
                f"Insufficient training data: {len(training_data)} matches found, "
                f"but {self.config.model.min_training_matches} are required."
            )
            return

        exclude_cols = [
            "match_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "goal_difference",
            "result",
            "is_finished",
        ]
        # Strict feature selection: only use features listed in kept_features.yaml
        selected = self._load_selected_features() or []
        feature_df = training_data.drop(columns=exclude_cols, errors="ignore")
        # Candidate numeric/bool columns
        candidate_cols = feature_df.select_dtypes(
            include=[np.number, bool]
        ).columns.tolist()
        if selected:
            self.feature_columns = [c for c in candidate_cols if c in set(selected)]
            missing = [c for c in selected if c not in self.feature_columns]
            if missing:
                self._log(
                    f"Warning: {len(missing)} selected features missing from training data: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
        else:
            # Fallback to prior permissive behavior if selection file isn't available
            self.feature_columns = candidate_cols

        if "tanh_tamed_elo" not in self.feature_columns:
            self._log(
                "WARNING: 'tanh_tamed_elo' not found in feature columns; "
                "verify feature engineering and YAML selection."
            )
        if len(self.feature_columns) < 10:
            self._log(
                f"WARNING: Very few features selected "
                f"({len(self.feature_columns)}). "
                "Check kept_features.yaml wiring and data completeness."
            )

        X = training_data[self.feature_columns].fillna(0)
        y_home = training_data["home_score"]
        y_away = training_data["away_score"]
        y_result = training_data["result"]
        y_result_encoded = self.label_encoder.fit_transform(y_result)

        time_weights = self._compute_time_decay_weights(training_data)

        counts = y_result.value_counts()
        self._log(
            f"Training on {len(training_data)} matches with "
            f"{len(self.feature_columns)} features."
        )

        dates = (
            pd.to_datetime(training_data["date"], errors="coerce")
            if "date" in training_data.columns
            else None
        )
        with self.console.status("Training goal regressors...", spinner="dots"):
            self._train_goal_models(
                X, y_home, y_away, sample_weights=time_weights, dates=dates
            )
        with self.console.status("Training outcome classifier...", spinner="dots"):
            val_ctx = self._train_outcome_model(X, y_result_encoded, time_weights)




        self._log("Training completed.")

    # --- Training Utilities ---
    def _compute_time_decay_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Calculates time-decay weights to prioritize more recent matches."""
        if not self.config.model.use_time_decay or "date" not in df.columns:
            return np.ones(len(df))

        dates = pd.to_datetime(df["date"])
        days_old = (dates.max() - dates).dt.days.astype(float)
        half_life = float(self.config.model.time_decay_half_life_days)
        decay_rate = np.log(2.0) / max(1.0, half_life)
        return np.exp(-decay_rate * days_old.values)

    def _train_goal_models(
        self,
        X: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
        sample_weights: np.ndarray | None,
        dates: pd.Series | None,
    ):
        """Trains the home and away goal regression models."""
        start = time.perf_counter()

        # Hardcode low-impact params to defaults for simplicity
        goals_params = {
            **self.config.model.goals_params,
            "reg_alpha": 0.0,
            "gamma": 0.0,
            "colsample_bytree": 0.8,
        }
        self.home_goals_model = XGBRegressor(**goals_params)
        self.away_goals_model = XGBRegressor(**goals_params)

        train_mask = self._get_train_validation_split(X, dates)
        X_tr, X_val = X[train_mask], X[~train_mask]
        yh_tr, yh_val = y_home[train_mask], y_home[~train_mask]
        ya_tr, ya_val = y_away[train_mask], y_away[~train_mask]
        sw_tr = sample_weights[train_mask] if sample_weights is not None else None

        self._fit_xgb_regressor(
            self.home_goals_model, X_tr, yh_tr, X_val, yh_val, sw_tr
        )
        self._fit_xgb_regressor(
            self.away_goals_model, X_tr, ya_tr, X_val, ya_val, sw_tr
        )

        elapsed = time.perf_counter() - start
        self._log(f"Goal regressors trained in {elapsed:.2f}s")

    def _get_train_validation_split(
        self, X: pd.DataFrame, dates: pd.Series | None
    ) -> np.ndarray:
        """Creates a train/validation split, time-based if possible."""
        if dates is not None and not dates.isnull().all():
            cutoff = dates.quantile(0.9)
            return dates < cutoff

        _, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.1,
            random_state=self.config.model.random_state,
        )
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[val_idx] = False
        return train_mask

    def _train_outcome_model(
        self, X: pd.DataFrame, y_result_encoded: np.ndarray, time_weights: np.ndarray
    ):
        """Trains the outcome classification model."""
        start = time.perf_counter()

        X_train, X_val, y_train, y_val, tw_train, tw_val = train_test_split(
            X,
            y_result_encoded,
            time_weights,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y_result_encoded,
        )

        balanced_weights = compute_sample_weight("balanced", y=y_train)
        draw_boost = float(self.config.model.draw_boost)
        if "D" in self.label_encoder.classes_ and draw_boost != 1.0:
            draw_class_label = np.where(self.label_encoder.classes_ == "D")[0][0]
            boost_weights = np.where(y_train == draw_class_label, draw_boost, 1.0)
        else:
            boost_weights = np.ones(len(y_train))

        sample_weights = balanced_weights * tw_train * boost_weights

        # Hardcode low-impact params to defaults for simplicity
        outcome_params = {
            **self.config.model.outcome_params,
            "reg_alpha": 0.0,
            "gamma": 0.0,
            "colsample_bytree": 0.8,
        }
        self.outcome_model = XGBClassifier(**outcome_params)
        self.outcome_model.fit(
            X_train, y_train, sample_weight=sample_weights, verbose=False
        )

        self._log(f"Outcome classifier trained in {time.perf_counter() - start:.2f}s")
        # Return validation context for calibrator fitting
        try:
            cls_val_probs = self.outcome_model.predict_proba(X_val)
        except Exception:
            cls_val_probs = None
        return {"X_val": X_val, "y_val_enc": y_val, "cls_val_probs": cls_val_probs}

    MIN_LAMBDA: float = 0.2  # clamp for expected goals to avoid degenerate predictions

    def predict(
        self, features_df: pd.DataFrame
    ) -> list[dict]:
        """Predicts match outcomes and scorelines.

        Args:
            features_df: A DataFrame with match features.

        Returns:
            A list of prediction dictionaries.
        """
        if not all([
            self.outcome_model,
            self.home_goals_model,
            self.away_goals_model,
        ]):
            raise ValueError(
                "Models must be trained via train() or loaded via "
                "load_models() before prediction."
            )

        X = self._prepare_features(features_df)
        classifier_probs = self.outcome_model.predict_proba(X)
        home_lambdas = np.maximum(
            self.home_goals_model.predict(X), self.MIN_LAMBDA
        )
        away_lambdas = np.maximum(
            self.away_goals_model.predict(X), self.MIN_LAMBDA
        )

        final_probs, diag = self._derive_final_probabilities(
            classifier_probs, home_lambdas, away_lambdas
        )

        # Base outcomes: argmax over final probabilities
        classes = self.label_encoder.classes_
        prob_map = {label: i for i, label in enumerate(classes)}
        outcomes_idx = np.argmax(final_probs, axis=1)

        outcomes = self.label_encoder.inverse_transform(outcomes_idx)

        # Compute scorelines using EP-only selection
        scorelines = self._compute_ep_scorelines(
            home_lambdas, away_lambdas
        )

        return self._format_predictions(
            features_df,
            outcomes,
            scorelines,
            final_probs,
            home_lambdas,
            away_lambdas,
            diag,
        )

    # --- Prediction Assembly ---
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns DataFrame columns with the model's feature schema."""
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_columns].fillna(0)


    # --- New: EP scoreline computation ---
    def _compute_ep_scorelines(self, home_lambdas, away_lambdas):
        """Computes EP-maximizing scorelines for each match (no parallelism)."""
        G = int(self.config.model.max_goals)
        n = len(home_lambdas)
        return [
            compute_ep_scoreline(home_lambdas[i], away_lambdas[i], G)
            for i in range(n)
        ]

    # --- Probability Utilities ---
    def _derive_final_probabilities(self, classifier_probs, home_lambdas, away_lambdas):
        """Derives final outcome probabilities using hybrid-only blending.

        Returns:
            A tuple containing the final probabilities and a dictionary with
            diagnostic information.
        """
        diag = {"classifier_probs": classifier_probs, "poisson_probs": np.zeros_like(classifier_probs)}

        poisson_probs = self._calculate_poisson_outcome_probs(home_lambdas, away_lambdas)
        diag["poisson_probs"] = poisson_probs
        diag["weights"] = np.full(len(classifier_probs), self.HYBRID_WEIGHT)

        blended = self._blend_probs(classifier_probs, poisson_probs)
        return blended, diag


    def _blend_probs(
        self,
        cls: np.ndarray,
        pois: np.ndarray,
        weights: np.ndarray | float | None = None,
    ) -> np.ndarray:
        if weights is None or isinstance(weights, float):
            w = float(weights) if weights is not None else float(self.HYBRID_WEIGHT)
            blend = (1.0 - w) * cls + w * pois
        else:
            w = weights.reshape(-1, 1)
            blend = (1.0 - w) * cls + w * pois
        blend = np.clip(blend, 1e-15, 1.0)
        return blend / blend.sum(axis=1, keepdims=True)

    def _calculate_poisson_outcome_probs(self, home_lambdas, away_lambdas):
        """Outcome probabilities from Poisson lambdas with per-match dynamic grid."""
        n = len(home_lambdas)
        probs = np.zeros((n, 3), dtype=float)
        max_cap = int(self.config.model.max_goals)
        for i in range(n):
            lam_h = float(home_lambdas[i])
            lam_a = float(away_lambdas[i])
            G_cap = int(max_cap)
            G_dynamic = int(np.ceil(max(lam_h, lam_a) + 4))
            G = int(min(G_cap, G_dynamic))
            x = np.arange(G + 1)
            ph = poisson.pmf(x, lam_h)
            pa = poisson.pmf(x, lam_a)
            grid = np.outer(ph, pa)
            total = np.sum(grid)
            if total <= 0:
                probs[i] = np.array([1 / 3, 1 / 3, 1 / 3])
                continue
            grid /= total
            upper = np.triu(grid, k=1)
            lower = np.tril(grid, k=-1)
            diag = np.diag(np.diag(grid))
            pH = float(np.sum(upper))
            pD = float(np.sum(np.diag(grid)))
            pA = float(np.sum(lower))
            probs[i] = np.array([pH, pD, pA])
        probs = np.clip(probs, 1e-15, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)


    def _format_predictions(
        self,
        df,
        outcomes,
        scorelines,
        final_probs,
        home_lambdas,
        away_lambdas,
        diag: dict,
    ):
        """Assembles the final list of prediction dictionaries."""
        predictions = []
        classes = self.label_encoder.classes_
        prob_map = {label: i for i, label in enumerate(classes)}

        for i in range(len(df)):
            probs_sorted = sorted(final_probs[i], reverse=True)
            pred = {
                "match_id": df.iloc[i]["match_id"],
                "home_team": df.iloc[i]["home_team"],
                "away_team": df.iloc[i]["away_team"],
                "predicted_home_score": int(scorelines[i][0]),
                "predicted_away_score": int(scorelines[i][1]),
                "home_expected_goals": float(home_lambdas[i]),
                "away_expected_goals": float(away_lambdas[i]),
                "predicted_result": outcomes[i],
                "home_win_probability": float(final_probs[i][prob_map.get("H", 0)]),
                "draw_probability": float(final_probs[i][prob_map.get("D", 1)]),
                "away_win_probability": float(final_probs[i][prob_map.get("A", 2)]),
                "confidence": float(probs_sorted[0] - probs_sorted[1]),
                "max_probability": float(probs_sorted[0]),
                # Diagnostics
                "blend_weight": float(diag.get("weights", [np.nan] * len(df))[i])
                if isinstance(diag.get("weights"), np.ndarray)
                else float("nan"),
                "cls_p_H": float(
                    diag.get("classifier_probs", np.zeros_like(final_probs))[i][
                        prob_map.get("H", 0)
                    ]
                ),
                "cls_p_D": float(
                    diag.get("classifier_probs", np.zeros_like(final_probs))[i][
                        prob_map.get("D", 1)
                    ]
                ),
                "cls_p_A": float(
                    diag.get("classifier_probs", np.zeros_like(final_probs))[i][
                        prob_map.get("A", 2)
                    ]
                ),
                "pois_p_H": float(
                    diag.get("poisson_probs", np.zeros_like(final_probs))[i][
                        prob_map.get("H", 0)
                    ]
                ),
                "pois_p_D": float(
                    diag.get("poisson_probs", np.zeros_like(final_probs))[i][
                        prob_map.get("D", 1)
                    ]
                ),
                "pois_p_A": float(
                    diag.get("poisson_probs", np.zeros_like(final_probs))[i][
                        prob_map.get("A", 2)
                    ]
                ),
            }
            predictions.append(pred)
        return predictions

    # --- Persistence ---
    def save_models(self):
        """Saves the trained models and metadata to disk."""
        if not all([self.outcome_model, self.home_goals_model, self.away_goals_model]):
            raise ValueError("No models to save. Must train first.")

        self._log(f"Saving models to {self.config.paths.models_dir}")
        joblib.dump(self.outcome_model, self.config.paths.outcome_model_path)
        joblib.dump(self.home_goals_model, self.config.paths.home_goals_model_path)
        joblib.dump(self.away_goals_model, self.config.paths.away_goals_model_path)

        metadata = {
            "feature_columns": self.feature_columns,
            "label_encoder": self.label_encoder,

        }
        joblib.dump(metadata, self.config.paths.models_dir / "metadata.joblib")
        self._log("Models saved successfully.")

    def load_models(self):
        """Loads trained models and metadata from disk."""
        self._log(f"Loading models from {self.config.paths.models_dir}")
        if not self.config.paths.outcome_model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.config.paths.outcome_model_path}"
            )

        self.outcome_model = joblib.load(self.config.paths.outcome_model_path)
        self.home_goals_model = joblib.load(self.config.paths.home_goals_model_path)
        self.away_goals_model = joblib.load(self.config.paths.away_goals_model_path)

        metadata = joblib.load(self.config.paths.models_dir / "metadata.joblib")
        self.feature_columns = metadata["feature_columns"]
        self.label_encoder = metadata["label_encoder"]

        self._log("Models loaded successfully.")
