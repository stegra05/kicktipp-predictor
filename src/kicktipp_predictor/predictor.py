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

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from .config import get_config


def compute_scoreline_for_outcome(
    outcome: str, home_lambda: float, away_lambda: float, max_goals: int
) -> tuple[int, int]:
    """Selects the most probable scoreline for a given outcome using a Poisson grid.

    This function is designed for parallel execution. It constructs a probability
    grid for all possible scorelines up to `max_goals` and filters it to find
    the scoreline with the highest probability that matches the specified outcome.

    Args:
        outcome: The predicted outcome ('H', 'D', 'A').
        home_lambda: The expected number of goals for the home team.
        away_lambda: The expected number of goals for the away team.
        max_goals: The maximum number of goals to consider for the grid.

    Returns:
        A tuple containing the most probable (home_score, away_score).
    """
    max_goals = int(max(0, max_goals))

    grid = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            grid[h, a] = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)

    if outcome == "H":
        grid = np.tril(grid, k=-1)
    elif outcome == "A":
        grid = np.triu(grid, k=1)
    else:
        grid = np.diag(np.diag(grid))

    if np.max(grid) == 0:
        return (2, 1) if outcome == "H" else ((1, 2) if outcome == "A" else (1, 1))

    candidates = np.argwhere(grid == np.max(grid))
    if len(candidates) == 1:
        return int(candidates[0][0]), int(candidates[0][1])

    common_scorelines = [
        (2, 1),
        (1, 0),
        (1, 1),
        (0, 1),
        (2, 0),
        (0, 0),
        (2, 2),
        (3, 1),
        (1, 2),
        (3, 0),
        (0, 3),
        (3, 2),
        (2, 3),
    ]
    for h, a in common_scorelines:
        if h <= max_goals and a <= max_goals and grid[h, a] == np.max(grid):
            return h, a

    return int(candidates[0][0]), int(candidates[0][1])


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

    def __init__(self, quiet: bool = False):
        """Initializes the MatchPredictor with its configuration."""
        self.config = get_config()
        self.quiet = quiet
        self.outcome_model: XGBClassifier | None = None
        self.home_goals_model: XGBRegressor | None = None
        self.away_goals_model: XGBRegressor | None = None
        self.feature_columns: list[str] = []
        self.label_encoder = LabelEncoder()
        self.calibrator: LogisticRegression | None = None
        self.train_class_prior: np.ndarray | None = None
        self._log(f"[MatchPredictor] Initialized with config: {self.config}")

    def _log(self, *args, **kwargs) -> None:
        if not self.quiet:
            print(*args, **kwargs)

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
        self.feature_columns = (
            training_data.drop(columns=exclude_cols, errors="ignore")
            .select_dtypes(include=[np.number, bool])
            .columns.tolist()
        )

        X = training_data[self.feature_columns].fillna(0)
        y_home = training_data["home_score"]
        y_away = training_data["away_score"]
        y_result = training_data["result"]
        y_result_encoded = self.label_encoder.fit_transform(y_result)

        time_weights = self._compute_time_decay_weights(training_data)

        counts = y_result.value_counts()
        self._log(
            f"Training on {len(training_data)} matches with {len(self.feature_columns)} features."
        )
        self._log(
            "Outcome distribution:",
            {k: f"{v} ({v / len(y_result):.1%})" for k, v in counts.items()},
        )

        dates = (
            pd.to_datetime(training_data["date"], errors="coerce")
            if "date" in training_data.columns
            else None
        )
        self._train_goal_models(
            X, y_home, y_away, sample_weights=time_weights, dates=dates
        )
        val_ctx = self._train_outcome_model(X, y_result_encoded, time_weights)

        # Store class prior from full training labels (for optional prior anchoring)
        prior = np.zeros(3, dtype=float)
        for i, lab in enumerate(self.label_encoder.classes_):
            prior[i] = float(np.mean(y_result.values == lab))
        self.train_class_prior = prior / max(1e-15, prior.sum())

        # Optional calibrator on blended validation probabilities (leakage-safe holdout)
        try:
            if self.config.model.calibrator_enabled and val_ctx is not None:
                X_val = val_ctx.get("X_val")
                y_val_enc = val_ctx.get("y_val_enc")
                cls_val_probs = val_ctx.get("cls_val_probs")
                if (
                    X_val is not None
                    and y_val_enc is not None
                    and cls_val_probs is not None
                    and self.home_goals_model is not None
                    and self.away_goals_model is not None
                ):
                    # Get Poisson probabilities for the same validation rows
                    hlam = np.maximum(
                        self.home_goals_model.predict(X_val),
                        self.config.model.min_lambda,
                    )
                    alam = np.maximum(
                        self.away_goals_model.predict(X_val),
                        self.config.model.min_lambda,
                    )
                    pois_val = self._calculate_poisson_outcome_probs(hlam, alam)
                    blend_val = self._blend_probs(cls_val_probs, pois_val)
                    self._fit_calibrator(blend_val, y_val_enc)
        except Exception as e:  # pragma: no cover - optional
            self._log(f"[Calibrator] Skipped due to error: {e}")

        self._log("Training completed.")

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
        self._log("Training goal regressors...")
        start = time.perf_counter()

        self.home_goals_model = XGBRegressor(**self.config.model.goals_params)
        self.away_goals_model = XGBRegressor(**self.config.model.goals_params)

        train_mask = self._get_train_validation_split(X, dates)
        X_tr, X_val = X[train_mask], X[~train_mask]
        yh_tr, yh_val = y_home[train_mask], y_home[~train_mask]
        ya_tr, ya_val = y_away[train_mask], y_away[~train_mask]
        sw_tr = sample_weights[train_mask] if sample_weights is not None else None

        fit_params = {"eval_set": [(X_val, yh_val)], "verbose": False}
        if sw_tr is not None:
            fit_params["sample_weight"] = sw_tr
        self.home_goals_model.fit(X_tr, yh_tr, **fit_params)

        fit_params["eval_set"] = [(X_val, ya_val)]
        self.away_goals_model.fit(X_tr, ya_tr, **fit_params)

        self._log(f"Goal regressors trained in {time.perf_counter() - start:.2f}s")

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
        self._log("Training outcome classifier...")
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

        self.outcome_model = XGBClassifier(**self.config.model.outcome_params)
        self.outcome_model.fit(
            X_train, y_train, sample_weight=sample_weights, verbose=False
        )

        self._log(f"Outcome classifier trained in {time.perf_counter() - start:.2f}s")
        self._log(
            f"Applied balanced class weights, draw_boost={draw_boost}, and time-decay weighting."
        )
        # Return validation context for calibrator fitting
        try:
            cls_val_probs = self.outcome_model.predict_proba(X_val)
        except Exception:
            cls_val_probs = None
        return {"X_val": X_val, "y_val_enc": y_val, "cls_val_probs": cls_val_probs}

    def predict(
        self, features_df: pd.DataFrame, workers: int | None = None
    ) -> list[dict]:
        """Predicts match outcomes and scorelines.

        Args:
            features_df: A DataFrame with match features.
            workers: The number of parallel workers for scoreline selection.

        Returns:
            A list of prediction dictionaries.
        """
        if not all([self.outcome_model, self.home_goals_model, self.away_goals_model]):
            raise ValueError("Models must be trained or loaded before prediction.")

        X = self._prepare_features(features_df)
        classifier_probs = self._get_calibrated_probabilities(X)
        home_lambdas = np.maximum(
            self.home_goals_model.predict(X), self.config.model.min_lambda
        )
        away_lambdas = np.maximum(
            self.away_goals_model.predict(X), self.config.model.min_lambda
        )

        # Final blended probabilities (with optional calibration & anchoring)
        final_probs, diag = self._derive_final_probabilities(
            classifier_probs, home_lambdas, away_lambdas
        )

        # Choose scoreline via expected KickTipp points
        scorelines, ep_values = self._compute_ep_scorelines(home_lambdas, away_lambdas)

        outcomes = self.label_encoder.inverse_transform(np.argmax(final_probs, axis=1))

        return self._format_predictions(
            features_df,
            outcomes,
            scorelines,
            final_probs,
            home_lambdas,
            away_lambdas,
            diag,
            ep_values,
        )

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns DataFrame columns with the model's feature schema."""
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_columns].fillna(0)

    def _get_calibrated_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Applies temperature scaling and prior blending to classifier probabilities."""
        probs = self.outcome_model.predict_proba(X)
        temp = float(self.config.model.proba_temperature)
        if temp != 1.0:
            with np.errstate(over="ignore"):
                logits = np.log(np.clip(probs, 1e-15, 1.0)) / max(1e-6, temp)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)

        alpha = float(self.config.model.prior_blend_alpha)
        if alpha > 0.0 and self.config.model.prob_source == "classifier":
            prior = np.full(3, 1.0 / 3.0)
            probs = (1.0 - alpha) * probs + alpha * prior
            probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _compute_scorelines(self, outcomes, home_lambdas, away_lambdas, workers):
        """Computes the most likely scoreline for each match in parallel."""
        max_goals = self.config.model.max_goals
        n = len(outcomes)
        if workers and workers > 1 and n > 0:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                return list(
                    executor.map(
                        compute_scoreline_for_outcome,
                        outcomes,
                        home_lambdas,
                        away_lambdas,
                        itertools.repeat(max_goals, n),
                    )
                )
        return [
            compute_scoreline_for_outcome(
                outcomes[i], home_lambdas[i], away_lambdas[i], max_goals
            )
            for i in range(n)
        ]

    def _derive_final_probabilities(self, classifier_probs, home_lambdas, away_lambdas):
        """Blend classifier and Poisson probabilities, then calibrate and anchor."""
        prob_source = self.config.model.prob_source
        pois_probs = self._calculate_poisson_outcome_probs(home_lambdas, away_lambdas)

        if prob_source == "classifier":
            blend = classifier_probs
            weights = np.zeros(len(classifier_probs))
            entropy = -np.sum(
                np.clip(classifier_probs, 1e-15, 1.0)
                * np.log(np.clip(classifier_probs, 1e-15, 1.0)),
                axis=1,
            ) / np.log(3.0)
        elif prob_source == "poisson":
            blend = pois_probs
            weights = np.ones(len(pois_probs))
            entropy = -np.sum(
                np.clip(classifier_probs, 1e-15, 1.0)
                * np.log(np.clip(classifier_probs, 1e-15, 1.0)),
                axis=1,
            ) / np.log(3.0)
        else:
            # Hybrid
            weights = self._compute_entropy_weights(classifier_probs)
            blend = self._blend_probs(classifier_probs, pois_probs, weights)
            entropy = -np.sum(
                np.clip(classifier_probs, 1e-15, 1.0)
                * np.log(np.clip(classifier_probs, 1e-15, 1.0)),
                axis=1,
            ) / np.log(3.0)

        # Calibrate and prior-anchor
        calibrated = self._apply_calibration_and_anchoring(blend)
        diag = {
            "weights": weights,
            "entropy": entropy,
            "classifier_probs": classifier_probs,
            "poisson_probs": pois_probs,
            "precalibrated": blend,
            "postcalibrated": calibrated,
        }
        return calibrated, diag

    def _compute_entropy_weights(self, classifier_probs: np.ndarray) -> np.ndarray:
        scheme = getattr(self.config.model, "hybrid_scheme", "entropy").lower()
        if scheme != "entropy":
            return np.full(
                len(classifier_probs), float(self.config.model.hybrid_poisson_weight)
            )
        h = -np.sum(
            np.clip(classifier_probs, 1e-15, 1.0)
            * np.log(np.clip(classifier_probs, 1e-15, 1.0)),
            axis=1,
        )
        h_norm = h / np.log(3.0)
        w_min = float(self.config.model.hybrid_entropy_w_min)
        w_max = float(self.config.model.hybrid_entropy_w_max)
        w = w_min + (w_max - w_min) * np.clip(h_norm, 0.0, 1.0)
        return np.clip(w, 0.0, 1.0)

    def _blend_probs(
        self,
        cls: np.ndarray,
        pois: np.ndarray,
        weights: np.ndarray | float | None = None,
    ) -> np.ndarray:
        if weights is None or isinstance(weights, float):
            w = (
                float(weights)
                if weights is not None
                else float(self.config.model.hybrid_poisson_weight)
            )
            blend = (1.0 - w) * cls + w * pois
        else:
            w = weights.reshape(-1, 1)
            blend = (1.0 - w) * cls + w * pois
        blend = np.clip(blend, 1e-15, 1.0)
        return blend / blend.sum(axis=1, keepdims=True)

    def _calculate_poisson_outcome_probs(self, home_lambdas, away_lambdas):
        """Outcome probabilities from Poisson lambdas with per-match dynamic grid and optional diagonal bump."""
        n = len(home_lambdas)
        probs = np.zeros((n, 3), dtype=float)
        max_cap = int(self.config.model.proba_grid_max_goals)
        rho = float(self.config.model.poisson_draw_rho)
        for i in range(n):
            lam_h = float(home_lambdas[i])
            lam_a = float(away_lambdas[i])
            G = int(min(max_cap, int(np.ceil(max(lam_h, lam_a) + 4))))
            x = np.arange(G + 1)
            ph = poisson.pmf(x, lam_h)
            pa = poisson.pmf(x, lam_a)
            grid = np.outer(ph, pa)
            # Optional Dixon–Coles low-score dependence
            grid = self._maybe_apply_dixon_coles(grid, lam_h, lam_a)
            # Optional diagonal bump legacy (minor tweak)
            if rho != 0.0:
                idx = np.arange(G + 1)
                grid[idx, idx] *= np.exp(rho)
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

    def _compute_ep_scorelines(
        self, home_lambdas: np.ndarray, away_lambdas: np.ndarray
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Choose scorelines maximizing expected KickTipp points for each match."""
        n = len(home_lambdas)
        picks: list[tuple[int, int]] = []
        ep_values = np.zeros(n, dtype=float)
        max_cap = int(self.config.model.max_goals)
        rho = float(self.config.model.poisson_draw_rho)
        common_order = [
            (2, 1),
            (1, 0),
            (1, 1),
            (0, 1),
            (2, 0),
            (0, 0),
            (2, 2),
            (3, 1),
            (1, 2),
            (3, 0),
            (0, 3),
            (3, 2),
            (2, 3),
        ]
        for i in range(n):
            lam_h = float(home_lambdas[i])
            lam_a = float(away_lambdas[i])
            G = int(min(max_cap, int(np.ceil(max(lam_h, lam_a) + 4))))
            x = np.arange(G + 1)
            ph = poisson.pmf(x, lam_h)
            pa = poisson.pmf(x, lam_a)
            grid = np.outer(ph, pa)
            # Optional Dixon–Coles
            grid = self._maybe_apply_dixon_coles(grid, lam_h, lam_a)
            if rho != 0.0:
                idx = np.arange(G + 1)
                grid[idx, idx] *= np.exp(rho)
            total = np.sum(grid)
            if total <= 0:
                picks.append((2, 1))
                ep_values[i] = 0.0
                continue
            grid /= total
            # Precompute sums for same outcome categories and same goal-difference
            pH = float(np.sum(np.triu(grid, k=1)))
            pD = float(np.sum(np.diag(grid)))
            pA = float(np.sum(np.tril(grid, k=-1)))
            # Precompute sum of each diagonal (goal diff d = h - a)
            sum_by_diff = {}
            for d in range(-G, G + 1):
                sum_by_diff[d] = float(np.sum(np.diagonal(grid, offset=d)))

            best_ep = -1.0
            best = (2, 1)
            # Evaluate expected points efficiently using identity: EP = exact + sum_diff + 2 * p_outcome
            for h in range(G + 1):
                for a in range(G + 1):
                    e = float(grid[h, a])
                    d = h - a
                    s = sum_by_diff.get(d, 0.0)
                    outcome_p = pD if d == 0 else (pH if d > 0 else pA)
                    ep = e + s + 2.0 * outcome_p
                    if ep > best_ep + 1e-12:
                        best_ep = ep
                        best = (h, a)
                    elif abs(ep - best_ep) <= 1e-12:
                        # Tie-break with common scoreline preference
                        if best not in common_order and (h, a) in common_order:
                            best = (h, a)
                        elif best in common_order and (h, a) in common_order:
                            if common_order.index((h, a)) < common_order.index(best):
                                best = (h, a)
            picks.append(best)
            ep_values[i] = best_ep
        return picks, ep_values

    def _maybe_apply_dixon_coles(
        self, grid: np.ndarray, lam_h: float, lam_a: float
    ) -> np.ndarray:
        """Apply Dixon–Coles low-score adjustment if enabled in config.

        Adjusts probabilities for (0,0), (0,1), (1,0), (1,1) using a small rho.
        """
        try:
            if (
                getattr(self.config.model, "poisson_joint", "independent").lower()
                != "dixon_coles"
            ):
                return grid
            rho = float(getattr(self.config.model, "dixon_coles_rho", 0.0))
            if abs(rho) < 1e-12:
                return grid
            G = grid.shape[0] - 1
            P = np.array(grid, copy=True)
            # τ adjustments (common approximation)
            # clip indexes
            P[0, 0] *= max(0.0, 1.0 - lam_h * lam_a * rho)
            if G >= 1:
                P[0, 1] *= max(0.0, 1.0 + lam_h * rho)
                P[1, 0] *= max(0.0, 1.0 + lam_a * rho)
                P[1, 1] *= max(0.0, 1.0 - rho)
            return P
        except Exception:
            return grid

    def _apply_calibration_and_anchoring(self, probs: np.ndarray) -> np.ndarray:
        calibrated = np.array(probs, copy=True)
        if self.config.model.calibrator_enabled and self.calibrator is not None:
            try:
                X_cal = np.log(np.clip(calibrated, 1e-15, 1.0))
                calibrated = self.calibrator.predict_proba(X_cal)
            except Exception:
                pass
        if (
            self.config.model.prior_anchor_enabled
            and self.train_class_prior is not None
        ):
            s = float(self.config.model.prior_anchor_strength)
            prior = np.clip(self.train_class_prior, 1e-8, 1.0)
            anchored = (np.clip(calibrated, 1e-15, 1.0) ** (1.0 - s)) * (
                prior.reshape(1, -1) ** s
            )
            anchored = np.clip(anchored, 1e-15, 1.0)
            anchored /= anchored.sum(axis=1, keepdims=True)
            return anchored
        return calibrated

    def _fit_calibrator(self, probs: np.ndarray, y_enc: np.ndarray) -> None:
        method = getattr(self.config.model, "calibrator_method", "multinomial_logistic")
        if method == "multinomial_logistic":
            X_cal = np.log(np.clip(probs, 1e-15, 1.0))
            C = float(getattr(self.config.model, "calibrator_C", 1.0))
            lr = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", C=C, max_iter=1000
            )
            lr.fit(X_cal, y_enc)
            self.calibrator = lr
        else:
            # Placeholder: keep None if unsupported method
            self.calibrator = None

    def _format_predictions(
        self,
        df,
        outcomes,
        scorelines,
        final_probs,
        home_lambdas,
        away_lambdas,
        diag: dict,
        ep_values: np.ndarray,
    ):
        """Assembles the final list of prediction dictionaries."""
        predictions = []
        prob_map = {label: i for i, label in enumerate(self.label_encoder.classes_)}

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
                "entropy": float(diag.get("entropy", [np.nan] * len(df))[i])
                if isinstance(diag.get("entropy"), np.ndarray)
                else float("nan"),
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
                "expected_points_pick": float(ep_values[i]),
            }
            predictions.append(pred)
        return predictions

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
            "train_class_prior": self.train_class_prior,
        }
        joblib.dump(metadata, self.config.paths.models_dir / "metadata.joblib")
        # Save calibrator if available
        try:
            if self.calibrator is not None:
                joblib.dump(
                    self.calibrator, self.config.paths.models_dir / "calibrator.joblib"
                )
        except Exception:
            pass
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
        self.train_class_prior = metadata.get("train_class_prior")
        # Load calibrator if present
        try:
            cal_path = self.config.paths.models_dir / "calibrator.joblib"
            if cal_path.exists():
                self.calibrator = joblib.load(cal_path)
        except Exception:
            self.calibrator = None
        self._log("Models loaded successfully.")

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluates the predictor on a test dataset.

        Args:
            test_df: A DataFrame with features and actual results.

        Returns:
            A dictionary of evaluation metrics.
        """
        from .evaluate import evaluate_predictor

        return evaluate_predictor(self, test_df)
