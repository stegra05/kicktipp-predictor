"""Match predictor using Predictor-Selector architecture.

This module implements a two-step prediction system:
1. Outcome Prediction (Selector): XGBClassifier determines match result (H/D/A)
2. Scoreline Selection (Predictor): XGBRegressors + Poisson find the most likely score
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from scipy.stats import poisson
from typing import Dict, List, Tuple
import joblib
import time

from .config import get_config


class MatchPredictor:
    """Two-step predictor: outcome classifier + goal regressors + Poisson scoreline selection."""

    def __init__(self, quiet: bool = False):
        """Initialize with configuration."""
        self.config = get_config()
        self.quiet = quiet

        # Models
        self.outcome_model: XGBClassifier | None = None  # Predicts H/D/A
        self.home_goals_model: XGBRegressor | None = None  # Predicts home goals
        self.away_goals_model: XGBRegressor | None = None  # Predicts away goals

        # Feature metadata
        self.feature_columns: List[str] = []
        self.label_encoder = LabelEncoder()

        self._log(f"[MatchPredictor] Initialized with config: {self.config}")

    def _log(self, *args, **kwargs) -> None:
        if not getattr(self, 'quiet', False):
            print(*args, **kwargs)

    def train(self, matches_df: pd.DataFrame):
        """Train both outcome classifier and goal regressors.

        Args:
            matches_df: DataFrame with features and target variables.
        """
        # Filter only finished matches with scores
        training_data = matches_df[matches_df['home_score'].notna()].copy()

        if len(training_data) < self.config.model.min_training_matches:
            self._log(f"Not enough training data. Need at least {self.config.model.min_training_matches} matches.")
            return

        # Identify feature columns
        exclude_cols = ['match_id', 'home_team', 'away_team', 'home_score',
                       'away_score', 'goal_difference', 'result']
        self.feature_columns = [col for col in training_data.columns
                               if col not in exclude_cols]

        X = training_data[self.feature_columns].fillna(0)
        y_home = training_data['home_score']
        y_away = training_data['away_score']
        y_result = training_data['result']

        # Encode result labels
        y_result_encoded = self.label_encoder.fit_transform(y_result)

        # Log training distribution
        counts = y_result.value_counts()
        total = len(y_result)
        self._log(f"Training on {len(training_data)} matches with {len(self.feature_columns)} features")
        self._log("Outcome distribution:", {k: f"{int(v)} ({v/total:.1%})" for k, v in counts.items()})

        # Train goal regressors
        self._train_goal_models(X, y_home, y_away)

        # Train outcome classifier
        self._train_outcome_model(X, y_result_encoded)

        self._log("Training completed!")

    def _train_goal_models(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        """Train home and away goal regressors."""
        self._log("Training goal regressors...")
        start = time.perf_counter()

        self.home_goals_model = XGBRegressor(
            n_estimators=self.config.model.goals_n_estimators,
            max_depth=self.config.model.goals_max_depth,
            learning_rate=self.config.model.goals_learning_rate,
            subsample=self.config.model.goals_subsample,
            colsample_bytree=self.config.model.goals_colsample_bytree,
            objective='count:poisson',
            tree_method='hist',
            random_state=self.config.model.random_state,
            n_jobs=self.config.model.n_jobs,
        )

        self.away_goals_model = XGBRegressor(
            n_estimators=self.config.model.goals_n_estimators,
            max_depth=self.config.model.goals_max_depth,
            learning_rate=self.config.model.goals_learning_rate,
            subsample=self.config.model.goals_subsample,
            colsample_bytree=self.config.model.goals_colsample_bytree,
            objective='count:poisson',
            tree_method='hist',
            random_state=self.config.model.random_state,
            n_jobs=self.config.model.n_jobs,
        )

        self.home_goals_model.fit(X, y_home)
        self.away_goals_model.fit(X, y_away)

        elapsed = time.perf_counter() - start
        self._log(f"Goal regressors trained in {elapsed:.2f}s")

    def _train_outcome_model(self, X: pd.DataFrame, y_result_encoded: np.ndarray):
        """Train outcome classifier with class weights."""
        self._log("Training outcome classifier...")
        start = time.perf_counter()

        # Split for proper evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_result_encoded,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y_result_encoded
        )

        # Compute class weights with draw boost
        classes_unique = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes_unique,
            y=y_train
        )

        # Apply draw boost
        weight_map = {c: float(w) for c, w in zip(classes_unique, class_weights)}
        try:
            draw_idx = int(np.where(self.label_encoder.classes_ == 'D')[0][0])
            if draw_idx in weight_map:
                weight_map[draw_idx] *= self.config.model.draw_boost
                self._log(f"Applied draw boost: {self.config.model.draw_boost}x")
        except Exception:
            pass

        sample_weights = np.array([weight_map[int(y)] for y in y_train], dtype=float)

        # Train with early stopping
        self.outcome_model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=self.config.model.outcome_n_estimators,
            max_depth=self.config.model.outcome_max_depth,
            learning_rate=self.config.model.outcome_learning_rate,
            subsample=self.config.model.outcome_subsample,
            colsample_bytree=self.config.model.outcome_colsample_bytree,
            random_state=self.config.model.random_state,
            n_jobs=self.config.model.n_jobs,
            early_stopping_rounds=50,
        )

        self.outcome_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        elapsed = time.perf_counter() - start
        self._log(f"Outcome classifier trained in {elapsed:.2f}s")
        self._log(f"Class weights: {dict(zip(self.label_encoder.classes_, class_weights))}")

    def predict(self, features_df: pd.DataFrame) -> List[Dict]:
        """Predict match outcomes and scorelines.

        This implements the two-step Predictor-Selector process:
        1. Outcome Selector: Determine H/D/A from classifier
        2. Scoreline Predictor: Select most probable scoreline matching outcome

        Args:
            features_df: DataFrame with match features.

        Returns:
            List of prediction dictionaries.
        """
        if self.outcome_model is None or self.home_goals_model is None:
            raise ValueError("Models not trained. Call train() first.")

        X = features_df[self.feature_columns].fillna(0)

        # Step 1: Predict outcome (H/D/A) - The Selector
        outcome_probs = self.outcome_model.predict_proba(X)
        outcome_classes = self.outcome_model.predict(X)

        # Step 2: Predict expected goals - The Predictor
        home_lambdas = np.maximum(self.home_goals_model.predict(X), self.config.model.min_lambda)
        away_lambdas = np.maximum(self.away_goals_model.predict(X), self.config.model.min_lambda)

        predictions = []

        for idx in range(len(X)):
            # Get outcome from classifier
            outcome_encoded = outcome_classes[idx]
            outcome = self.label_encoder.inverse_transform([outcome_encoded])[0]

            # Get outcome probabilities (ensure correct mapping)
            prob_dict = {}
            for class_idx, class_label in enumerate(self.label_encoder.classes_):
                prob_dict[class_label] = float(outcome_probs[idx][class_idx])

            home_win_prob = prob_dict.get('H', 0.0)
            draw_prob = prob_dict.get('D', 0.0)
            away_win_prob = prob_dict.get('A', 0.0)

            # Step 3: Select scoreline matching the outcome using Poisson
            home_score, away_score = self._select_scoreline(
                outcome,
                float(home_lambdas[idx]),
                float(away_lambdas[idx])
            )

            # Calculate confidence (margin between top two probabilities)
            probs_sorted = sorted([home_win_prob, draw_prob, away_win_prob], reverse=True)
            confidence = probs_sorted[0] - probs_sorted[1]

            # Build prediction
            pred = {
                'match_id': features_df.iloc[idx]['match_id'],
                'home_team': features_df.iloc[idx]['home_team'],
                'away_team': features_df.iloc[idx]['away_team'],
                'predicted_home_score': int(home_score),
                'predicted_away_score': int(away_score),
                'home_expected_goals': float(home_lambdas[idx]),
                'away_expected_goals': float(away_lambdas[idx]),
                'predicted_result': outcome,
                'home_win_probability': home_win_prob,
                'draw_probability': draw_prob,
                'away_win_probability': away_win_prob,
                'confidence': float(confidence),
                'max_probability': float(probs_sorted[0]),
            }

            predictions.append(pred)

        return predictions

    def _select_scoreline(self, outcome: str, home_lambda: float, away_lambda: float) -> Tuple[int, int]:
        """Select the most probable scoreline matching the predicted outcome.

        Args:
            outcome: Predicted outcome ('H', 'D', or 'A').
            home_lambda: Expected home goals.
            away_lambda: Expected away goals.

        Returns:
            Tuple of (home_score, away_score).
        """
        max_goals = self.config.model.max_goals

        # Create Poisson probability grid
        grid = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                grid[h, a] = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)

        # Filter grid by outcome
        if outcome == 'H':
            # Home win: keep only h > a
            for h in range(max_goals + 1):
                for a in range(h + 1, max_goals + 1):  # a >= h
                    grid[h, a] = 0
        elif outcome == 'A':
            # Away win: keep only a > h
            for h in range(max_goals + 1):
                for a in range(h + 1):  # a <= h
                    grid[h, a] = 0
        else:  # outcome == 'D'
            # Draw: keep only h == a
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    if h != a:
                        grid[h, a] = 0

        # Find maximum probability scoreline
        max_prob = np.max(grid)
        if max_prob == 0:
            # Fallback if no valid scoreline found (shouldn't happen)
            if outcome == 'H':
                return (2, 1)
            elif outcome == 'A':
                return (1, 2)
            else:
                return (1, 1)

        # Get scoreline with max probability (with tie-breaking toward realistic scores)
        candidates = np.argwhere(grid == max_prob)

        if len(candidates) == 1:
            return int(candidates[0][0]), int(candidates[0][1])

        # Multiple candidates: prefer common realistic scorelines
        common_scorelines = [(2, 1), (1, 0), (1, 1), (0, 1), (2, 0), (0, 0),
                            (2, 2), (3, 1), (1, 2), (3, 0), (0, 3), (3, 2), (2, 3)]

        for h, a in common_scorelines:
            if h <= max_goals and a <= max_goals and grid[h, a] == max_prob:
                return h, a

        # Fallback to first candidate
        return int(candidates[0][0]), int(candidates[0][1])

    def save_models(self):
        """Save trained models to disk."""
        if self.outcome_model is None:
            raise ValueError("No models to save. Train first.")

        self._log("Saving models...")

        joblib.dump(self.outcome_model, self.config.paths.outcome_model_path)
        joblib.dump(self.home_goals_model, self.config.paths.home_goals_model_path)
        joblib.dump(self.away_goals_model, self.config.paths.away_goals_model_path)

        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'label_encoder': self.label_encoder,
        }
        metadata_path = self.config.paths.models_dir / 'metadata.joblib'
        joblib.dump(metadata, metadata_path)

        self._log(f"Models saved to {self.config.paths.models_dir}")

    def load_models(self):
        """Load trained models from disk."""
        self._log("Loading models...")

        if not self.config.paths.outcome_model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.paths.outcome_model_path}")

        self.outcome_model = joblib.load(self.config.paths.outcome_model_path)
        self.home_goals_model = joblib.load(self.config.paths.home_goals_model_path)
        self.away_goals_model = joblib.load(self.config.paths.away_goals_model_path)

        # Load metadata
        metadata_path = self.config.paths.models_dir / 'metadata.joblib'
        metadata = joblib.load(metadata_path)
        self.feature_columns = metadata['feature_columns']
        self.label_encoder = metadata['label_encoder']

        self._log(f"Models loaded from {self.config.paths.models_dir}")

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate predictor on test data.

        Args:
            test_df: DataFrame with features and actual results.

        Returns:
            Dictionary with evaluation metrics.
        """
        from .evaluate import evaluate_predictor
        return evaluate_predictor(self, test_df)
