import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Tuple
import joblib
import os


class MLPredictor:
    """
    Machine learning predictor using gradient boosting and random forests.
    Predicts both scores and match outcomes.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Models for different predictions
        self.score_model_home = None  # Predicts home goals
        self.score_model_away = None  # Predicts away goals
        self.result_model = None      # Predicts match result (H/D/A)
        self.result_calibrator: CalibratedClassifierCV | None = None  # Calibrated wrapper for probabilities

        # Feature columns (excluding identifiers and targets)
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        # Calibrators to map raw goal predictions to calibrated expected goals (lambdas)
        self.home_goal_calibrator: IsotonicRegression | None = None
        self.away_goal_calibrator: IsotonicRegression | None = None
        # Resolve threads for XGBoost from OMP_NUM_THREADS (fallback to CPU count)
        self.num_threads = self._resolve_num_threads()

    def _resolve_num_threads(self) -> int:
        env_value = os.getenv("OMP_NUM_THREADS")
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                pass
        return max(1, os.cpu_count() or 1)

    def train(self, matches_df: pd.DataFrame):
        """
        Train ML models on historical match data.

        Args:
            matches_df: DataFrame with features and target variables
        """
        # Filter only finished matches with scores
        training_data = matches_df[matches_df['home_score'].notna()].copy()

        if len(training_data) < 50:
            print("Not enough training data. Need at least 50 matches.")
            return

        # Identify feature columns (exclude identifiers and targets)
        exclude_cols = ['match_id', 'home_team', 'away_team', 'home_score',
                       'away_score', 'goal_difference', 'result']
        self.feature_columns = [col for col in training_data.columns
                               if col not in exclude_cols]

        X = training_data[self.feature_columns]
        y_home = training_data['home_score']
        y_away = training_data['away_score']
        y_result = training_data['result']

        # Handle any missing values
        X = X.fillna(0)

        # Encode result labels
        y_result_encoded = self.label_encoder.fit_transform(y_result)
        # Log training label distribution
        counts = y_result.value_counts()
        total = len(y_result)
        print("Training label distribution:", {k: f"{int(v)} ({v/total:.1%})" for k, v in counts.items()})

        print(f"Training on {len(training_data)} matches with {len(self.feature_columns)} features")

        # Train home goals model
        print("Training home goals model...")
        self.score_model_home = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='count:poisson',
            random_state=42,
            n_jobs=self.num_threads,
            nthread=self.num_threads
        )
        self.score_model_home.fit(X, y_home)

        # Train away goals model
        print("Training away goals model...")
        self.score_model_away = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='count:poisson',
            random_state=42,
            n_jobs=self.num_threads,
            nthread=self.num_threads
        )
        self.score_model_away.fit(X, y_away)

        # Fit isotonic calibrators to better align raw regressions with observed goals
        print("Fitting expected-goals calibrators...")
        home_raw_in_sample = self.score_model_home.predict(X)
        away_raw_in_sample = self.score_model_away.predict(X)

        # Ensure non-negative inputs to calibrators
        home_raw_in_sample = np.maximum(home_raw_in_sample, 0)
        away_raw_in_sample = np.maximum(away_raw_in_sample, 0)

        self.home_goal_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.away_goal_calibrator = IsotonicRegression(out_of_bounds='clip')
        # Map raw -> actual goals
        self.home_goal_calibrator.fit(home_raw_in_sample, y_home)
        self.away_goal_calibrator.fit(away_raw_in_sample, y_away)

        # Train result classifier with class weights to address outcome bias
        print("Training result classifier...")
        from sklearn.utils.class_weight import compute_class_weight

        # Split for calibration to avoid bias
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y_result_encoded, test_size=0.2, random_state=42, stratify=y_result_encoded
        )

        # Calculate class weights on training split
        class_weights_array = compute_class_weight(
            'balanced', classes=np.unique(y_tr), y=y_tr
        )
        # Map to training indices
        cw_map = {c: w for c, w in zip(np.unique(y_tr), class_weights_array)}
        sample_weights_tr = np.array([cw_map[y] for y in y_tr])

        self.result_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=self.num_threads,
            nthread=self.num_threads
        )
        self.result_model.fit(X_tr, y_tr, sample_weight=sample_weights_tr)

        # Probability calibration (isotonic) on held-out split
        try:
            self.result_calibrator = CalibratedClassifierCV(self.result_model, method='isotonic', cv='prefit')
            self.result_calibrator.fit(X_cal, y_cal)
        except Exception:
            # Fallback: no calibration if isotonic fails
            self.result_calibrator = None

        print(f"  Class weights applied: {dict(zip(self.label_encoder.classes_, compute_class_weight('balanced', classes=np.unique(y_result_encoded), y=y_result_encoded)))}")

        print("Training completed!")

    def predict(self, features_df: pd.DataFrame) -> List[Dict]:
        """
        Predict outcomes for matches.

        Args:
            features_df: DataFrame with match features

        Returns:
            List of prediction dictionaries
        """
        if self.score_model_home is None or self.score_model_away is None:
            raise ValueError("Models not trained. Call train() first.")

        # Align incoming features to training feature set, filling any missing with zeros
        # and ignoring any extra columns safely.
        missing_cols = [c for c in self.feature_columns if c not in features_df.columns]
        if missing_cols:
            # Create missing columns with default 0 to maintain model input shape
            for col in missing_cols:
                features_df[col] = 0
        X = features_df[self.feature_columns].fillna(0)

        # Predict raw expected goals from ML regressors
        home_scores_raw = self.score_model_home.predict(X)
        away_scores_raw = self.score_model_away.predict(X)

        # Calibrate to expected goals (lambdas)
        if self.home_goal_calibrator is not None:
            home_expected = self.home_goal_calibrator.predict(np.maximum(home_scores_raw, 0))
        else:
            home_expected = home_scores_raw

        if self.away_goal_calibrator is not None:
            away_expected = self.away_goal_calibrator.predict(np.maximum(away_scores_raw, 0))
        else:
            away_expected = away_scores_raw

        # Ensure non-negative lambdas
        home_expected = np.maximum(home_expected, 0)
        away_expected = np.maximum(away_expected, 0)

        # Integer score suggestion for backward-compat; do NOT use to decide outcome downstream
        home_scores = np.round(home_expected).astype(int)
        away_scores = np.round(away_expected).astype(int)

        # Ensure non-negative
        home_scores = np.maximum(home_scores, 0)
        away_scores = np.maximum(away_scores, 0)

        # Predict result probabilities (use calibrated model if available)
        if self.result_calibrator is not None:
            result_probs = self.result_calibrator.predict_proba(X)
        else:
            result_probs = self.result_model.predict_proba(X)

        predictions = []
        for i in range(len(features_df)):
            pred = {
                'match_id': features_df.iloc[i]['match_id'],
                'home_team': features_df.iloc[i]['home_team'],
                'away_team': features_df.iloc[i]['away_team'],
                'predicted_home_score': int(home_scores[i]),
                'predicted_away_score': int(away_scores[i]),
                'home_expected_goals': float(home_expected[i]),
                'away_expected_goals': float(away_expected[i]),
            }

            # Add result probabilities
            # Label encoder order: typically ['A', 'D', 'H']
            prob_dict = {}
            for idx, label in enumerate(self.label_encoder.classes_):
                prob_dict[f'{label}_probability'] = float(result_probs[i][idx])

            pred.update(prob_dict)
            pred['home_win_probability'] = prob_dict.get('H_probability', 0.33)
            pred['draw_probability'] = prob_dict.get('D_probability', 0.33)
            pred['away_win_probability'] = prob_dict.get('A_probability', 0.33)

            predictions.append(pred)

        return predictions

    def save_models(self, prefix: str = "model"):
        """Save trained models to disk."""
        if self.score_model_home is None:
            print("No models to save.")
            return

        joblib.dump(self.score_model_home, os.path.join(self.model_dir, f"{prefix}_home.pkl"))
        joblib.dump(self.score_model_away, os.path.join(self.model_dir, f"{prefix}_away.pkl"))
        joblib.dump(self.result_model, os.path.join(self.model_dir, f"{prefix}_result.pkl"))
        if self.result_calibrator is not None:
            joblib.dump(self.result_calibrator, os.path.join(self.model_dir, f"{prefix}_result_calibrator.pkl"))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, f"{prefix}_encoder.pkl"))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, f"{prefix}_features.pkl"))
        # Calibrators are optional
        if self.home_goal_calibrator is not None:
            joblib.dump(self.home_goal_calibrator, os.path.join(self.model_dir, f"{prefix}_home_calibrator.pkl"))
        if self.away_goal_calibrator is not None:
            joblib.dump(self.away_goal_calibrator, os.path.join(self.model_dir, f"{prefix}_away_calibrator.pkl"))

        print(f"Models saved to {self.model_dir}")

    def load_models(self, prefix: str = "model"):
        """Load trained models from disk."""
        try:
            self.score_model_home = joblib.load(os.path.join(self.model_dir, f"{prefix}_home.pkl"))
            self.score_model_away = joblib.load(os.path.join(self.model_dir, f"{prefix}_away.pkl"))
            self.result_model = joblib.load(os.path.join(self.model_dir, f"{prefix}_result.pkl"))
            # Load optional probability calibrator
            calib_path = os.path.join(self.model_dir, f"{prefix}_result_calibrator.pkl")
            if os.path.exists(calib_path):
                self.result_calibrator = joblib.load(calib_path)
            self.label_encoder = joblib.load(os.path.join(self.model_dir, f"{prefix}_encoder.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, f"{prefix}_features.pkl"))

            # Load optional calibrators if present
            home_cal_path = os.path.join(self.model_dir, f"{prefix}_home_calibrator.pkl")
            away_cal_path = os.path.join(self.model_dir, f"{prefix}_away_calibrator.pkl")
            if os.path.exists(home_cal_path):
                self.home_goal_calibrator = joblib.load(home_cal_path)
            if os.path.exists(away_cal_path):
                self.away_goal_calibrator = joblib.load(away_cal_path)

            print("Models loaded successfully!")
            return True

        except FileNotFoundError:
            print("Model files not found. Please train models first.")
            return False

    def evaluate(self, matches_df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on a test set.

        Args:
            matches_df: DataFrame with actual results

        Returns:
            Dictionary with evaluation metrics
        """
        if self.score_model_home is None:
            raise ValueError("Models not trained.")

        predictions = self.predict(matches_df)

        correct_results = 0
        correct_differences = 0
        correct_scores = 0
        total_points = 0

        for i, pred in enumerate(predictions):
            actual_home = matches_df.iloc[i]['home_score']
            actual_away = matches_df.iloc[i]['away_score']

            pred_home = pred['predicted_home_score']
            pred_away = pred['predicted_away_score']

            # Exact score
            if pred_home == actual_home and pred_away == actual_away:
                correct_scores += 1
                total_points += 4
                continue

            # Correct goal difference
            if (pred_home - pred_away) == (actual_home - actual_away):
                correct_differences += 1
                total_points += 3
                continue

            # Correct winner
            pred_winner = 'H' if pred_home > pred_away else ('A' if pred_away > pred_home else 'D')
            actual_winner = 'H' if actual_home > actual_away else ('A' if actual_away > actual_home else 'D')

            if pred_winner == actual_winner:
                correct_results += 1
                total_points += 2

        n = len(predictions)
        return {
            'total_matches': n,
            'correct_scores': correct_scores,
            'correct_differences': correct_differences,
            'correct_results': correct_results,
            'total_points': total_points,
            'avg_points_per_match': total_points / n if n > 0 else 0,
            'score_accuracy': correct_scores / n if n > 0 else 0,
            'difference_accuracy': correct_differences / n if n > 0 else 0,
            'result_accuracy': correct_results / n if n > 0 else 0,
        }



