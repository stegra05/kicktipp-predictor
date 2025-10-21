import pandas as pd
import numpy as np
from typing import Dict, List
import os
import json
from .ml_model import MLPredictor
from .poisson_model import PoissonPredictor
from .confidence_selector import extract_display_confidence
from scipy.stats import poisson

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class HybridPredictor:
    """
    Hybrid predictor that combines ML and statistical Poisson models.
    Uses ensemble approach with weighted predictions.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.ml_predictor = MLPredictor(model_dir)
        self.poisson_predictor = PoissonPredictor()

        # Weights for ensemble (can be tuned)
        self.ml_weight = 0.6
        self.poisson_weight = 0.4

        # Probability blending weight (grid vs ML classifier)
        # Slightly increase grid influence to recover realism
        self.prob_blend_alpha = 0.65

        # Minimum lambda to avoid degenerate predictions (slightly higher for realistic low scores)
        self.min_lambda = 0.12

        # Temperature scaling for expected goals (Phase 2)
        # Separate temperatures for home and away lambdas
        self.goal_temperature_home = 1.3
        self.goal_temperature_away = 1.3

        # Confidence threshold for adaptive safe strategy
        self.confidence_threshold: float = 0.4

        # Unified grid size
        self.max_goals = 8

        # Default prediction strategy (used when not overridden)
        self.strategy: str = "safe"

        # Try load best params from config
        try:
            cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
            cfg_yaml = os.path.join(cfg_dir, "best_params.yaml")
            cfg_json = os.path.join(cfg_dir, "best_params.json")
            params = None
            if yaml is not None and os.path.exists(cfg_yaml):
                with open(cfg_yaml, "r", encoding="utf-8") as f:
                    params = yaml.safe_load(f)
            elif os.path.exists(cfg_json):
                with open(cfg_json, "r", encoding="utf-8") as f:
                    params = json.load(f)
            if isinstance(params, dict):
                self.ml_weight = float(params.get("ml_weight", self.ml_weight))
                self.poisson_weight = 1.0 - self.ml_weight
                self.prob_blend_alpha = float(params.get("prob_blend_alpha", self.prob_blend_alpha))
                self.min_lambda = float(params.get("min_lambda", self.min_lambda))
                # Back-compat: allow either separate temps or single shared temp
                if "goal_temperature_home" in params or "goal_temperature_away" in params:
                    self.goal_temperature_home = float(params.get("goal_temperature_home", self.goal_temperature_home))
                    self.goal_temperature_away = float(params.get("goal_temperature_away", self.goal_temperature_away))
                else:
                    shared_temp = float(params.get("goal_temperature", self.goal_temperature_home))
                    self.goal_temperature_home = shared_temp
                    self.goal_temperature_away = shared_temp
                self.confidence_threshold = float(params.get("confidence_threshold", self.confidence_threshold))
                # Optional strategy override from config
                self.strategy = str(params.get("strategy", self.strategy))
        except Exception:
            pass

        print(f"[HybridPredictor] params ml_weight={self.ml_weight:.2f} poisson_weight={self.poisson_weight:.2f} "
              f"alpha={self.prob_blend_alpha:.2f} min_lambda={self.min_lambda:.2f} tempH={self.goal_temperature_home:.2f} tempA={self.goal_temperature_away:.2f} "
              f"conf_thr={self.confidence_threshold:.2f} max_goals={self.max_goals} strategy={self.strategy}")


    def train(self, matches_df: pd.DataFrame):
        """
        Train both ML and Poisson models.

        Args:
            matches_df: DataFrame with historical match data
        """
        print("Training ML models...")
        self.ml_predictor.train(matches_df)

        print("\nTraining Poisson model...")
        self.poisson_predictor.train(matches_df)

        print("\nHybrid training completed!")

    def predict(self, features_df: pd.DataFrame) -> List[Dict]:
        """
        Predict using hybrid ensemble approach.

        Args:
            features_df: DataFrame with match features

        Returns:
            List of prediction dictionaries with combined predictions
        """
        # Get predictions from both models
        ml_preds = self.ml_predictor.predict(features_df)

        poisson_preds = []
        for _, row in features_df.iterrows():
            poisson_pred = self.poisson_predictor.predict_match(
                row['home_team'],
                row['away_team']
            )
            poisson_preds.append(poisson_pred)

        # Combine predictions
        hybrid_predictions = []

        for i in range(len(ml_preds)):
            ml_pred = ml_preds[i]
            poisson_pred = poisson_preds[i]

            # Weighted average for expected goals (lambdas)
            home_lambda = (self.ml_weight * ml_pred['home_expected_goals'] +
                           self.poisson_weight * poisson_pred['home_expected_goals'])
            away_lambda = (self.ml_weight * ml_pred['away_expected_goals'] +
                           self.poisson_weight * poisson_pred['away_expected_goals'])

            # Clamp lambdas to avoid degenerate near-zero predictions
            home_lambda = max(home_lambda, self.min_lambda)
            away_lambda = max(away_lambda, self.min_lambda)

            # Apply temperature scaling to match observed goal distributions (Phase 2)
            home_lambda *= self.goal_temperature_home
            away_lambda *= self.goal_temperature_away

            # Build probability grid with DC correction
            grid = np.zeros((self.max_goals, self.max_goals))
            for hg in range(self.max_goals):
                for ag in range(self.max_goals):
                    p = poisson.pmf(hg, max(home_lambda, 1e-9)) * poisson.pmf(ag, max(away_lambda, 1e-9))
                    # Apply DC correction using rho from PoissonPredictor
                    rho = getattr(self.poisson_predictor, 'rho', 0.0)
                    if hg == 0 and ag == 0:
                        p *= (1.0 + rho)
                    elif (hg == 0 and ag == 1) or (hg == 1 and ag == 0):
                        p *= (1.0 - rho)
                    elif hg == 1 and ag == 1:
                        p *= (1.0 - rho)
                    grid[hg, ag] = p

            total = np.sum(grid)
            if total > 0:
                grid /= total

            # MAP scoreline with small tie-break preference toward common scorelines
            max_p = np.max(grid)
            # Near-ties within epsilon
            epsilon = 1e-6
            candidates = np.argwhere(grid >= max_p - epsilon)
            if len(candidates) == 1:
                predicted_home, predicted_away = int(candidates[0][0]), int(candidates[0][1])
            else:
                # Prefer among common realistic football scorelines by their probability
                common = [(2,1), (1,0), (1,1), (0,1), (2,0), (0,0), (2,2), (3,1), (1,2)]
                chosen = None
                for (ch, ca) in common:
                    if ch < grid.shape[0] and ca < grid.shape[1]:
                        if grid[ch, ca] >= max_p - epsilon:
                            chosen = (ch, ca)
                            break
                if chosen is None:
                    # Fallback to strict argmax
                    flat_idx = np.argmax(grid)
                    chosen = np.unravel_index(flat_idx, grid.shape)
                predicted_home, predicted_away = int(chosen[0]), int(chosen[1])

            # Outcome probabilities from grid
            grid_home_win = float(np.sum(np.tril(grid, -1)))
            grid_draw = float(np.trace(grid))
            grid_away_win = float(np.sum(np.triu(grid, 1)))

            # Get ML classifier probabilities
            ml_home_win = ml_pred.get('home_win_probability', 1/3)
            ml_draw = ml_pred.get('draw_probability', 1/3)
            ml_away_win = ml_pred.get('away_win_probability', 1/3)

            # Blend probabilities: grid (Poisson-based) + ML classifier
            # This combines statistical reasoning with learned patterns
            home_win_prob = self.prob_blend_alpha * grid_home_win + (1 - self.prob_blend_alpha) * ml_home_win
            draw_prob = self.prob_blend_alpha * grid_draw + (1 - self.prob_blend_alpha) * ml_draw
            away_win_prob = self.prob_blend_alpha * grid_away_win + (1 - self.prob_blend_alpha) * ml_away_win

            # Near-tie draw nudge: if top two classes are very close and grid suggests
            # comparable or higher draw, give draw a tiny bump so draws can surface.
            try:
                probs_temp = np.array([home_win_prob, draw_prob, away_win_prob], dtype=float)
                top_two_margin = float(np.sort(probs_temp)[-1] - np.sort(probs_temp)[-2])
                # Conditions: very small margin and grid_draw within small epsilon of max grid side
                if top_two_margin < 0.06:
                    grid_max_side = max(grid_home_win, grid_away_win)
                    if grid_draw >= grid_max_side - 0.04:
                        draw_prob = float(min(1.0, draw_prob + 0.04))
            except Exception:
                pass

            # Normalize to ensure probabilities sum to 1
            total_prob = home_win_prob + draw_prob + away_win_prob
            if total_prob > 0:
                home_win_prob /= total_prob
                draw_prob /= total_prob
                away_win_prob /= total_prob

            # Outcome from probability argmax
            max_prob = max(home_win_prob, draw_prob, away_win_prob)
            if max_prob == home_win_prob:
                predicted_result = 'H'
            elif max_prob == away_win_prob:
                predicted_result = 'A'
            else:
                predicted_result = 'D'

            # Calculate margin-based confidence (separation between top 2 outcomes)
            # This better captures prediction certainty than just max probability
            probs_sorted = sorted([home_win_prob, draw_prob, away_win_prob], reverse=True)
            margin_confidence = probs_sorted[0] - probs_sorted[1]

            # Also calculate normalized entropy as alternative confidence metric
            # Low entropy = high confidence (peaked distribution)
            probs = np.array([home_win_prob, draw_prob, away_win_prob])
            probs = probs[probs > 0]  # Avoid log(0)
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs))
                entropy_confidence = 1 - (entropy / np.log(3))  # Normalize to [0,1]
            else:
                entropy_confidence = 0.0

            # Combined confidence: weighted average of max_prob and margin
            # max_prob for "how likely is top choice", margin for "how separated from alternatives"
            combined_confidence = 0.6 * max_prob + 0.4 * margin_confidence

            hybrid_pred = {
                'match_id': ml_pred['match_id'],
                'home_team': ml_pred['home_team'],
                'away_team': ml_pred['away_team'],
                'predicted_home_score': int(predicted_home),
                'predicted_away_score': int(predicted_away),
                'home_expected_goals': float(home_lambda),
                'away_expected_goals': float(away_lambda),
                'predicted_result': predicted_result,
                'home_win_probability': float(home_win_prob),
                'draw_probability': float(draw_prob),
                'away_win_probability': float(away_win_prob),
                'confidence': float(combined_confidence),  # New: margin-based confidence
                'max_probability': float(max_prob),  # Classic max prob for reference
                'margin': float(margin_confidence),  # Separation from second choice
                'entropy_confidence': float(entropy_confidence),  # Alternative metric
                # Individual model predictions for reference
                'ml_home_score': ml_pred['predicted_home_score'],
                'ml_away_score': ml_pred['predicted_away_score'],
                'poisson_home_score': poisson_pred['predicted_home_score'],
                'poisson_away_score': poisson_pred['predicted_away_score'],
            }

            hybrid_predictions.append(hybrid_pred)

        return hybrid_predictions

    def fit_goal_temperature(self, recent_matches_df: pd.DataFrame, clamp: tuple = (0.7, 2.0)) -> None:
        """
        Backward-compatible API: fit separate home/away goal temperatures based on
        recent finished matches using the Poisson prior expectations.

        Args:
            recent_matches_df: DataFrame of finished matches with columns
                               ['home_team','away_team','home_score','away_score']
            clamp: (min, max) bounds for temperatures
        """
        try:
            if recent_matches_df is None or len(recent_matches_df) == 0:
                return

            # Actual means
            actual_home_mean = float(recent_matches_df['home_score'].mean())
            actual_away_mean = float(recent_matches_df['away_score'].mean())
            if not np.isfinite(actual_home_mean) or not np.isfinite(actual_away_mean):
                return

            # Predicted (Poisson prior) means over a capped random sample
            sample = recent_matches_df.sample(min(400, len(recent_matches_df)), random_state=42)
            pred_home_vals: list[float] = []
            pred_away_vals: list[float] = []
            for _, m in sample.iterrows():
                p = self.poisson_predictor.predict_match(m['home_team'], m['away_team'])
                pred_home_vals.append(float(p['home_expected_goals']))
                pred_away_vals.append(float(p['away_expected_goals']))
            if not pred_home_vals or not pred_away_vals:
                return
            pred_home_mean = float(np.mean(pred_home_vals))
            pred_away_mean = float(np.mean(pred_away_vals))
            if pred_home_mean <= 0 or pred_away_mean <= 0:
                return

            new_temp_h = float(self.goal_temperature_home) * (actual_home_mean / pred_home_mean)
            new_temp_a = float(self.goal_temperature_away) * (actual_away_mean / pred_away_mean)
            new_temp_h = max(clamp[0], min(clamp[1], new_temp_h))
            new_temp_a = max(clamp[0], min(clamp[1], new_temp_a))

            if abs(new_temp_h - self.goal_temperature_home) > 1e-3 or abs(new_temp_a - self.goal_temperature_away) > 1e-3:
                print(f"[HybridPredictor] Adjusting goal temperatures H {self.goal_temperature_home:.3f}->{new_temp_h:.3f} "
                      f"A {self.goal_temperature_away:.3f}->{new_temp_a:.3f}")
                self.goal_temperature_home = new_temp_h
                self.goal_temperature_away = new_temp_a
        except Exception:
            # Robust to any data issues; fallback to existing temperatures
            pass

    def fit_goal_temperatures_from_validation(self, actual_df: pd.DataFrame, features_df: pd.DataFrame,
                                              clamp: tuple = (0.7, 2.0)) -> None:
        """
        Fit separate home/away temperatures using a validation split by aligning
        combined pre-temperature lambdas to actual average goals.
        """
        try:
            if actual_df is None or features_df is None or len(actual_df) == 0 or len(features_df) == 0:
                return

            # Compute combined lambdas BEFORE temperature scaling, mirroring predict()
            ml_preds = self.ml_predictor.predict(features_df)
            poisson_preds = []
            for _, row in features_df.iterrows():
                poisson_pred = self.poisson_predictor.predict_match(row['home_team'], row['away_team'])
                poisson_preds.append(poisson_pred)

            home_lambdas = []
            away_lambdas = []
            for i in range(len(ml_preds)):
                ml_pred = ml_preds[i]
                pp = poisson_preds[i]
                hl = (self.ml_weight * ml_pred['home_expected_goals'] + self.poisson_weight * pp['home_expected_goals'])
                al = (self.ml_weight * ml_pred['away_expected_goals'] + self.poisson_weight * pp['away_expected_goals'])
                hl = max(hl, self.min_lambda)
                al = max(al, self.min_lambda)
                home_lambdas.append(float(hl))
                away_lambdas.append(float(al))

            pred_home_mean = float(np.mean(home_lambdas)) if home_lambdas else 0.0
            pred_away_mean = float(np.mean(away_lambdas)) if away_lambdas else 0.0

            actual_home_mean = float(actual_df['home_score'].mean())
            actual_away_mean = float(actual_df['away_score'].mean())

            if pred_home_mean > 0 and np.isfinite(actual_home_mean):
                self.goal_temperature_home = max(clamp[0], min(clamp[1], actual_home_mean / pred_home_mean))
            if pred_away_mean > 0 and np.isfinite(actual_away_mean):
                self.goal_temperature_away = max(clamp[0], min(clamp[1], actual_away_mean / pred_away_mean))

            print(f"[HybridPredictor] Fitted validation goal temps: tempH={self.goal_temperature_home:.3f} tempA={self.goal_temperature_away:.3f}")
        except Exception:
            pass

    def _calculate_expected_points(self, grid: np.ndarray) -> tuple:
        """
        Calculate expected Kicktipp points for each scoreline in probability grid.
        Returns (best_home_score, best_away_score, expected_points)

        Args:
            grid: Probability grid (max_goals x max_goals)

        Returns:
            Tuple of (home_score, away_score, expected_points) that maximizes E[points]
        """
        max_goals = grid.shape[0]
        best_score = (0, 0)
        best_expected_points = 0

        # For each possible predicted scoreline
        for pred_h in range(max_goals):
            for pred_a in range(max_goals):
                expected_points = 0

                # Calculate expected points across all actual scorelines
                for actual_h in range(max_goals):
                    for actual_a in range(max_goals):
                        prob = grid[actual_h, actual_a]

                        # Exact score: 4 points
                        if pred_h == actual_h and pred_a == actual_a:
                            expected_points += 4 * prob

                        # Correct goal difference: 3 points
                        elif (pred_h - pred_a) == (actual_h - actual_a):
                            expected_points += 3 * prob

                        # Correct winner: 2 points
                        else:
                            pred_winner = 'H' if pred_h > pred_a else ('A' if pred_a > pred_h else 'D')
                            actual_winner = 'H' if actual_h > actual_a else ('A' if actual_a > actual_h else 'D')

                            if pred_winner == actual_winner:
                                expected_points += 2 * prob

                if expected_points > best_expected_points:
                    best_expected_points = expected_points
                    best_score = (pred_h, pred_a)

        return (best_score[0], best_score[1], best_expected_points)

    def predict_optimized(self, features_df: pd.DataFrame, strategy: str | None = None) -> List[Dict]:
        """
        Predict with optimization strategy for maximizing points.
        This version is simplified and always uses the best strategy.
        """
        base_predictions = self.predict(features_df)
        optimized_predictions = []
        # Currently we use the same flow regardless of strategy value (optimize + safe gating),
        # but accept the parameter for API compatibility and future control.
        _effective_strategy = strategy or self.strategy

        for pred in base_predictions:
            optimized = pred.copy()

            # Always optimize for points
            hg = float(pred['home_expected_goals'])
            ag = float(pred['away_expected_goals'])
            grid = np.zeros((self.max_goals, self.max_goals))
            rho = getattr(self.poisson_predictor, 'rho', 0.0)
            for h in range(self.max_goals):
                for a in range(self.max_goals):
                    p = poisson.pmf(h, max(hg, 1e-9)) * poisson.pmf(a, max(ag, 1e-9))
                    if h == 0 and a == 0:
                        p *= (1.0 + rho)
                    elif (h == 0 and a == 1) or (h == 1 and a == 0):
                        p *= (1.0 - rho)
                    elif h == 1 and a == 1:
                        p *= (1.0 - rho)
                    grid[h, a] = p
            total = np.sum(grid)
            if total > 0:
                grid /= total
            best_h, best_a, _ = self._calculate_expected_points(grid)
            optimized['predicted_home_score'] = int(best_h)
            optimized['predicted_away_score'] = int(best_a)

            # Confidence-adaptive safe strategy for low-confidence predictions
            # Use selected confidence metric for gating to improve points
            confidence = extract_display_confidence(pred)
            if confidence < self.confidence_threshold:
                home_prob = pred['home_win_probability']
                away_prob = pred['away_win_probability']
                draw_prob = pred['draw_probability']

                if home_prob > away_prob and home_prob > draw_prob:
                    optimized['predicted_home_score'] = 2
                    optimized['predicted_away_score'] = 1
                elif away_prob > home_prob and away_prob > draw_prob:
                    optimized['predicted_home_score'] = 1
                    optimized['predicted_away_score'] = 2
                else:
                    optimized['predicted_home_score'] = 1
                    optimized['predicted_away_score'] = 1

            optimized_predictions.append(optimized)

        return optimized_predictions


    def save_models(self, prefix: str = "hybrid"):
        """Save all models."""
        self.ml_predictor.save_models(prefix)
        # Note: Poisson model is trained on-the-fly, no need to save

    def load_models(self, prefix: str = "hybrid") -> bool:
        """Load saved models."""
        return self.ml_predictor.load_models(prefix)

    def evaluate(self, matches_df: pd.DataFrame, features_df: pd.DataFrame) -> Dict:
        """
        Evaluate hybrid model performance.

        Args:
            matches_df: DataFrame with actual results
            features_df: DataFrame with features for prediction

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(features_df)

        correct_results = 0
        correct_differences = 0
        correct_scores = 0
        total_points = 0

        for i, pred in enumerate(predictions):
            actual_home = matches_df.iloc[i]['home_score']
            actual_away = matches_df.iloc[i]['away_score']

            pred_home = pred['predicted_home_score']
            pred_away = pred['predicted_away_score']

            # Exact score: 4 points
            if pred_home == actual_home and pred_away == actual_away:
                correct_scores += 1
                total_points += 4
                continue

            # Correct goal difference: 3 points
            if (pred_home - pred_away) == (actual_home - actual_away):
                correct_differences += 1
                total_points += 3
                continue

            # Correct winner: 2 points
            pred_winner = 'H' if pred_home > pred_away else ('A' if pred_away > pred_home else 'D')
            actual_winner = 'H' if actual_home > actual_away else ('A' if actual_away > actual_home else 'D')

            if pred_winner == actual_winner:
                correct_results += 1
                total_points += 2

        n = len(predictions)

        print("\n=== Hybrid Model Evaluation ===")
        print(f"Total matches: {n}")
        print(f"Correct scores (4pts): {correct_scores} ({correct_scores/n*100:.1f}%)")
        print(f"Correct differences (3pts): {correct_differences} ({correct_differences/n*100:.1f}%)")
        print(f"Correct results (2pts): {correct_results} ({correct_results/n*100:.1f}%)")
        print(f"Total points: {total_points}")
        print(f"Average points per match: {total_points/n:.2f}")
        print(f"Expected points per season (38 games): {(total_points/n)*38:.1f}")

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


