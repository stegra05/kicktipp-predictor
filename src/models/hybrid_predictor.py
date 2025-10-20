import pandas as pd
import numpy as np
from typing import Dict, List
from .ml_model import MLPredictor
from .poisson_model import PoissonPredictor
from scipy.stats import poisson


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
        # Higher = more weight to Poisson grid, lower = more weight to ML classifier
        self.prob_blend_alpha = 0.65

        # Minimum lambda to avoid degenerate predictions
        self.min_lambda = 0.25

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

            # Build probability grid with DC correction
            max_goals = 7
            grid = np.zeros((max_goals, max_goals))
            for hg in range(max_goals):
                for ag in range(max_goals):
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

    def predict_optimized(self, features_df: pd.DataFrame,
                         strategy: str = 'balanced') -> List[Dict]:
        """
        Predict with optimization strategy for maximizing points.

        Args:
            features_df: DataFrame with match features
            strategy: Prediction strategy
                - 'balanced': Default hybrid predictions
                - 'conservative': Favor more likely scorelines (fewer goals)
                - 'aggressive': Go for exact scores with higher risk
                - 'safe': Prioritize correct winner over exact scores

        Returns:
            List of optimized prediction dictionaries
        """
        base_predictions = self.predict(features_df)

        if strategy == 'balanced':
            return base_predictions

        optimized_predictions = []

        for pred in base_predictions:
            optimized = pred.copy()

            if strategy == 'conservative':
                # Reduce predicted goals slightly (conservative)
                home_expected = pred['home_expected_goals']
                away_expected = pred['away_expected_goals']

                # Floor instead of round for lower scores
                optimized['predicted_home_score'] = max(0, int(home_expected * 0.9))
                optimized['predicted_away_score'] = max(0, int(away_expected * 0.9))

            elif strategy == 'aggressive':
                # Try to predict exact common scorelines
                home_exp = pred['home_expected_goals']
                away_exp = pred['away_expected_goals']

                # Round to nearest common scoreline
                optimized['predicted_home_score'] = self._round_to_common(home_exp)
                optimized['predicted_away_score'] = self._round_to_common(away_exp)

            elif strategy == 'safe':
                # Prioritize getting the winner right
                home_prob = pred['home_win_probability']
                away_prob = pred['away_win_probability']
                draw_prob = pred['draw_probability']

                if home_prob > away_prob and home_prob > draw_prob:
                    # Predict home win with typical score
                    optimized['predicted_home_score'] = 2
                    optimized['predicted_away_score'] = 1
                elif away_prob > home_prob and away_prob > draw_prob:
                    # Predict away win with typical score
                    optimized['predicted_home_score'] = 1
                    optimized['predicted_away_score'] = 2
                else:
                    # Predict draw with typical score
                    optimized['predicted_home_score'] = 1
                    optimized['predicted_away_score'] = 1

            optimized_predictions.append(optimized)

        return optimized_predictions

    def _round_to_common(self, expected_goals: float) -> int:
        """Round expected goals to common scoreline values."""
        # Common scores in football: 0, 1, 2, (3)
        if expected_goals < 0.5:
            return 0
        elif expected_goals < 1.5:
            return 1
        elif expected_goals < 2.5:
            return 2
        else:
            return 3

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
