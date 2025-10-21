import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np


class PerformanceTracker:
    """
    Tracks prediction performance over time.
    Records predictions, actual results, and calculates points.
    """

    def __init__(self, storage_dir: str = "data/predictions"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.predictions_file = os.path.join(storage_dir, "predictions.json")
        self.performance_file = os.path.join(storage_dir, "performance.json")

        self.predictions = self._load_predictions()
        self.performance_history = self._load_performance()

    def _load_predictions(self) -> List[Dict]:
        """Load saved predictions from disk."""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return []

    def _save_predictions(self):
        """Save predictions to disk."""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)

    def _load_performance(self) -> List[Dict]:
        """Load performance history from disk."""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        return []

    def _save_performance(self):
        """Save performance history to disk."""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)

    def record_predictions(self, predictions: List[Dict], matchday: int = None):
        """
        Record new predictions.

        Args:
            predictions: List of prediction dictionaries
            matchday: Optional matchday number
        """
        timestamp = datetime.now().isoformat()

        for pred in predictions:
            prediction_record = {
                'timestamp': timestamp,
                'matchday': matchday,
                'match_id': pred.get('match_id'),
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'predicted_home_score': pred['predicted_home_score'],
                'predicted_away_score': pred['predicted_away_score'],
                'home_win_probability': pred.get('home_win_probability'),
                'draw_probability': pred.get('draw_probability'),
                'away_win_probability': pred.get('away_win_probability'),
                'confidence': pred.get('confidence'),
                'actual_home_score': None,
                'actual_away_score': None,
                'points_earned': None,
                'is_evaluated': False,
            }

            # Check if prediction already exists and update
            existing_idx = self._find_prediction_index(pred.get('match_id'),
                                                      pred['home_team'],
                                                      pred['away_team'])

            if existing_idx is not None:
                # Update existing prediction if not yet evaluated
                if not self.predictions[existing_idx]['is_evaluated']:
                    self.predictions[existing_idx].update(prediction_record)
            else:
                self.predictions.append(prediction_record)

        self._save_predictions()
        print(f"Recorded {len(predictions)} predictions")

    def _find_prediction_index(self, match_id, home_team, away_team) -> int:
        """Find index of existing prediction."""
        for i, pred in enumerate(self.predictions):
            if pred.get('match_id') == match_id:
                return i
            if (pred['home_team'] == home_team and
                pred['away_team'] == away_team and
                not pred['is_evaluated']):
                return i
        return None

    def update_results(self, results: List[Dict]):
        """
        Update predictions with actual results and calculate points.

        Args:
            results: List of result dictionaries with actual scores
        """
        updated_count = 0

        for result in results:
            match_id = result.get('match_id')
            home_team = result['home_team']
            away_team = result['away_team']
            actual_home = result['home_score']
            actual_away = result['away_score']

            # Find corresponding prediction
            pred_idx = self._find_prediction_index(match_id, home_team, away_team)

            if pred_idx is not None and not self.predictions[pred_idx]['is_evaluated']:
                pred = self.predictions[pred_idx]
                pred_home = pred['predicted_home_score']
                pred_away = pred['predicted_away_score']

                # Calculate points
                points = self._calculate_points(pred_home, pred_away,
                                               actual_home, actual_away)

                # Update prediction record
                pred['actual_home_score'] = actual_home
                pred['actual_away_score'] = actual_away
                pred['points_earned'] = points
                pred['is_evaluated'] = True
                pred['evaluation_date'] = datetime.now().isoformat()

                updated_count += 1

        self._save_predictions()

        if updated_count > 0:
            self._update_performance_stats()
            print(f"Updated {updated_count} predictions with results")

        return updated_count

    def _calculate_points(self, pred_home: int, pred_away: int,
                         actual_home: int, actual_away: int) -> int:
        """
        Calculate points for a prediction using the scoring rules.

        Returns:
            Points earned (0, 2, 3, or 4)
        """
        # Exact score: 4 points
        if pred_home == actual_home and pred_away == actual_away:
            return 4

        # Correct goal difference: 3 points
        if (pred_home - pred_away) == (actual_home - actual_away):
            return 3

        # Correct winner: 2 points
        pred_winner = 'H' if pred_home > pred_away else ('A' if pred_away > pred_home else 'D')
        actual_winner = 'H' if actual_home > actual_away else ('A' if actual_away > actual_home else 'D')

        if pred_winner == actual_winner:
            return 2

        return 0

    def _update_performance_stats(self):
        """Update overall performance statistics."""
        evaluated_predictions = [p for p in self.predictions if p['is_evaluated']]

        if not evaluated_predictions:
            return

        total_matches = len(evaluated_predictions)
        total_points = sum(p['points_earned'] for p in evaluated_predictions)

        points_distribution = {0: 0, 2: 0, 3: 0, 4: 0}
        for pred in evaluated_predictions:
            points_distribution[pred['points_earned']] += 1

        # Calculate by matchday if available
        matchday_stats = {}
        for pred in evaluated_predictions:
            md = pred.get('matchday')
            if md is not None:
                if md not in matchday_stats:
                    matchday_stats[md] = {'matches': 0, 'points': 0}
                matchday_stats[md]['matches'] += 1
                matchday_stats[md]['points'] += pred['points_earned']

        performance_summary = {
            'last_updated': datetime.now().isoformat(),
            'total_predictions': total_matches,
            'total_points': total_points,
            'avg_points_per_match': total_points / total_matches if total_matches > 0 else 0,
            'exact_scores': points_distribution[4],
            'correct_differences': points_distribution[3],
            'correct_results': points_distribution[2],
            'incorrect': points_distribution[0],
            'exact_score_rate': points_distribution[4] / total_matches if total_matches > 0 else 0,
            'matchday_stats': matchday_stats,
        }

        self.performance_history.append(performance_summary)
        self._save_performance()

    def get_current_stats(self) -> Dict:
        """Get current performance statistics."""
        evaluated_predictions = [p for p in self.predictions if p['is_evaluated']]

        if not evaluated_predictions:
            return {
                'total_predictions': 0,
                'total_points': 0,
                'avg_points_per_match': 0,
            }

        total_matches = len(evaluated_predictions)
        total_points = sum(p['points_earned'] for p in evaluated_predictions)

        points_dist = {0: 0, 2: 0, 3: 0, 4: 0}
        for pred in evaluated_predictions:
            points_dist[pred['points_earned']] += 1

        return {
            'total_predictions': total_matches,
            'total_points': total_points,
            'avg_points_per_match': total_points / total_matches if total_matches > 0 else 0,
            'exact_scores': points_dist[4],
            'correct_differences': points_dist[3],
            'correct_results': points_dist[2],
            'incorrect': points_dist[0],
            'exact_score_percentage': (points_dist[4] / total_matches * 100) if total_matches > 0 else 0,
            'difference_percentage': (points_dist[3] / total_matches * 100) if total_matches > 0 else 0,
            'result_percentage': (points_dist[2] / total_matches * 100) if total_matches > 0 else 0,
            'incorrect_percentage': (points_dist[0] / total_matches * 100) if total_matches > 0 else 0,
        }

    def get_unevaluated_predictions(self) -> List[Dict]:
        """Get predictions that haven't been evaluated yet."""
        return [p for p in self.predictions if not p['is_evaluated']]

    def get_matchday_stats(self, matchday: int) -> Dict:
        """Get statistics for a specific matchday."""
        matchday_preds = [p for p in self.predictions
                         if p.get('matchday') == matchday and p['is_evaluated']]

        if not matchday_preds:
            return {'matchday': matchday, 'matches': 0, 'points': 0}

        total_points = sum(p['points_earned'] for p in matchday_preds)

        return {
            'matchday': matchday,
            'matches': len(matchday_preds),
            'points': total_points,
            'avg_points': total_points / len(matchday_preds),
        }

    def print_summary(self):
        """Print a summary of current performance."""
        stats = self.get_current_stats()

        print("\n" + "="*50)
        print("PREDICTION PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Total Points: {stats['total_points']}")
        print(f"Average Points/Match: {stats['avg_points_per_match']:.2f}")
        print()
        print("Accuracy Breakdown:")
        print(f"  Exact Scores (4pts):      {stats['exact_scores']:3d} ({stats['exact_score_percentage']:5.1f}%)")
        print(f"  Correct Differences (3pts): {stats['correct_differences']:3d} ({stats['difference_percentage']:5.1f}%)")
        print(f"  Correct Results (2pts):   {stats['correct_results']:3d} ({stats['result_percentage']:5.1f}%)")
        print(f"  Incorrect (0pts):         {stats['incorrect']:3d} ({stats['incorrect_percentage']:5.1f}%)")
        print()

        if stats['total_predictions'] > 0:
            projected_season = stats['avg_points_per_match'] * 38
            print(f"Projected Season Total (38 matchdays): {projected_season:.1f} points")

        print("="*50 + "\n")



