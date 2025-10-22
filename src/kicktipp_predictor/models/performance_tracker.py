import json
import os
from datetime import datetime


class PerformanceTracker:
    """Tracks prediction performance over time, recording predictions, results, and points."""

    def __init__(self, storage_dir: str = "data/predictions"):
        """Initializes the tracker, creating storage directories and loading historical data."""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.predictions_file = os.path.join(storage_dir, "predictions.json")
        self.performance_file = os.path.join(storage_dir, "performance.json")

        self.predictions = self._load_json(self.predictions_file)
        self.performance_history = self._load_json(self.performance_file)

    def _load_json(self, file_path: str) -> list[dict]:
        """Loads a list of dictionaries from a JSON file, returning an empty list if not found."""
        if os.path.exists(file_path):
            with open(file_path) as f:
                return json.load(f)
        return []

    def _save_json(self, data: list[dict], file_path: str):
        """Saves a list of dictionaries to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def record_predictions(self, predictions: list[dict], matchday: int = None):
        """Records a list of new predictions, updating existing ones if they are not yet evaluated.

        Args:
            predictions: A list of prediction dictionaries.
            matchday: The matchday number for these predictions.
        """
        timestamp = datetime.now().isoformat()
        for pred in predictions:
            # Standardize the prediction record structure.
            prediction_record = {
                "timestamp": timestamp,
                "matchday": matchday,
                "match_id": pred.get("match_id"),
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "predicted_home_score": pred["predicted_home_score"],
                "predicted_away_score": pred["predicted_away_score"],
                "home_win_probability": pred.get("home_win_probability"),
                "draw_probability": pred.get("draw_probability"),
                "away_win_probability": pred.get("away_win_probability"),
                "confidence": pred.get("confidence"),
                "actual_home_score": None,
                "actual_away_score": None,
                "points_earned": None,
                "is_evaluated": False,
            }

            # If a prediction for this match already exists and hasn't been scored, update it.
            # Otherwise, add it as a new prediction.
            existing_idx = self._find_prediction_index(
                pred.get("match_id"), pred["home_team"], pred["away_team"]
            )
            if existing_idx is not None and not self.predictions[existing_idx]["is_evaluated"]:
                self.predictions[existing_idx].update(prediction_record)
            else:
                self.predictions.append(prediction_record)

        self._save_json(self.predictions, self.predictions_file)
        print(f"Recorded {len(predictions)} predictions")

    def _find_prediction_index(self, match_id, home_team, away_team) -> int | None:
        """Finds the index of a prediction by match ID or team names for unevaluated matches."""
        for i, pred in enumerate(self.predictions):
            if pred.get("match_id") and pred.get("match_id") == match_id:
                return i
            # Fallback for matches without a stable ID: match by teams if not yet evaluated.
            if (
                not pred["is_evaluated"]
                and pred["home_team"] == home_team
                and pred["away_team"] == away_team
            ):
                return i
        return None

    def update_results(self, results: list[dict]) -> int:
        """Updates predictions with actual match results and calculates points.

        Args:
            results: A list of result dictionaries containing actual scores.

        Returns:
            The number of predictions that were updated.
        """
        updated_count = 0
        for result in results:
            pred_idx = self._find_prediction_index(
                result.get("match_id"), result["home_team"], result["away_team"]
            )

            if pred_idx is not None and not self.predictions[pred_idx]["is_evaluated"]:
                pred = self.predictions[pred_idx]
                points = self._calculate_points(
                    pred["predicted_home_score"],
                    pred["predicted_away_score"],
                    result["home_score"],
                    result["away_score"],
                )

                # Update the prediction record with the actual outcome and calculated points.
                pred.update({
                    "actual_home_score": result["home_score"],
                    "actual_away_score": result["away_score"],
                    "points_earned": points,
                    "is_evaluated": True,
                    "evaluation_date": datetime.now().isoformat(),
                })
                updated_count += 1

        if updated_count > 0:
            self._save_json(self.predictions, self.predictions_file)
            self._update_performance_stats()
            print(f"Updated {updated_count} predictions with results")
        return updated_count

    def _calculate_points(
        self, pred_home: int, pred_away: int, actual_home: int, actual_away: int
    ) -> int:
        """Calculates points for a prediction based on Kicktipp scoring rules.

        - 4 points for the exact score.
        - 3 points for the correct goal difference.
        - 2 points for the correct winner (or draw).
        - 0 points otherwise.
        """
        if pred_home == actual_home and pred_away == actual_away:
            return 4  # Exact score
        if (pred_home - pred_away) == (actual_home - actual_away):
            return 3  # Correct goal difference

        pred_winner = "H" if pred_home > pred_away else ("A" if pred_away > pred_home else "D")
        actual_winner = "H" if actual_home > actual_away else ("A" if actual_away > actual_home else "D")
        if pred_winner == actual_winner:
            return 2  # Correct winner/draw

        return 0

    def _update_performance_stats(self):
        """Calculates and saves overall performance statistics from evaluated predictions."""
        evaluated = [p for p in self.predictions if p["is_evaluated"]]
        if not evaluated:
            return

        total_matches = len(evaluated)
        total_points = sum(p["points_earned"] for p in evaluated)
        points_dist = {0: 0, 2: 0, 3: 0, 4: 0}
        for pred in evaluated:
            points_dist[pred["points_earned"]] += 1

        # Aggregate performance by matchday.
        matchday_stats = {}
        for pred in evaluated:
            md = pred.get("matchday")
            if md is not None:
                if md not in matchday_stats:
                    matchday_stats[md] = {"matches": 0, "points": 0}
                matchday_stats[md]["matches"] += 1
                matchday_stats[md]["points"] += pred["points_earned"]

        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_predictions": total_matches,
            "total_points": total_points,
            "avg_points_per_match": total_points / total_matches if total_matches else 0,
            "points_distribution": points_dist,
            "matchday_stats": matchday_stats,
        }
        self.performance_history.append(summary)
        self._save_json(self.performance_history, self.performance_file)

    def get_current_stats(self) -> dict:
        """Returns a dictionary of the current overall performance statistics."""
        evaluated = [p for p in self.predictions if p["is_evaluated"]]
        if not evaluated:
            return {"total_predictions": 0, "total_points": 0, "avg_points_per_match": 0}

        total_matches = len(evaluated)
        total_points = sum(p["points_earned"] for p in evaluated)
        points_dist = {0: 0, 2: 0, 3: 0, 4: 0}
        for pred in evaluated:
            points_dist[pred["points_earned"]] += 1

        return {
            "total_predictions": total_matches,
            "total_points": total_points,
            "avg_points_per_match": total_points / total_matches if total_matches else 0,
            "exact_scores": points_dist[4],
            "correct_differences": points_dist[3],
            "correct_results": points_dist[2],
            "incorrect": points_dist[0],
        }

    def get_unevaluated_predictions(self) -> list[dict]:
        """Returns a list of predictions that have not yet been evaluated."""
        return [p for p in self.predictions if not p["is_evaluated"]]

    def get_matchday_stats(self, matchday: int) -> dict:
        """Returns performance statistics for a specific matchday."""
        preds = [p for p in self.predictions if p.get("matchday") == matchday and p["is_evaluated"]]
        if not preds:
            return {"matchday": matchday, "matches": 0, "points": 0, "avg_points": 0}

        points = sum(p["points_earned"] for p in preds)
        return {
            "matchday": matchday,
            "matches": len(preds),
            "points": points,
            "avg_points": points / len(preds),
        }

    def print_summary(self):
        """Prints a formatted summary of the current prediction performance."""
        stats = self.get_current_stats()
        if stats["total_predictions"] == 0:
            print("No performance data available yet.")
            return

        total_matches = stats["total_predictions"]

        # Helper to calculate percentage safely.
        def get_perc(value):
            return (value / total_matches * 100) if total_matches > 0 else 0

        print("\n" + "=" * 50)
        print("PREDICTION PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Total Points: {stats['total_points']}")
        print(f"Average Points/Match: {stats['avg_points_per_match']:.2f}")
        print("\nAccuracy Breakdown:")
        print(f"  Exact Scores (4pts):      {stats['exact_scores']:3d} ({get_perc(stats['exact_scores']):5.1f}%)")
        print(f"  Correct Differences (3pts): {stats['correct_differences']:3d} ({get_perc(stats['correct_differences']):5.1f}%)")
        print(f"  Correct Results (2pts):   {stats['correct_results']:3d} ({get_perc(stats['correct_results']):5.1f}%)")
        print(f"  Incorrect (0pts):         {stats['incorrect']:3d} ({get_perc(stats['incorrect']):5.1f}%)")
        print("=" * 50 + "\n")
