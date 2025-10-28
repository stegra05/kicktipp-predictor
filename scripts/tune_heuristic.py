"""
Lightweight tuner for tiered scoreline heuristic thresholds.

This does NOT retrain the main model. It:
- Loads the pre-trained GoalDifferencePredictor
- Loads a validation dataset (current season or a specified season)
- Uses Optuna to tune tier thresholds to maximize average points (PPG)
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import optuna

from kicktipp_predictor.config import get_config
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.metrics import KicktippScoring
from kicktipp_predictor.predictor import GoalDifferencePredictor


def _predict_with_thresholds(
    pred_gd: np.ndarray, t1: float, t2: float, t3: float, draw_goal: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predicted scorelines from predicted goal differences using tiered rules."""
    ph = np.zeros_like(pred_gd, dtype=int)
    pa = np.zeros_like(pred_gd, dtype=int)
    for i, gd in enumerate(pred_gd):
        if gd >= t3:
            ph[i], pa[i] = 3, 0
        elif gd >= t2:
            ph[i], pa[i] = 2, 0
        elif gd >= t1:
            ph[i], pa[i] = 2, 1
        elif gd <= -t3:
            ph[i], pa[i] = 0, 3
        elif gd <= -t2:
            ph[i], pa[i] = 0, 2
        elif gd <= -t1:
            ph[i], pa[i] = 1, 2
        else:
            ph[i], pa[i] = draw_goal, draw_goal
    return ph, pa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024, help="Validation season to use")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    args = parser.parse_args()

    cfg = get_config()

    # Load model
    predictor = GoalDifferencePredictor(cfg)
    predictor.load_model()

    # Load data for the season
    dl = DataLoader()
    matches = dl.fetch_season_matches(args.season)
    finished = [m for m in matches if m.get("is_finished")]
    if not finished:
        raise SystemExit("No finished matches in selected season.")
    feats = dl.create_features_from_matches(finished)
    if feats.empty:
        raise SystemExit("Empty features for the selected season.")

    # Align features to model and get predicted GD
    X = feats.reindex(columns=predictor.feature_columns).fillna(0.0)
    pred_gd = predictor.model.predict(X)  # type: ignore

    actual_home = feats["home_score"].astype(int).to_numpy()
    actual_away = feats["away_score"].astype(int).to_numpy()

    def objective(trial: optuna.trial.Trial) -> float:
        t1 = trial.suggest_float("t1", 0.2, 1.0)
        t2 = trial.suggest_float("t2", 0.8, 2.0)
        t3 = trial.suggest_float("t3", 1.6, 3.2)
        # Ensure ordering t1 < t2 < t3 by penalizing invalid configs
        if not (t1 < t2 < t3):
            return -1e6
        draw_goal = trial.suggest_categorical("draw_goal", [0, 1])

        ph, pa = _predict_with_thresholds(pred_gd, t1, t2, t3, int(draw_goal))
        points = KicktippScoring.compute_points(ph, pa, actual_home, actual_away)
        return float(np.mean(points))

    study = optuna.create_study(
        direction="maximize",
        study_name="tiered_heuristic_tuning",
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)

    best = study.best_params
    best_ppg = study.best_value
    print(f"Best params: {best}  (PPG={best_ppg:.4f})")

    # Save to config YAML location: update best_params.yaml
    # We do minimal IO here: append keys or instruct user to copy values.
    # Since config reading is one-way, we simply print in YAML form.
    print("\nAdd the following to src/kicktipp_predictor/config/best_params.yaml:")
    print(f"use_tiered_heuristic: true")
    print(f"gd_tier_t1: {best['t1']}")
    print(f"gd_tier_t2: {best['t2']}")
    print(f"gd_tier_t3: {best['t3']}")
    print(f"draw_goal: {best['draw_goal']}")


if __name__ == "__main__":
    main()


