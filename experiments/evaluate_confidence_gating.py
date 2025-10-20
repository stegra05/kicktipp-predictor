#!/usr/bin/env python3
"""
Evaluate impact of different confidence metrics on safe-strategy gating.
Compares average points using 'confidence' (baseline), 'margin', and 'entropy_confidence'.
"""

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor


def calculate_points(pred_h: int, pred_a: int, act_h: int, act_a: int) -> int:
    if pred_h == act_h and pred_a == act_a:
        return 4
    if (pred_h - pred_a) == (act_h - act_a):
        return 3
    pred_w = 'H' if pred_h > pred_a else ('A' if pred_a > pred_h else 'D')
    act_w = 'H' if act_h > act_a else ('A' if act_a > act_h else 'D')
    return 2 if pred_w == act_w else 0


def build_grid(hg: float, ag: float, max_goals: int, rho: float) -> np.ndarray:
    from scipy.stats import poisson  # type: ignore
    grid = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            p = poisson.pmf(h, max(hg, 1e-9)) * poisson.pmf(a, max(ag, 1e-9))
            if h == 0 and a == 0:
                p *= (1.0 + rho)
            elif (h == 0 and a == 1) or (h == 1 and a == 0):
                p *= (1.0 - rho)
            elif h == 1 and a == 1:
                p *= (1.0 - rho)
            grid[h, a] = p
    total = float(np.sum(grid))
    if total > 0:
        grid /= total
    return grid


def derive_margin(pH: float, pD: float, pA: float) -> float:
    top2 = sorted([pH, pD, pA], reverse=True)[:2]
    return float(top2[0] - top2[1])


def derive_entropy_confidence(pH: float, pD: float, pA: float) -> float:
    probs = np.array([pH, pD, pA], dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(1.0 - (entropy / math.log(3)))


def main():
    print("=" * 80)
    print("Evaluating confidence metrics for safe-strategy gating")
    print("=" * 80)

    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    predictor = HybridPredictor()

    if not predictor.load_models("hybrid"):
        print("ERROR: No trained models found. Run train_model.py first.")
        return

    current_season = data_fetcher.get_current_season()
    start_season = current_season - 2
    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)
    finished = [m for m in all_matches if m.get('is_finished')]
    if not finished:
        print("No finished matches available in cache.")
        return

    print("Creating features...")
    features_df = feature_engineer.create_features_from_matches(all_matches)
    if features_df is None or len(features_df) == 0:
        print("No features created.")
        return

    split_idx = int(len(features_df) * 0.7)
    test_df = features_df[split_idx:]
    if len(test_df) == 0:
        print("Empty test set.")
        return

    # Prepare test features
    test_features = test_df.drop(columns=['home_score', 'away_score', 'goal_difference', 'result'], errors='ignore')

    # Train Poisson only on prior matches relative to test start date
    id_to_date = {m['match_id']: m['date'] for m in all_matches if m.get('match_id') is not None and m.get('date') is not None}
    test_ids = list(test_df['match_id']) if 'match_id' in test_df.columns else []
    test_dates = [id_to_date.get(mid) for mid in test_ids]
    test_dates = [d for d in test_dates if d is not None]
    if test_dates:
        first_test_date = min(test_dates)
        hist_prior = [m for m in finished if m.get('date') is not None and m['date'] < first_test_date]
    else:
        hist_prior = finished
    hist_df = pd.DataFrame(hist_prior)
    predictor.poisson_predictor.train(hist_df)

    print("Generating base predictions...")
    base_predictions = predictor.predict(test_features)
    actuals = test_df.to_dict('records')

    # Evaluate different gating metrics
    metrics_to_try = ["confidence", "margin", "entropy_confidence"]
    results: Dict[str, Dict[str, float]] = {}

    rho = getattr(predictor.poisson_predictor, 'rho', 0.0)
    max_goals = predictor.max_goals
    thr = float(predictor.confidence_threshold)

    total_points_by_metric = {m: 0 for m in metrics_to_try}
    n = len(base_predictions)

    for pred, actual in zip(base_predictions, actuals):
        hg = float(pred['home_expected_goals'])
        ag = float(pred['away_expected_goals'])
        grid = build_grid(hg, ag, max_goals, rho)
        best_h, best_a, _ = predictor._calculate_expected_points(grid)

        pH = float(pred['home_win_probability'])
        pD = float(pred['draw_probability'])
        pA = float(pred['away_win_probability'])

        values = {
            'confidence': float(pred.get('confidence', 0.0)),
            'margin': float(pred.get('margin', derive_margin(pH, pD, pA))),
            'entropy_confidence': float(pred.get('entropy_confidence', derive_entropy_confidence(pH, pD, pA))),
        }

        for metric in metrics_to_try:
            conf = float(values.get(metric, 0.0))
            ph, pa = int(best_h), int(best_a)
            if conf < thr:
                # Safe override
                if pH > pA and pH > pD:
                    ph, pa = 2, 1
                elif pA > pH and pA > pD:
                    ph, pa = 1, 2
                else:
                    ph, pa = 1, 1

            pts = calculate_points(ph, pa, int(actual['home_score']), int(actual['away_score']))
            total_points_by_metric[metric] += pts

    print("\nAverage points per match by gating metric:")
    rows: List[Tuple[str, float]] = []
    for metric, total_pts in total_points_by_metric.items():
        avg = total_pts / n if n > 0 else 0.0
        rows.append((metric, avg))
    rows.sort(key=lambda x: x[1], reverse=True)
    for m, avg in rows:
        print(f"  {m:20s} {avg:.3f}")

    baseline = next((avg for m, avg in rows if m == 'confidence'), None)
    best = rows[0] if rows else None
    if baseline is not None and best is not None:
        lift = best[1] - baseline
        print(f"\nLift over baseline ('confidence'): {lift:+.3f} avg pts/match")
    print("\nDone.")


if __name__ == "__main__":
    main()


