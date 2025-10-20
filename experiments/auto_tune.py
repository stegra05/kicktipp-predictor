#!/usr/bin/env python3
"""
Iterative hyperparameter tuner with time-series CV, composite objective, and zoom-in.
Writes trial logs to CSV and saves best params to config/best_params.yaml.
"""

import os
import sys
import csv
import math
import argparse
import random
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Ensure project root is on sys.path so `src.*` imports work when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from src.scraper.data_fetcher import DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.hybrid_predictor import HybridPredictor

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class TrialResult:
    params: Dict[str, float]
    avg_points: float
    obj_score: float
    zero_zero_rate: float
    pred_H: float
    pred_D: float
    pred_A: float
    home_mean_pred: float
    away_mean_pred: float
    home_mean_act: float
    away_mean_act: float


def calculate_points(predictions: List[Dict], actuals: List[Dict]) -> int:
    total_points = 0
    for pred, act in zip(predictions, actuals):
        ph, pa = pred['predicted_home_score'], pred['predicted_away_score']
        ah, aa = int(act['home_score']), int(act['away_score'])
        if ph == ah and pa == aa:
            total_points += 4
        elif (ph - pa) == (ah - aa):
            total_points += 3
        elif (ph > pa and ah > aa) or (ph < pa and ah < aa) or (ph == pa and ah == aa):
            total_points += 2
    return total_points


def outcome(h: int, a: int) -> str:
    return 'H' if h > a else ('A' if a > h else 'D')


def realism_metrics(preds: List[Dict], acts: List[Dict]) -> Tuple[float, float, float, float, float, float]:
    n = len(preds)
    zero_zero = sum(1 for p in preds if p['predicted_home_score'] == 0 and p['predicted_away_score'] == 0) / max(n, 1)
    pred_outcomes = [outcome(p['predicted_home_score'], p['predicted_away_score']) for p in preds]
    pred_H = pred_outcomes.count('H') / max(n, 1)
    pred_D = pred_outcomes.count('D') / max(n, 1)
    pred_A = pred_outcomes.count('A') / max(n, 1)
    home_mean_pred = float(np.mean([p['predicted_home_score'] for p in preds]))
    away_mean_pred = float(np.mean([p['predicted_away_score'] for p in preds]))
    home_mean_act = float(np.mean([int(a['home_score']) for a in acts]))
    away_mean_act = float(np.mean([int(a['away_score']) for a in acts]))
    return zero_zero, pred_H, pred_D, pred_A, home_mean_pred, away_mean_pred, home_mean_act, away_mean_act


def composite_objective(avg_points: float,
                        zero_zero_rate: float,
                        pred_H: float,
                        pred_D: float,
                        pred_A: float,
                        home_pred: float,
                        away_pred: float,
                        home_act: float,
                        away_act: float,
                        weights: Dict[str, float]) -> float:
    # Goal mean penalty beyond ±10%
    home_pen = max(0.0, abs(home_pred - home_act) / max(home_act, 1e-6) - 0.10)
    away_pen = max(0.0, abs(away_pred - away_act) / max(away_act, 1e-6) - 0.10)
    goal_pen = 0.5 * home_pen + 0.5 * away_pen

    # Outcome distribution penalties
    draw_pen = max(0.0, pred_D - 0.30)
    away_penalty = max(0.0, 0.25 - pred_A)
    zero_pen = max(0.0, zero_zero_rate - 0.15)

    penalty = (weights['goal'] * goal_pen +
               weights['draw'] * draw_pen +
               weights['away'] * away_penalty +
               weights['zero'] * zero_pen)

    return avg_points - penalty


def evaluate_params(params: Dict[str, float],
                    features_df,
                    tscv: TimeSeriesSplit,
                    strategy: str,
                    confidence_threshold: float,
                    weights: Dict[str, float]) -> TrialResult:
    fold_points: List[float] = []
    fold_objs: List[float] = []
    # Accumulate realism metrics across folds
    zero_rates: List[float] = []
    pred_Hs: List[float] = []
    pred_Ds: List[float] = []
    pred_As: List[float] = []
    home_preds: List[float] = []
    away_preds: List[float] = []
    home_acts: List[float] = []
    away_acts: List[float] = []

    for train_idx, test_idx in tscv.split(features_df):
        train_df = features_df.iloc[train_idx]
        test_df = features_df.iloc[test_idx]
        test_feats = test_df.drop(columns=['home_score','away_score','goal_difference','result'], errors='ignore')

        predictor = HybridPredictor()
        predictor.ml_weight = float(params['ml_weight'])
        predictor.poisson_weight = 1.0 - predictor.ml_weight
        predictor.prob_blend_alpha = float(params['prob_blend_alpha'])
        predictor.min_lambda = float(params['min_lambda'])
        predictor.goal_temperature = float(params['goal_temperature'])
        predictor.confidence_threshold = float(confidence_threshold)

        predictor.train(train_df)
        # Use simplified maximize-points predictor (strategy handled internally; threshold set above)
        preds = predictor.predict_optimized(test_feats)

        acts = test_df.to_dict('records')

        pts = calculate_points(preds, acts) / max(len(acts), 1)
        fold_points.append(pts)

        z0, pH, pD, pA, hpred, apred, hact, aact = realism_metrics(preds, acts)
        zero_rates.append(z0)
        pred_Hs.append(pH)
        pred_Ds.append(pD)
        pred_As.append(pA)
        home_preds.append(hpred)
        away_preds.append(apred)
        home_acts.append(hact)
        away_acts.append(aact)

        obj = composite_objective(pts, z0, pH, pD, pA, hpred, apred, hact, aact, weights)
        fold_objs.append(obj)

    avg_points = float(np.mean(fold_points))
    avg_obj = float(np.mean(fold_objs))

    return TrialResult(
        params=params,
        avg_points=avg_points,
        obj_score=avg_obj,
        zero_zero_rate=float(np.mean(zero_rates)),
        pred_H=float(np.mean(pred_Hs)),
        pred_D=float(np.mean(pred_Ds)),
        pred_A=float(np.mean(pred_As)),
        home_mean_pred=float(np.mean(home_preds)),
        away_mean_pred=float(np.mean(away_preds)),
        home_mean_act=float(np.mean(home_acts)),
        away_mean_act=float(np.mean(away_acts)),
    )


def successive_halving(features_df, n_splits: int, max_trials: int) -> Dict[str, float]:
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()

    # TimeSeries CV (modest folds for stability vs. runtime)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Expanded parameter ranges
    ml_weights = [0.60, 0.65, 0.70, 0.75, 0.80]
    prob_alphas = [0.30, 0.35, 0.40, 0.45, 0.50]
    min_lambdas = [0.15, 0.20, 0.25, 0.30]
    goal_temps = [1.4, 1.5, 1.6, 1.7]
    strategies = ['safe', 'conservative']
    thresholds = [0.40, 0.45, 0.50, 0.55]

    weights = {'goal': 0.2, 'draw': 0.2, 'away': 0.2, 'zero': 0.2}

    # Prepare logging
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, 'data', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'tuning_runs.csv')
    write_header = not os.path.exists(log_path)

    def log_trial(trial: TrialResult, strategy: str, threshold: float) -> None:
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header and f.tell() == 0:
                writer.writerow([
                    'ml_weight','prob_blend_alpha','min_lambda','goal_temperature','strategy','confidence_threshold',
                    'avg_points','objective','zero_zero_rate','pred_H','pred_D','pred_A','home_mean_pred','away_mean_pred','home_mean_act','away_mean_act'
                ])
            writer.writerow([
                trial.params['ml_weight'], trial.params['prob_blend_alpha'], trial.params['min_lambda'], trial.params['goal_temperature'],
                strategy, threshold, trial.avg_points, trial.obj_score, trial.zero_zero_rate, trial.pred_H, trial.pred_D, trial.pred_A,
                trial.home_mean_pred, trial.away_mean_pred, trial.home_mean_act, trial.away_mean_act
            ])

    # Build full grid and optionally sample down to a budget
    full_grid = list(product(ml_weights, prob_alphas, min_lambdas, goal_temps, strategies, thresholds))
    if max_trials > 0 and len(full_grid) > max_trials:
        random.seed(42)
        grid = random.sample(full_grid, max_trials)
    else:
        grid = full_grid

    # Coarse pass over the (possibly sampled) grid
    candidates: List[Tuple[TrialResult, str, float]] = []
    # Progress accounting: number of trials and fold-trainings
    total_trials = len(grid)
    try:
        n_splits = tscv.get_n_splits(features_df)
    except Exception:
        n_splits = 3
    total_trainings = total_trials * n_splits
    trials_done = 0
    trainings_done = 0
    for w, a, m, t, strategy, thr in grid:
        params = {
            'ml_weight': w,
            'prob_blend_alpha': a,
            'min_lambda': m,
            'goal_temperature': t,
        }
        trial = evaluate_params(params, features_df, tscv, strategy, thr, weights)
        log_trial(trial, strategy, thr)
        candidates.append((trial, strategy, thr))
        # Update progress
        trials_done += 1
        trainings_done += n_splits
        remaining_trials = total_trials - trials_done
        remaining_trainings = max(0, total_trainings - trainings_done)
        print(f"Progress: {trials_done}/{total_trials} trials done, {remaining_trials} left; ~{remaining_trainings} fold-trainings remaining")

    # Keep top 50% by objective
    candidates.sort(key=lambda x: x[0].obj_score, reverse=True)
    top = candidates[: max(1, len(candidates)//2)]

    # Zoom-in ranges around top configs
    def expand_range(values: List[float], center: float) -> List[float]:
        # Quick mode: no expansion
        return [center]

    refined_candidates: List[Tuple[TrialResult, str, float]] = []
    best_obj = -1e9
    for trial, strategy, thr in top[:0]:  # skip refinement entirely in quick mode
        w_vals = expand_range(ml_weights, float(trial.params['ml_weight']))
        a_vals = expand_range(prob_alphas, float(trial.params['prob_blend_alpha']))
        m_vals = expand_range(min_lambdas, float(trial.params['min_lambda']))
        t_vals = expand_range(goal_temps, float(trial.params['goal_temperature']))
        th_vals = expand_range(thresholds, float(thr))

        for w in w_vals:
            for a in a_vals:
                for m in m_vals:
                    for t in t_vals:
                        for th in th_vals:
                            params = {
                                'ml_weight': float(np.clip(w, 0.0, 1.0)),
                                'prob_blend_alpha': float(np.clip(a, 0.0, 1.0)),
                                'min_lambda': float(max(0.0, m)),
                                'goal_temperature': float(max(0.5, t)),
                            }
                            tr = evaluate_params(params, features_df, tscv, strategy, th, weights)
                            log_trial(tr, strategy, th)
                            refined_candidates.append((tr, strategy, th))
                            best_obj = max(best_obj, tr.obj_score)

    all_candidates = top + refined_candidates
    all_candidates.sort(key=lambda x: x[0].obj_score, reverse=True)
    best_trial, best_strategy, best_thr = all_candidates[0]

    # Save to config
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_dir = os.path.join(project_root, 'config')
    os.makedirs(cfg_dir, exist_ok=True)
    best_params = dict(best_trial.params)
    best_params['confidence_threshold'] = float(best_thr)
    best_params['strategy'] = best_strategy
    if yaml is not None:
        with open(os.path.join(cfg_dir, 'best_params.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(best_params, f, sort_keys=True)
    else:
        import json
        with open(os.path.join(cfg_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)

    # Print leaderboard
    print("\nTOP 10 CONFIGS BY OBJECTIVE:")
    for i, (tr, strat, thr) in enumerate(all_candidates[:10], start=1):
        print(f"{i:2d}. obj={tr.obj_score:.3f} pts={tr.avg_points:.3f} ml={tr.params['ml_weight']:.2f} a={tr.params['prob_blend_alpha']:.2f} minL={tr.params['min_lambda']:.2f} temp={tr.params['goal_temperature']:.2f} strat={strat} thr={thr:.2f} 0-0={tr.zero_zero_rate*100:.1f}% D={tr.pred_D*100:.1f}% A={tr.pred_A*100:.1f}%")

    print("\nBest params saved to config/best_params.yaml (or .json)")
    return best_params


def main():
    print("="*80)
    print("AUTOMATED HYPERPARAMETER TUNING")
    print("="*80)

    # CLI args for budget and CV folds
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-trials', type=int, default=0, help='Max parameter combos to evaluate (0 = full grid)')
    parser.add_argument('--n-splits', type=int, default=3, help='TimeSeriesSplit folds')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    current_season = data_fetcher.get_current_season()
    start_season = current_season - 2
    all_matches = data_fetcher.fetch_historical_seasons(start_season, current_season)
    print(f"Loaded {len(all_matches)} matches")

    print("Creating features...")
    features_df = feature_engineer.create_features_from_matches(all_matches)
    print(f"Created {len(features_df)} samples")

    best = successive_halving(features_df, n_splits=args.n_splits, max_trials=args.max_trials)
    print("\nBest configuration:")
    for k, v in best.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


