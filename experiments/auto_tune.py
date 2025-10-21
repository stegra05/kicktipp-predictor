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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import time
from contextlib import redirect_stdout, redirect_stderr
import io
import multiprocessing as mp

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Ensure project root is on sys.path so `src.*` imports work when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor

# NOTE: This tuning script was designed for the legacy HybridPredictor architecture.
# The v2 MatchPredictor uses a config-based approach instead of direct parameter setting.
# This script needs to be redesigned to work with the config system in config.py

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
                        act_D: float,
                        act_A: float,
                        weights: Dict[str, float]) -> float:
    # Goal mean penalty beyond ±10%
    home_pen = max(0.0, abs(home_pred - home_act) / max(home_act, 1e-6) - 0.10)
    away_pen = max(0.0, abs(away_pred - away_act) / max(away_act, 1e-6) - 0.10)
    goal_pen = 0.5 * home_pen + 0.5 * away_pen

    # Outcome distribution penalties
    draw_pen = max(0.0, pred_D - 0.30)
    away_penalty = max(0.0, 0.25 - pred_A)
    zero_pen = max(0.0, zero_zero_rate - 0.15)

    drift_pen = (weights.get('draw_drift', 0.0) * abs(pred_D - act_D) +
                 weights.get('away_drift', 0.0) * abs(pred_A - act_A))
    penalty = (weights['goal'] * goal_pen +
               weights['draw'] * draw_pen +
               weights['away'] * away_penalty +
               weights['zero'] * zero_pen +
               drift_pen)

    return avg_points - penalty


def evaluate_params(params: Dict[str, float],
                    features_df,
                    tscv: TimeSeriesSplit,
                    objective: str,
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

        # NOTE: v2 MatchPredictor doesn't support direct parameter tuning like this.
        # Parameters are now managed through the config system.
        # This code needs to be redesigned to modify config values.
        predictor = MatchPredictor()
        # TODO: Set config parameters here once config tuning is implemented
        # predictor.ml_weight = float(params['ml_weight'])
        # predictor.poisson_weight = 1.0 - predictor.ml_weight
        # predictor.prob_blend_alpha = float(params['prob_blend_alpha'])
        # predictor.min_lambda = float(params['min_lambda'])
        # predictor.goal_temperature = float(params['goal_temperature'])
        # predictor.confidence_threshold = float(confidence_threshold)

        predictor.train(train_df)
        preds = predictor.predict(test_feats)

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

        # Actual outcome shares for drift penalties
        def outc(h, a):
            return 'H' if h > a else ('D' if h == a else 'A')
        act_outcomes = [outc(int(a['home_score']), int(a['away_score'])) for a in acts]
        act_H = act_outcomes.count('H') / max(len(act_outcomes), 1)
        act_D = act_outcomes.count('D') / max(len(act_outcomes), 1)
        act_A = act_outcomes.count('A') / max(len(act_outcomes), 1)

        obj = pts if objective == 'points' else composite_objective(pts, z0, pH, pD, pA, hpred, apred, hact, aact, act_D, act_A, weights)
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


def _evaluate_trial(args) -> Tuple[TrialResult, float]:
    """Process-pool friendly wrapper to evaluate a single trial.

    Args tuple: (params, features_df, n_splits, objective, threshold, weights, omp_threads, quiet_workers)
    """
    import os
    params, features_df, n_splits, objective, threshold, weights, omp_threads, quiet_workers = args
    # Limit per-process parallelism to avoid oversubscription
    if omp_threads is not None:
        os.environ.setdefault('OMP_NUM_THREADS', str(omp_threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(omp_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(omp_threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(omp_threads))

    tscv = TimeSeriesSplit(n_splits=n_splits)
    if quiet_workers:
        # Suppress noisy training prints inside worker
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            trial = evaluate_params(params, features_df, tscv, objective, threshold, weights)
    else:
        trial = evaluate_params(params, features_df, tscv, objective, threshold, weights)
    return trial, threshold


def successive_halving(
        features_df,
        n_splits: int,
        max_trials: int,
        progress_interval: int = 10,
        objective: str = 'points',
        n_jobs: int = 0,
        omp_threads: int | None = None,
        quiet_workers: bool = True,
        # refinement controls
        enable_refine: bool = False,
        refine_top_k: int = 8,
        refine_steps: int = 5,
        span_ml_weight: float = 0.03,
        span_prob_alpha: float = 0.05,
        span_min_lambda: float = 0.05,
        span_goal_temp: float = 0.10,
        span_conf_thr: float = 0.05,
        # final model save controls
        save_final_model: bool = False,
        seasons_back: int = 3,
    ) -> Dict[str, float]:
    data_loader = DataLoader()

    # TimeSeries CV (modest folds for stability vs. runtime)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Expanded parameter ranges
    ml_weights = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    prob_alphas = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    min_lambdas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    goal_temps = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    weights = {'goal': 0.2, 'draw': 0.2, 'away': 0.2, 'zero': 0.2, 'draw_drift': 0.2, 'away_drift': 0.2}

    # Prepare logging
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, 'data', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'tuning_runs.csv')
    write_header = not os.path.exists(log_path)

    def log_trial(trial: TrialResult, objective_type: str, threshold: float) -> None:
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header and f.tell() == 0:
                writer.writerow([
                    'ml_weight','prob_blend_alpha','min_lambda','goal_temperature','objective_type','confidence_threshold',
                    'avg_points','objective','zero_zero_rate','pred_H','pred_D','pred_A','home_mean_pred','away_mean_pred','home_mean_act','away_mean_act'
                ])
            writer.writerow([
                trial.params['ml_weight'], trial.params['prob_blend_alpha'], trial.params['min_lambda'], trial.params['goal_temperature'],
                objective_type, threshold, trial.avg_points, trial.obj_score, trial.zero_zero_rate, trial.pred_H, trial.pred_D, trial.pred_A,
                trial.home_mean_pred, trial.away_mean_pred, trial.home_mean_act, trial.away_mean_act
            ])

    # Build full grid and optionally sample down to a budget
    full_grid = list(product(ml_weights, prob_alphas, min_lambdas, goal_temps, thresholds))
    if max_trials > 0 and len(full_grid) > max_trials:
        random.seed(42)
        grid = random.sample(full_grid, max_trials)
    else:
        grid = full_grid

    # Coarse pass over the (possibly sampled) grid
    candidates: List[Tuple[TrialResult, float]] = []
    total_trials = len(grid)
    total_trainings = total_trials * n_splits
    try:
        n_splits = tscv.get_n_splits(features_df)
    except Exception:
        n_splits = 3
    trainings_done = 0
    best_so_far: TrialResult | None = None
    # Determine workers
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 2) // 2)

    # Build arg tuples
    task_args = []
    for (w, a, m, t, thr) in grid:
        params = {'ml_weight': w, 'prob_blend_alpha': a, 'min_lambda': m, 'goal_temperature': t}
        task_args.append((params, features_df, n_splits, objective, thr, weights, omp_threads, quiet_workers))

    # Execute in parallel
    start_time = time.time()
    def fmt_dur(sec: float) -> str:
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    # Also constrain threads in parent to avoid eager thread pools during imports
    if omp_threads is not None:
        os.environ.setdefault('OMP_NUM_THREADS', str(omp_threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(omp_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(omp_threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(omp_threads))

    print(f"[TUNE] workers={n_jobs} x threads={omp_threads or 1} | trials={total_trials} | cv-trainings={total_trainings}")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn')) as ex:
        futures = [ex.submit(_evaluate_trial, ta) for ta in task_args]
        for idx, fut in enumerate(as_completed(futures), start=1):
            trial, thr = fut.result()
            log_trial(trial, objective, thr)
            candidates.append((trial, thr))
            trainings_done += n_splits
            if best_so_far is None or trial.obj_score > best_so_far.obj_score:
                best_so_far = trial
            if (idx % max(1, progress_interval) == 0) or idx == total_trials:
                elapsed = time.time() - start_time
                # Simple ETA based on completed trials
                rate = elapsed / max(1, idx)
                remaining = rate * (total_trials - idx)
                pct = idx / total_trials
                bar_w = 28
                filled = int(pct * bar_w)
                bar = "#" * filled + "." * (bar_w - filled)
                line = (
                    f"\r[TUNE] [{bar}] {idx}/{total_trials} ({pct*100:5.1f}%) "
                    f"trainings {trainings_done}/{total_trainings} | elapsed {fmt_dur(elapsed)} | eta {fmt_dur(remaining)} | "
                    f"best obj={best_so_far.obj_score:.3f} pts={best_so_far.avg_points:.3f}"
                )
                print(line, end="", flush=True)
        # ensure newline after progress line
        print()

    # Keep top 50% by objective
    candidates.sort(key=lambda x: x[0].obj_score, reverse=True)
    top = candidates[: max(1, len(candidates)//2)]

    # Zoom-in refinement around top configs (optional)
    refined_candidates: List[Tuple[TrialResult, float]] = []
    if enable_refine and len(top) > 0 and refine_steps >= 1 and refine_top_k > 0:
        def expand_lin(center: float, span: float, steps: int, lo: float, hi: float) -> List[float]:
            if steps <= 1 or span <= 0:
                return [float(np.clip(center, lo, hi))]
            lo_b = max(lo, center - span)
            hi_b = min(hi, center + span)
            if hi_b < lo_b:
                lo_b, hi_b = hi_b, lo_b
            arr = np.linspace(lo_b, hi_b, num=steps)
            return [float(np.clip(x, lo, hi)) for x in arr]

        top_k = top[:min(refine_top_k, len(top))]

        # Build all refinement tasks
        ref_task_args: List[Tuple[Dict[str, float], Any, int, str, float, Dict[str, float], int | None, bool]] = []
        # Estimate total number of refined trials for progress reporting
        total_ref_trials = 0
        for tr, thr in top_k:
            w_vals = expand_lin(float(tr.params['ml_weight']), span_ml_weight, refine_steps, 0.0, 1.0)
            a_vals = expand_lin(float(tr.params['prob_blend_alpha']), span_prob_alpha, refine_steps, 0.0, 1.0)
            m_vals = expand_lin(float(tr.params['min_lambda']), span_min_lambda, refine_steps, 0.0, 1.0)
            t_vals = expand_lin(float(tr.params['goal_temperature']), span_goal_temp, refine_steps, 0.5, 2.0)
            th_vals = expand_lin(float(thr), span_conf_thr, refine_steps, 0.0, 1.0)
            total_ref_trials += max(1, len(w_vals)) * max(1, len(a_vals)) * max(1, len(m_vals)) * max(1, len(t_vals)) * max(1, len(th_vals))
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
                                ref_task_args.append((params, features_df, n_splits, objective, float(th), weights, omp_threads, quiet_workers))

        # Execute refinement in parallel with progress bar
        print(f"[REFN] workers={n_jobs} x threads={omp_threads or 1} | trials={total_ref_trials} | cv-trainings={total_ref_trials * n_splits}")
        ref_start = time.time()
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn')) as ex:
            futures = [ex.submit(_evaluate_trial, ta) for ta in ref_task_args]
            for ridx, fut in enumerate(as_completed(futures), start=1):
                rtrial, rthr = fut.result()
                log_trial(rtrial, objective, rthr)
                refined_candidates.append((rtrial, rthr))
                if (ridx % max(1, progress_interval) == 0) or ridx == total_ref_trials:
                    elapsed = time.time() - ref_start
                    rate = elapsed / max(1, ridx)
                    remaining = rate * (total_ref_trials - ridx)
                    pct = ridx / total_ref_trials
                    bar_w = 28
                    filled = int(pct * bar_w)
                    bar = "#" * filled + "." * (bar_w - filled)
                    line = (
                        f"\r[REFN] [{bar}] {ridx}/{total_ref_trials} ({pct*100:5.1f}%) "
                        f"elapsed {fmt_dur(elapsed)} | eta {fmt_dur(remaining)}"
                    )
                    print(line, end="", flush=True)
            print()

    all_candidates = top + refined_candidates
    all_candidates.sort(key=lambda x: x[0].obj_score, reverse=True)
    best_trial, best_thr = all_candidates[0]

    # Save to config
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_dir = os.path.join(project_root, 'config')
    os.makedirs(cfg_dir, exist_ok=True)
    best_params = dict(best_trial.params)
    best_params['confidence_threshold'] = float(best_thr)
    best_params['strategy'] = 'optimized'
    if yaml is not None:
        with open(os.path.join(cfg_dir, 'best_params.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(best_params, f, sort_keys=True)
    else:
        import json
        with open(os.path.join(cfg_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)

    # Print leaderboard
    print("\nTOP 10 CONFIGS BY OBJECTIVE:")
    for i, (tr, thr) in enumerate(all_candidates[:10], start=1):
        print(f"{i:2d}. obj={tr.obj_score:.3f} pts={tr.avg_points:.3f} ml={tr.params['ml_weight']:.2f} a={tr.params['prob_blend_alpha']:.2f} minL={tr.params['min_lambda']:.2f} temp={tr.params['goal_temperature']:.2f} thr={thr:.2f} 0-0={tr.zero_zero_rate*100:.1f}% D={tr.pred_D*100:.1f}% A={tr.pred_A*100:.1f}%")

    print("\nBest params saved to config/best_params.yaml (or .json)")

    # Optional: Train final model on full dataset and save
    if save_final_model:
        try:
            print("\n[FINAL] Training final MatchPredictor on full dataset...")
            predictor = MatchPredictor()
            # TODO: Apply best parameters through config system
            # predictor.ml_weight = float(best_trial.params['ml_weight'])
            # predictor.poisson_weight = 1.0 - predictor.ml_weight
            # predictor.prob_blend_alpha = float(best_trial.params['prob_blend_alpha'])
            # predictor.min_lambda = float(best_trial.params['min_lambda'])
            # predictor.goal_temperature = float(best_trial.params['goal_temperature'])
            # predictor.confidence_threshold = float(best_thr)

            # Rebuild dataset with configurable horizon
            current_season = data_loader.get_current_season()
            start_season = current_season - int(max(1, seasons_back))
            all_matches_full = data_loader.fetch_historical_seasons(start_season, current_season)
            features_full = data_loader.create_features_from_matches(all_matches_full)

            predictor.train(features_full)
            predictor.save_models()

            # Persist run meta
            out_dir = os.path.join(project_root, 'data', 'predictions')
            os.makedirs(out_dir, exist_ok=True)
            meta_path = os.path.join(out_dir, 'run_meta.json')
            from datetime import datetime as _dt
            meta = {
                'script': 'experiments/auto_tune.py',
                'timestamp': _dt.now().isoformat(),
                'ml_weight': predictor.ml_weight,
                'poisson_weight': predictor.poisson_weight,
                'prob_blend_alpha': predictor.prob_blend_alpha,
                'min_lambda': predictor.min_lambda,
                'goal_temperature': predictor.goal_temperature,
                'confidence_threshold': predictor.confidence_threshold,
                'max_goals': getattr(predictor, 'max_goals', 8),
                'strategy': getattr(predictor, 'strategy', 'optimized'),
            }
            try:
                import json as _json
                with open(meta_path, 'w', encoding='utf-8') as f:
                    _json.dump(meta, f, indent=2, default=str)
                print(f"[FINAL] Saved trained models and run meta: {meta_path}")
            except Exception:
                pass
        except Exception as e:
            print(f"[FINAL] WARNING: Failed to train/save final model: {e}")
    return best_params


def main():
    print("="*80)
    print("AUTOMATED HYPERPARAMETER TUNING")
    print("="*80)

    # CLI args for budget and CV folds
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-trials', type=int, default=0, help='Max parameter combos to evaluate (0 = full grid)')
    parser.add_argument('--n-splits', type=int, default=3, help='TimeSeriesSplit folds')
    parser.add_argument('--progress-interval', type=int, default=10, help='Trials per progress print')
    parser.add_argument('--objective', type=str, default='points', choices=['points','composite'], help='Optimization objective')
    parser.add_argument('--n-jobs', type=int, default=0, help='Parallel processes (0 = half CPUs)')
    parser.add_argument('--omp-threads', type=int, default=1, help='Threads per process for BLAS/OMP')
    # Refinement controls
    parser.add_argument('--refine', action='store_true', help='Enable zoom-in refinement around top configs')
    parser.add_argument('--refine-top-k', type=int, default=8, help='Top-K coarse configs to refine')
    parser.add_argument('--refine-steps', type=int, default=5, help='Points per parameter during local sweep')
    parser.add_argument('--span-ml-weight', type=float, default=0.03, help='Local span for ml_weight')
    parser.add_argument('--span-prob-alpha', type=float, default=0.05, help='Local span for prob_blend_alpha')
    parser.add_argument('--span-min-lambda', type=float, default=0.05, help='Local span for min_lambda')
    parser.add_argument('--span-goal-temp', type=float, default=0.10, help='Local span for goal_temperature')
    parser.add_argument('--span-conf-thr', type=float, default=0.05, help='Local span for confidence_threshold')
    # Final model save controls
    parser.add_argument('--save-final-model', action='store_true', help='Train on full dataset with best params and save models')
    parser.add_argument('--seasons-back', type=int, default=3, help='Number of past seasons to include for final training')
    # Optional Optuna scaffold
    parser.add_argument('--optuna', type=int, default=0, help='Run Optuna with N trials instead of grid (requires optuna)')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    current_season = data_loader.get_current_season()
    start_season = current_season - 2
    all_matches = data_loader.fetch_historical_seasons(start_season, current_season)
    print(f"Loaded {len(all_matches)} matches")

    print("Creating features...")
    features_df = data_loader.create_features_from_matches(all_matches)
    print(f"Created {len(features_df)} samples")

    # Optional Optuna branch (scaffold)
    if args.optuna and args.optuna > 0:
        try:
            import optuna  # type: ignore
        except Exception:
            print("Optuna is not installed. Please install optuna to use --optuna.")
            sys.exit(1)

        def _optuna_objective(trial):
            params = {
                'ml_weight': trial.suggest_float('ml_weight', 0.5, 0.9, step=0.01),
                'prob_blend_alpha': trial.suggest_float('prob_blend_alpha', 0.2, 0.7, step=0.01),
                'min_lambda': trial.suggest_float('min_lambda', 0.02, 0.4, step=0.01),
                'goal_temperature': trial.suggest_float('goal_temperature', 1.1, 1.9, step=0.01),
            }
            threshold = trial.suggest_float('confidence_threshold', 0.30, 0.70, step=0.01)
            weights = {'goal': 0.2, 'draw': 0.2, 'away': 0.2, 'zero': 0.2, 'draw_drift': 0.2, 'away_drift': 0.2}
            tscv = TimeSeriesSplit(n_splits=args.n_splits)
            tr = evaluate_params(params, features_df, tscv, args.objective, threshold, weights)
            return tr.obj_score

        print(f"Running Optuna study for {args.optuna} trials...")
        study = optuna.create_study(direction='maximize')
        study.optimize(_optuna_objective, n_trials=args.optuna, show_progress_bar=True)
        best_params = dict(study.best_params)
        # Persist best params
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cfg_dir = os.path.join(project_root, 'config')
        os.makedirs(cfg_dir, exist_ok=True)
        if yaml is not None:
            with open(os.path.join(cfg_dir, 'best_params.yaml'), 'w', encoding='utf-8') as f:
                yaml.safe_dump({
                    'ml_weight': float(best_params['ml_weight']),
                    'prob_blend_alpha': float(best_params['prob_blend_alpha']),
                    'min_lambda': float(best_params['min_lambda']),
                    'goal_temperature': float(best_params['goal_temperature']),
                    'confidence_threshold': float(best_params.get('confidence_threshold', 0.5)),
                    'strategy': 'optimized',
                }, f, sort_keys=True)
        else:
            import json
            with open(os.path.join(cfg_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'ml_weight': float(best_params['ml_weight']),
                    'prob_blend_alpha': float(best_params['prob_blend_alpha']),
                    'min_lambda': float(best_params['min_lambda']),
                    'goal_temperature': float(best_params['goal_temperature']),
                    'confidence_threshold': float(best_params.get('confidence_threshold', 0.5)),
                    'strategy': 'optimized',
                }, f, indent=2)
        print("Optuna tuning complete; best params saved. Exiting.")
        return

    best = successive_halving(
        features_df,
        n_splits=args.n_splits,
        max_trials=args.max_trials,
        progress_interval=args.progress_interval,
        objective=args.objective,
        n_jobs=args.n_jobs,
        omp_threads=args.omp_threads,
        quiet_workers=True,
        enable_refine=args.refine,
        refine_top_k=args.refine_top_k,
        refine_steps=args.refine_steps,
        span_ml_weight=args.span_ml_weight,
        span_prob_alpha=args.span_prob_alpha,
        span_min_lambda=args.span_min_lambda,
        span_goal_temp=args.span_goal_temp,
        span_conf_thr=args.span_conf_thr,
        save_final_model=args.save_final_model,
        seasons_back=args.seasons_back,
    )
    print("\nBest configuration:")
    for k, v in best.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


