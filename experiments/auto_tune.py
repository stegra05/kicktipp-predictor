#!/usr/bin/env python3
"""
Optuna-only hyperparameter tuning optimizing points-per-game (PPG).
Writes best params to config/best_params.yaml and optionally trains final model.
"""

import os
import sys
import argparse
from typing import Dict, List
import time

import numpy as np
import sys
from contextlib import redirect_stdout, redirect_stderr
from sklearn.model_selection import TimeSeriesSplit

# Ensure project root is on sys.path so `src.*` imports work when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor
from kicktipp_predictor.config import reset_config, get_config

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import optuna  # type: ignore
except Exception as _e:  # pragma: no cover
    optuna = None


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


def _apply_params_to_config(params: Dict[str, float]) -> None:
    """Mutate the global Config instance with trial parameters."""
    cfg = get_config()
    # Core
    if 'draw_boost' in params:
        cfg.model.draw_boost = float(params['draw_boost'])
    if 'min_lambda' in params:
        cfg.model.min_lambda = float(params['min_lambda'])

    # Outcome classifier
    if 'outcome_n_estimators' in params:
        cfg.model.outcome_n_estimators = int(params['outcome_n_estimators'])
    if 'outcome_max_depth' in params:
        cfg.model.outcome_max_depth = int(params['outcome_max_depth'])
    if 'outcome_learning_rate' in params:
        cfg.model.outcome_learning_rate = float(params['outcome_learning_rate'])
    if 'outcome_subsample' in params:
        cfg.model.outcome_subsample = float(params['outcome_subsample'])
    if 'outcome_colsample_bytree' in params:
        cfg.model.outcome_colsample_bytree = float(params['outcome_colsample_bytree'])

    # Goal regressors
    if 'goals_n_estimators' in params:
        cfg.model.goals_n_estimators = int(params['goals_n_estimators'])
    if 'goals_max_depth' in params:
        cfg.model.goals_max_depth = int(params['goals_max_depth'])
    if 'goals_learning_rate' in params:
        cfg.model.goals_learning_rate = float(params['goals_learning_rate'])
    if 'goals_subsample' in params:
        cfg.model.goals_subsample = float(params['goals_subsample'])
    if 'goals_colsample_bytree' in params:
        cfg.model.goals_colsample_bytree = float(params['goals_colsample_bytree'])


def _objective_builder(features_df, n_splits: int, omp_threads: int, verbose: bool):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
        # Limit BLAS/OMP threads to avoid oversubscription
        if omp_threads and omp_threads > 0:
            os.environ['OMP_NUM_THREADS'] = str(omp_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(omp_threads)
            os.environ['MKL_NUM_THREADS'] = str(omp_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(omp_threads)

        # Search space
        params: Dict[str, float] = {
            # Class weighting
            'draw_boost': trial.suggest_float('draw_boost', 1.0, 2.0, step=0.05),
            # Outcome XGB
            'outcome_n_estimators': trial.suggest_int('outcome_n_estimators', 100, 600, step=25),
            'outcome_max_depth': trial.suggest_int('outcome_max_depth', 3, 10),
            'outcome_learning_rate': trial.suggest_float('outcome_learning_rate', 0.02, 0.30, step=0.01),
            'outcome_subsample': trial.suggest_float('outcome_subsample', 0.5, 1.0, step=0.05),
            'outcome_colsample_bytree': trial.suggest_float('outcome_colsample_bytree', 0.5, 1.0, step=0.05),
            # Goals XGB
            'goals_n_estimators': trial.suggest_int('goals_n_estimators', 100, 600, step=25),
            'goals_max_depth': trial.suggest_int('goals_max_depth', 3, 10),
            'goals_learning_rate': trial.suggest_float('goals_learning_rate', 0.02, 0.30, step=0.01),
            'goals_subsample': trial.suggest_float('goals_subsample', 0.5, 1.0, step=0.05),
            'goals_colsample_bytree': trial.suggest_float('goals_colsample_bytree', 0.5, 1.0, step=0.05),
            # Scoreline selection floor
            'min_lambda': trial.suggest_float('min_lambda', 0.05, 0.40, step=0.01),
        }

        fold_points: List[float] = []

        for train_idx, test_idx in tscv.split(features_df):
            # Reset and apply params
            reset_config()
            _apply_params_to_config(params)

            # Prepare data
            train_df = features_df.iloc[train_idx]
            test_df = features_df.iloc[test_idx]
            test_feats = test_df.drop(columns=['home_score','away_score','goal_difference','result'], errors='ignore')

            if not verbose:
                # Suppress noisy prints from the inner training/prediction loop
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        predictor = MatchPredictor(quiet=not verbose)
                        predictor.train(train_df)
                        preds = predictor.predict(test_feats)
            else:
                predictor = MatchPredictor(quiet=not verbose)
                predictor.train(train_df)
                preds = predictor.predict(test_feats)

            acts = test_df.to_dict('records')
            pts = calculate_points(preds, acts) / max(len(acts), 1)
            fold_points.append(pts)

        # Objective: average points per game (maximize)
        return float(np.mean(fold_points)) if fold_points else 0.0

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials')
    parser.add_argument('--n-splits', type=int, default=3, help='TimeSeriesSplit folds')
    parser.add_argument('--n-jobs', type=int, default=0, help='Parallel Optuna workers (0=serial)')
    parser.add_argument('--omp-threads', type=int, default=1, help='Threads per worker for BLAS/OMP')
    parser.add_argument('--save-final-model', action='store_true', help='Train on full dataset with best params and save models')
    parser.add_argument('--seasons-back', type=int, default=3, help='Number of past seasons to include for final training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logs from inner training loop')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///study.db) for multi-process tuning')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name (used with --storage)')
    parser.add_argument('--pruner', type=str, choices=['none','median','hyperband'], default='median', help='Enable trial pruning strategy')
    parser.add_argument('--pruner-startup-trials', type=int, default=20, help='Trials before enabling pruning (median pruner)')
    args = parser.parse_args()

    # Propagate verbosity to submodules via environment variable
    os.environ['KTP_VERBOSE'] = '1' if args.verbose else '0'

    if optuna is None:
        print("Optuna is not installed. Please install optuna to run tuning.")
        sys.exit(1)

    # Limit threads in parent
    if args.omp_threads and args.omp_threads > 0:
        os.environ.setdefault('OMP_NUM_THREADS', str(args.omp_threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(args.omp_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(args.omp_threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(args.omp_threads))

    # Load data
    if args.verbose:
        print("Loading data...")
    data_loader = DataLoader()
    current_season = data_loader.get_current_season()
    start_season = current_season - 2
    all_matches = data_loader.fetch_historical_seasons(start_season, current_season)
    if args.verbose:
        print(f"Loaded {len(all_matches)} matches")

    if args.verbose:
        print("Creating features...")
    features_df = data_loader.create_features_from_matches(all_matches)
    if args.verbose:
        print(f"Created {len(features_df)} samples")

    # Build and run study
    def _fmt_dur(sec: float) -> str:
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    total_trials = int(max(0, args.n_trials))
    total_cv_trainings = total_trials * int(max(1, args.n_splits))
    if args.verbose:
        print(f"Planned: trials={total_trials} | cv-trainings={total_cv_trainings} | workers={args.n_jobs or 1} x threads={args.omp_threads or 1}")

    # Avoid stdout redirection when using parallel workers (not thread-safe)
    effective_verbose = bool(args.verbose or (args.n_jobs and args.n_jobs > 1))
    objective = _objective_builder(features_df, args.n_splits, args.omp_threads, effective_verbose)

    # Configure pruner
    pruner = None
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(max(0, args.pruner_startup_trials)))
    elif args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner()

    # Create study, optionally with storage for multi-process scaling
    if args.storage:
        study = optuna.create_study(
            direction='maximize',
            storage=args.storage,
            study_name=(args.study_name or 'kicktipp-tune'),
            load_if_exists=True,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(direction='maximize', pruner=pruner)

    progress = {
        'start': time.time(),
        'completed': 0,
        'best_value': float('-inf'),
        'best_trial': None,
        'last_print_len': 0,
        'last_update_ts': 0.0,
    }

    def _progress_cb(study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:  # type: ignore[name-defined]
        progress['completed'] += 1
        if study.best_trial is not None:
            progress['best_value'] = float(study.best_value)
            progress['best_trial'] = int(study.best_trial.number)
        elapsed = time.time() - progress['start']
        done = progress['completed']
        total = max(1, total_trials)
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = max(0.0, (total - done) / rate) if rate > 0 else float('inf')
        # Build status line
        rate_str = f"{rate:.2f} t/s" if rate > 0 else "-- t/s"
        pct = (done / total) * 100 if total > 0 else 0.0
        bar_len = 24
        filled = int((pct / 100) * bar_len)
        bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
        line = (
            f"\r[TUNE] {bar} {done}/{total} ({pct:5.1f}%) | elapsed {_fmt_dur(elapsed)} | "
            f"eta {_fmt_dur(remaining)} | speed {rate_str} | best={progress['best_value']:.3f}"
        )
        if progress['best_trial'] is not None:
            line += f" (#{progress['best_trial']})"
        # Throttle updates to avoid flooding when parallel logs are noisy
        now_ts = time.time()
        should_update = (now_ts - progress['last_update_ts'] >= 0.5) or (done == total)
        if not should_update:
            return
        progress['last_update_ts'] = now_ts

        # Clear previous line if shorter, write to stderr to separate from stdout logs
        pad = max(0, progress['last_print_len'] - len(line))
        if sys.stderr and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            print(line + (' ' * pad), end='', flush=True, file=sys.stderr)
        else:
            # Non-TTY: print as a normal log line
            print(line.replace('\r', ''), flush=True, file=sys.stderr)
        progress['last_print_len'] = len(line)

    if args.verbose:
        print(f"Running Optuna study for {args.n_trials} trials with n_jobs={args.n_jobs}...")
    start = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs if args.n_jobs and args.n_jobs > 0 else 1,
        callbacks=[_progress_cb],
        show_progress_bar=False,
    )
    # Ensure newline after progress line
    print(file=sys.stderr)
    duration = time.time() - start
    if args.verbose:
        print(f"Study complete in {duration:.1f}s. Best PPG={study.best_value:.4f}")

    # Persist best params
    best_params = dict(study.best_params)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_dir = os.path.join(project_root, 'config')
    os.makedirs(cfg_dir, exist_ok=True)
    if yaml is not None:
        with open(os.path.join(cfg_dir, 'best_params.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(best_params, f, sort_keys=True)
    else:
        import json
        with open(os.path.join(cfg_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)
    if args.verbose:
        print("Best params saved to config/best_params.yaml (or .json)")

    # Optional: Train final model on full dataset and save
    if args.save_final_model:
        try:
            if args.verbose:
                print("\n[FINAL] Training final MatchPredictor on full dataset with best params...")
            reset_config()
            _apply_params_to_config(best_params)

            current_season = data_loader.get_current_season()
            start_season = current_season - int(max(1, args.seasons_back))
            all_matches_full = data_loader.fetch_historical_seasons(start_season, current_season)
            features_full = data_loader.create_features_from_matches(all_matches_full)

            if not args.verbose:
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        predictor = MatchPredictor(quiet=not args.verbose)
                        predictor.train(features_full)
                        predictor.save_models()
            else:
                predictor = MatchPredictor(quiet=not args.verbose)
                predictor.train(features_full)
                predictor.save_models()
            if args.verbose:
                print("[FINAL] Saved trained models.")
        except Exception as e:  # pragma: no cover
            if args.verbose:
                print(f"[FINAL] WARNING: Failed to train/save final model: {e}")


if __name__ == "__main__":
    main()
