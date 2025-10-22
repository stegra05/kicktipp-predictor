#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning with selectable objectives (PPG, logloss, brier, etc.).
Saves per-objective best params (config/best_params_<objective>.yaml) and the winner to config/best_params.yaml.
Removes the Optuna SQLite storage database after run to start fresh each time.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import time

# Hard-cap BLAS/OpenMP threads before importing numpy/xgboost to avoid fork/thread storms
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('XGBOOST_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Ensure project root is on sys.path so `src.*` imports work when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor
from kicktipp_predictor.config import reset_config, get_config
from kicktipp_predictor.evaluate import compute_points, log_loss_multiclass, brier_score_multiclass, ranked_probability_score
from urllib.parse import urlparse, unquote

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import optuna  # type: ignore
except Exception as _e:  # pragma: no cover
    optuna = None


def _objective_direction(objective: str) -> str:
    obj = (objective or "").lower()
    if obj in ("logloss", "brier", "rps"):
        return "minimize"
    return "maximize"


def _sqlite_fs_path(storage: Optional[str]) -> Optional[str]:
    if not storage:
        return None
    s = storage.strip()
    if s in ("sqlite:///:memory:", ":memory:"):
        return None
    parsed = urlparse(s)
    if parsed.scheme != "sqlite":
        return None
    path = unquote(parsed.path or "")
    if not path:
        return None
    # Handle Windows drive letter like /C:/path.db
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


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
    if 'time_decay_half_life_days' in params:
        cfg.model.time_decay_half_life_days = float(params['time_decay_half_life_days'])
        cfg.model.use_time_decay = True
    # Optional feature knobs (recognized if present)
    if 'form_last_n' in params:
        try:
            cfg.model.form_last_n = int(params['form_last_n'])
        except Exception:
            cfg.model.form_last_n = int(float(params['form_last_n']))
    if 'momentum_decay' in params:
        cfg.model.momentum_decay = float(params['momentum_decay'])

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
    if 'outcome_reg_alpha' in params:
        cfg.model.outcome_reg_alpha = float(params['outcome_reg_alpha'])
    if 'outcome_reg_lambda' in params:
        cfg.model.outcome_reg_lambda = float(params['outcome_reg_lambda'])
    if 'outcome_gamma' in params:
        cfg.model.outcome_gamma = float(params['outcome_gamma'])
    if 'outcome_min_child_weight' in params:
        cfg.model.outcome_min_child_weight = float(params['outcome_min_child_weight'])

    # Post-processing probabilities
    if 'proba_temperature' in params:
        cfg.model.proba_temperature = float(params['proba_temperature'])
    if 'prior_blend_alpha' in params:
        cfg.model.prior_blend_alpha = float(params['prior_blend_alpha'])

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
    if 'goals_reg_alpha' in params:
        cfg.model.goals_reg_alpha = float(params['goals_reg_alpha'])
    if 'goals_reg_lambda' in params:
        cfg.model.goals_reg_lambda = float(params['goals_reg_lambda'])
    if 'goals_gamma' in params:
        cfg.model.goals_gamma = float(params['goals_gamma'])
    if 'goals_min_child_weight' in params:
        cfg.model.goals_min_child_weight = float(params['goals_min_child_weight'])


def _objective_builder(base_features_df, all_matches, folds: List[Tuple[np.ndarray, np.ndarray]], objective_name: str, direction: str, omp_threads: int, verbose: bool):
    # Cache features by feature-knob tuple to avoid recomputation across trials
    features_cache: Dict[tuple, any] = {}

    def obj_fn(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
        # Limit BLAS/OMP threads to avoid oversubscription
        if omp_threads and omp_threads > 0:
            os.environ['OMP_NUM_THREADS'] = str(omp_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(omp_threads)
            os.environ['MKL_NUM_THREADS'] = str(omp_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(omp_threads)

        # Search space
        params: Dict[str, float] = {
            # Class weighting - Widen range slightly
            'draw_boost': trial.suggest_float('draw_boost', 1.2, 2.5, step=0.1), # Default 1.5

            # Outcome XGB - Reduce upper bounds for regularization
            'outcome_n_estimators': trial.suggest_int('outcome_n_estimators', 100, 800, step=50), # Default 800
            'outcome_max_depth': trial.suggest_int('outcome_max_depth', 3, 8), # Default 6
            'outcome_learning_rate': trial.suggest_float('outcome_learning_rate', 0.01, 0.20, step=0.01), # Default 0.1
            'outcome_subsample': trial.suggest_float('outcome_subsample', 0.6, 1.0, step=0.05), # Default 0.8
            'outcome_colsample_bytree': trial.suggest_float('outcome_colsample_bytree', 0.6, 1.0, step=0.05), # Default 0.8
            'outcome_reg_alpha': trial.suggest_float('outcome_reg_alpha', 0.0, 0.5, step=0.05), # Was 0-1, Default 0.0
            'outcome_reg_lambda': trial.suggest_float('outcome_reg_lambda', 0.5, 2.0, step=0.05), # Was 0.5-3, Default 1.0
            'outcome_gamma': trial.suggest_float('outcome_gamma', 0.0, 2.0, step=0.1), # Was 0-5, Default 0.0
            'outcome_min_child_weight': trial.suggest_float('outcome_min_child_weight', 1.0, 7.0, step=0.5), # Was 1-10, Default 1.0

            # Goals XGB - Keep similar, maybe slightly less regularization too
            'goals_n_estimators': trial.suggest_int('goals_n_estimators', 100, 800, step=50), # Default 800
            'goals_max_depth': trial.suggest_int('goals_max_depth', 3, 9), # Default 6
            'goals_learning_rate': trial.suggest_float('goals_learning_rate', 0.01, 0.20, step=0.01), # Default 0.1
            'goals_subsample': trial.suggest_float('goals_subsample', 0.6, 1.0, step=0.05), # Default 0.8
            'goals_colsample_bytree': trial.suggest_float('goals_colsample_bytree', 0.6, 1.0, step=0.05), # Default 0.8
            'goals_reg_alpha': trial.suggest_float('goals_reg_alpha', 0.0, 0.5, step=0.05), # Was 0-1, Default 0.0
            'goals_reg_lambda': trial.suggest_float('goals_reg_lambda', 0.5, 2.0, step=0.05), # Was 0.5-3, Default 1.0
            'goals_gamma': trial.suggest_float('goals_gamma', 0.0, 2.0, step=0.1), # Was 0-5, Default 0.0
            'goals_min_child_weight': trial.suggest_float('goals_min_child_weight', 1.0, 7.0, step=0.5), # Was 1-10, Default 1.0

            # Scoreline selection floor
            'min_lambda': trial.suggest_float('min_lambda', 0.10, 0.35, step=0.01), # Was 0.05-0.40, Default 0.2

            # Time-decay half-life - Keep broad
            'time_decay_half_life_days': trial.suggest_float('time_decay_half_life_days', 45.0, 360.0, step=15.0), # Default 90

            # Outcome proba post-processing - Limit prior blending, narrow temp
            'proba_temperature': trial.suggest_float('proba_temperature', 0.85, 1.15, step=0.05), # Was 0.7-1.3, Default 1.0
            'prior_blend_alpha': trial.suggest_float('prior_blend_alpha', 0.0, 0.15, step=0.02), # Was 0-0.3, Default 0.0

            # Feature-engineering knobs (optional) - Keep as is
            'form_last_n': trial.suggest_int('form_last_n', 3, 10, step=1), # Default 5
            'momentum_decay': trial.suggest_float('momentum_decay', 0.70, 0.99, step=0.01), # Default 0.9
        }

        fold_metrics: List[float] = []

        # Determine which feature set to use for this trial
        # Apply params to config now so DataLoader sees knobs
        reset_config()
        _apply_params_to_config(params)

        # Cache key by (form_last_n, momentum_decay)
        cfg = get_config()
        key = (int(getattr(cfg.model, 'form_last_n', 5)), float(round(getattr(cfg.model, 'momentum_decay', 0.9), 3)))

        if key not in features_cache:
            # Recompute features for this knob combo
            dl = DataLoader()
            if verbose:
                print(f"[FEATS] Building features for knobs form_last_n={key[0]} momentum_decay={key[1]}...")
            feats_df = dl.create_features_from_matches(all_matches)
            features_cache[key] = feats_df
        features_df = features_cache[key]

        for train_idx, test_idx in folds:
            # Reset and apply params
            reset_config()
            _apply_params_to_config(params)

            # Prepare data
            train_df = features_df.iloc[train_idx]
            test_df = features_df.iloc[test_idx]
            test_feats = test_df.drop(columns=['home_score','away_score','goal_difference','result'], errors='ignore')

            predictor = MatchPredictor(quiet=not verbose)
            predictor.train(train_df)
            preds = predictor.predict(test_feats)

            # Build per-match points vector
            ph = np.array([int(p.get('predicted_home_score', 0)) for p in preds], dtype=int)
            pa = np.array([int(p.get('predicted_away_score', 0)) for p in preds], dtype=int)
            ah = np.asarray(test_df['home_score'], dtype=int)
            aa = np.asarray(test_df['away_score'], dtype=int)
            points_vec = compute_points(ph, pa, ah, aa).astype(float)

            # Recency weights for validation fold
            if 'date' in test_df.columns:
                fold_dates = pd.to_datetime(test_df['date'])
                days_old = (fold_dates.max() - fold_dates).dt.days.astype(float)
                half_life = float(params['time_decay_half_life_days'])
                decay_rate = np.log(2.0) / max(1.0, half_life)
                weights = np.exp(-decay_rate * days_old.values)
            else:
                weights = np.ones_like(points_vec, dtype=float)

            # Build proba matrix and labels
            proba = np.array([
                [p.get('home_win_probability', 0.0), p.get('draw_probability', 0.0), p.get('away_win_probability', 0.0)]
                for p in preds
            ], dtype=float)
            y_true = test_df['result'].tolist()
            y_pred_idx = np.argmax(proba, axis=1) if len(proba) else np.array([])
            idx_to_label = {0: 'H', 1: 'D', 2: 'A'}
            y_pred = [idx_to_label.get(int(i), 'H') for i in y_pred_idx]

            obj = (objective_name or 'ppg').lower()
            if obj == 'ppg':
                metric_val = float(np.sum(points_vec * weights) / max(1.0, np.sum(weights)))
            elif obj == 'ppg_unweighted':
                metric_val = float(np.mean(points_vec)) if len(points_vec) else 0.0
            elif obj == 'logloss':
                metric_val = log_loss_multiclass(y_true, proba)
            elif obj == 'brier':
                metric_val = brier_score_multiclass(y_true, proba)
            elif obj == 'rps':
                metric_val = ranked_probability_score(y_true, proba)
            elif obj == 'balanced_accuracy':
                try:
                    metric_val = float(balanced_accuracy_score(y_true, y_pred, sample_weight=weights))
                except Exception:
                    metric_val = float('nan')
            elif obj == 'accuracy':
                try:
                    metric_val = float(accuracy_score(y_true, y_pred, sample_weight=weights))
                except Exception:
                    metric_val = float('nan')
            else:
                metric_val = float(np.sum(points_vec * weights) / max(1.0, np.sum(weights)))

            if verbose:
                if obj in ('ppg', 'ppg_unweighted'):
                    w_ppg = float(np.sum(points_vec * weights) / max(1.0, np.sum(weights)))
                    u_ppg = float(np.mean(points_vec)) if len(points_vec) else 0.0
                    print(f"[FOLD] ppg_w={w_ppg:.4f} ppg={u_ppg:.4f} n={len(points_vec)}")
                else:
                    print(f"[FOLD] {obj}={metric_val:.6f} n={len(points_vec)}")

            fold_metrics.append(metric_val)

        # Objective: average of fold metrics
        if not fold_metrics:
            return float('inf') if direction == 'minimize' else float('-inf')
        return float(np.nanmean(fold_metrics))

    return obj_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials')
    parser.add_argument('--n-splits', type=int, default=3, help='TimeSeriesSplit folds')
    parser.add_argument('--omp-threads', type=int, default=1, help='Threads per worker for BLAS/OMP')
    # Removed final training options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logs from inner training loop')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///study.db) for multi-process tuning')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name (used with --storage)')
    parser.add_argument('--pruner', type=str, choices=['none','median','hyperband'], default='median', help='Enable trial pruning strategy')
    parser.add_argument('--pruner-startup-trials', type=int, default=20, help='Trials before enabling pruning (median pruner)')
    parser.add_argument('--objective', type=str, choices=['ppg','ppg_unweighted','logloss','brier','balanced_accuracy','accuracy','rps'], default='ppg', help='Tuning objective')
    parser.add_argument('--direction', type=str, choices=['auto','maximize','minimize'], default='auto', help='Study direction; auto selects based on objective')
    parser.add_argument('--compare', type=str, default=None, help='Comma-separated list of objectives to compare; when set, --objective is ignored')
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
        # Force single worker per process to avoid nested parallelism
        print(f"Planned: trials={total_trials} | cv-trainings={total_cv_trainings} | workers=1 x threads={args.omp_threads or 1}")

    # Verbosity reflects CLI flag only (quiet by default, even with multiple jobs)
    effective_verbose = bool(args.verbose)

    # Precompute CV folds once to ensure identical splits across objectives
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = list(tscv.split(features_df))

    # Determine objectives to run
    if args.compare:
        objectives_to_run = [o.strip() for o in args.compare.split(',') if o.strip()]
        if args.verbose:
            print(f"Compare mode active. Objectives: {objectives_to_run}. Ignoring --objective.")
            print(f"Compare mode will take approximately {len(objectives_to_run)}× the time of a single run.")
    else:
        objectives_to_run = [args.objective]

    # Configure pruner
    pruner = None
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(max(0, args.pruner_startup_trials)))
    elif args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner()

    # Track best per objective
    per_objective_best: Dict[str, Dict[str, object]] = {}
    storage_fs_path = _sqlite_fs_path(args.storage)

    try:
        for obj in objectives_to_run:
            study_direction = args.direction if args.direction != 'auto' else _objective_direction(obj)
            objective_fn = _objective_builder(features_df, all_matches, folds, obj, study_direction, args.omp_threads, effective_verbose)

            # Create study, optionally with storage
            if args.storage:
                study = optuna.create_study(
                    direction=study_direction,
                    storage=args.storage,
                    study_name=((args.study_name or 'kicktipp-tune') + f"-{obj}"),
                    load_if_exists=True,
                    pruner=pruner,
                )
            else:
                study = optuna.create_study(direction=study_direction, pruner=pruner)

            progress = {
                'start': time.time(),
                'completed': 0,
                'best_value': float('-inf') if study_direction == 'maximize' else float('inf'),
                'best_trial': None,
                'last_print_len': 0,
                'last_update_ts': 0.0,
            }

            def _progress_cb(study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:  # type: ignore[name-defined]
                progress['completed'] += 1
                try:
                    progress['best_value'] = float(study.best_value)
                    progress['best_trial'] = int(study.best_trial.number)
                except Exception:
                    pass
                elapsed = time.time() - progress['start']
                done = progress['completed']
                total = max(1, total_trials)
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = max(0.0, (total - done) / rate) if rate > 0 else float('inf')
                rate_str = f"{rate:.2f} t/s" if rate > 0 else "-- t/s"
                pct = (done / total) * 100 if total > 0 else 0.0
                bar_len = 24
                filled = int((pct / 100) * bar_len)
                bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
                line = (
                    f"\r[TUNE:{obj}] {bar} {done}/{total} ({pct:5.1f}%) | elapsed {_fmt_dur(elapsed)} | "
                    f"eta {_fmt_dur(remaining)} | speed {rate_str} | best={progress['best_value']:.3f}"
                )
                if progress['best_trial'] is not None:
                    line += f" (#{progress['best_trial']})"
                now_ts = time.time()
                should_update = (now_ts - progress['last_update_ts'] >= 0.5) or (done == total)
                if not should_update:
                    return
                progress['last_update_ts'] = now_ts
                pad = max(0, progress['last_print_len'] - len(line))
                if sys.stderr and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
                    print(line + (' ' * pad), end='', flush=True, file=sys.stderr)
                else:
                    print(line.replace('\r', ''), flush=True, file=sys.stderr)
                progress['last_print_len'] = len(line)

            if args.verbose:
                print(f"Running Optuna study for objective='{obj}' for {args.n_trials} trials with n_jobs=1...")
            start = time.time()
            study.optimize(
                objective_fn,
                n_trials=args.n_trials,
                n_jobs=1,
                callbacks=[_progress_cb],
                show_progress_bar=False,
            )
            print(file=sys.stderr)
            duration = time.time() - start
            if args.verbose:
                try:
                    print(f"Study complete in {duration:.1f}s. Best value={study.best_value:.6f} ({study_direction}).")
                except Exception:
                    print(f"Study complete in {duration:.1f}s. No completed trials.")

            try:
                completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))  # type: ignore[attr-defined]
            except Exception:
                completed_trials = []
            if not completed_trials:
                if args.verbose:
                    print(f"No completed trials for objective '{obj}'; skipping save.")
                continue

            best_params = dict(study.best_params)
            per_objective_best[obj] = {
                'best_value': float(study.best_value),
                'direction': study_direction,
                'params': best_params,
            }

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            cfg_dir = os.path.join(project_root, 'config')
            os.makedirs(cfg_dir, exist_ok=True)
            if yaml is not None:
                with open(os.path.join(cfg_dir, f'best_params_{obj}.yaml'), 'w', encoding='utf-8') as f:
                    yaml.safe_dump(best_params, f, sort_keys=True)
            else:
                import json
                with open(os.path.join(cfg_dir, f'best_params_{obj}.json'), 'w', encoding='utf-8') as f:
                    json.dump(best_params, f, indent=2)
            if args.verbose:
                print(f"Best params saved to config/best_params_{obj}.yaml (or .json)")

        # If we have results, re-evaluate on identical folds and choose winner by weighted PPG
        if not per_objective_best:
            if args.verbose:
                print("No completed trials across objectives; exiting.")
            return

        summaries: Dict[str, Dict[str, float]] = {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for obj, info in per_objective_best.items():
            params = info['params']  # type: ignore[assignment]
            reset_config()
            _apply_params_to_config(params)  # type: ignore[arg-type]
            # Recompute features for this params combo (ensures feature knobs take effect)
            dl = DataLoader()
            feats_df = dl.create_features_from_matches(all_matches)

            ppg_w_list: List[float] = []
            ppg_u_list: List[float] = []
            acc_w_list: List[float] = []
            bacc_w_list: List[float] = []
            brier_list: List[float] = []
            logloss_list: List[float] = []
            rps_list: List[float] = []

            for train_idx, test_idx in folds:
                reset_config()
                _apply_params_to_config(params)  # type: ignore[arg-type]
                train_df = feats_df.iloc[train_idx]
                test_df = feats_df.iloc[test_idx]
                test_feats = test_df.drop(columns=['home_score','away_score','goal_difference','result'], errors='ignore')
                predictor = MatchPredictor(quiet=not effective_verbose)
                predictor.train(train_df)
                preds = predictor.predict(test_feats)

                ph = np.array([int(p.get('predicted_home_score', 0)) for p in preds], dtype=int)
                pa = np.array([int(p.get('predicted_away_score', 0)) for p in preds], dtype=int)
                ah = np.asarray(test_df['home_score'], dtype=int)
                aa = np.asarray(test_df['away_score'], dtype=int)
                points_vec = compute_points(ph, pa, ah, aa).astype(float)

                if 'date' in test_df.columns:
                    fold_dates = pd.to_datetime(test_df['date'])
                    days_old = (fold_dates.max() - fold_dates).dt.days.astype(float)
                    half_life = float(params.get('time_decay_half_life_days', 90.0))  # type: ignore[union-attr]
                    decay_rate = np.log(2.0) / max(1.0, half_life)
                    weights = np.exp(-decay_rate * days_old.values)
                else:
                    weights = np.ones_like(points_vec, dtype=float)

                proba = np.array([
                    [p.get('home_win_probability', 0.0), p.get('draw_probability', 0.0), p.get('away_win_probability', 0.0)]
                    for p in preds
                ], dtype=float)
                y_true = test_df['result'].tolist()
                y_pred_idx = np.argmax(proba, axis=1) if len(proba) else np.array([])
                idx_to_label = {0: 'H', 1: 'D', 2: 'A'}
                y_pred = [idx_to_label.get(int(i), 'H') for i in y_pred_idx]

                ppg_w_list.append(float(np.sum(points_vec * weights) / max(1.0, np.sum(weights))))
                ppg_u_list.append(float(np.mean(points_vec)) if len(points_vec) else 0.0)
                try:
                    acc_w_list.append(float(accuracy_score(y_true, y_pred, sample_weight=weights)))
                except Exception:
                    acc_w_list.append(float('nan'))
                try:
                    bacc_w_list.append(float(balanced_accuracy_score(y_true, y_pred, sample_weight=weights)))
                except Exception:
                    bacc_w_list.append(float('nan'))
                brier_list.append(brier_score_multiclass(y_true, proba))
                logloss_list.append(log_loss_multiclass(y_true, proba))
                rps_list.append(ranked_probability_score(y_true, proba))

            summaries[obj] = {
                'ppg_weighted': float(np.nanmean(ppg_w_list)) if ppg_w_list else float('nan'),
                'ppg_unweighted': float(np.nanmean(ppg_u_list)) if ppg_u_list else float('nan'),
                'accuracy_weighted': float(np.nanmean(acc_w_list)) if acc_w_list else float('nan'),
                'balanced_accuracy_weighted': float(np.nanmean(bacc_w_list)) if bacc_w_list else float('nan'),
                'brier': float(np.nanmean(brier_list)) if brier_list else float('nan'),
                'log_loss': float(np.nanmean(logloss_list)) if logloss_list else float('nan'),
                'rps': float(np.nanmean(rps_list)) if rps_list else float('nan'),
            }

        # Choose winner by highest weighted PPG
        winner = None
        best_ppg = float('-inf')
        for obj, summ in summaries.items():
            ppgw = summ.get('ppg_weighted', float('-inf'))
            if ppgw is not None and ppgw > best_ppg:
                best_ppg = ppgw
                winner = obj

        cfg_dir = os.path.join(project_root, 'config')
        os.makedirs(cfg_dir, exist_ok=True)
        if winner:
            win_params = per_objective_best[winner]['params']  # type: ignore[index]
            if yaml is not None:
                with open(os.path.join(cfg_dir, 'best_params.yaml'), 'w', encoding='utf-8') as f:
                    yaml.safe_dump(win_params, f, sort_keys=True)
            else:
                import json
                with open(os.path.join(cfg_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
                    json.dump(win_params, f, indent=2)
            if args.verbose:
                print(f"Winner by weighted PPG: {winner}. Saved to config/best_params.yaml")

        # Save summary artifacts
        out_dir = os.path.join(project_root, 'data', 'predictions')
        os.makedirs(out_dir, exist_ok=True)
        try:
            import json
            with open(os.path.join(out_dir, 'metrics_tuning.json'), 'w', encoding='utf-8') as f:
                json.dump({'summaries': summaries, 'per_objective_best': per_objective_best}, f, indent=2)
            header = (
                "objective,ppg_weighted,ppg_unweighted,accuracy_weighted,balanced_accuracy_weighted,brier,log_loss,rps\n"
            )
            lines = [header]
            for obj, summ in summaries.items():
                lines.append(
                    f"{obj},{summ['ppg_weighted']:.6f},{summ['ppg_unweighted']:.6f},{summ['accuracy_weighted']:.6f},{summ['balanced_accuracy_weighted']:.6f},{summ['brier']:.6f},{summ['log_loss']:.6f},{summ['rps']:.6f}\n"
                )
            with open(os.path.join(out_dir, 'metrics_table_tuning.txt'), 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as _e:  # pragma: no cover
            if args.verbose:
                print(f"Warning: Failed to write comparison metrics: {_e}")

    finally:
        # Delete SQLite storage file, if applicable, unless coordinated by external CLI
        try:
            coordinated = os.environ.get('KTP_TUNE_COORDINATED', '0') == '1'
            if (not coordinated) and storage_fs_path and os.path.exists(storage_fs_path):
                os.remove(storage_fs_path)
                if args.verbose:
                    print(f"Deleted Optuna storage DB at {storage_fs_path}")
        except Exception as _e:
            if args.verbose:
                print(f"Warning: Failed to delete Optuna storage DB at {storage_fs_path}: {_e}")


if __name__ == "__main__":
    main()
