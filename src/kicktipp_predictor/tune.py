from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

try:
    import optuna  # type: ignore
    from optuna.samplers import NSGAIISampler  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    optuna = None

from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import multiprocessing as mp

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.table import Table

from .config import Config, get_config
from .data import DataLoader
from .metrics import ProbabilityMetrics, ConfusionMetrics, compute_points
from .predictor import CascadedPredictor
from sklearn.metrics import roc_auc_score, f1_score


def _prepare_datasets(seasons_back: int) -> tuple[pd.DataFrame, pd.DataFrame, DataLoader, int]:
    """Prepare training and validation datasets using a time-based split.

    - Training: older seasons (current_season - seasons_back .. current_season-1)
    - Validation: most recent season (current_season)

    Returns:
        (train_df, val_df, loader, current_season)
    """
    loader = DataLoader()
    current_season = loader.get_current_season()
    start_season = current_season - int(seasons_back)

    train_matches = loader.fetch_historical_seasons(start_season, current_season - 1)
    val_matches = loader.fetch_season_matches(current_season)

    train_df = loader.create_features_from_matches(train_matches)

    # Build validation features using ONLY training history as context to avoid leakage
    val_df = loader.create_prediction_features(val_matches, train_matches)
    # Derive true labels for finished validation matches from raw scores
    vm = pd.DataFrame(val_matches).copy()
    vm["match_id"] = vm["match_id"].astype(str)
    mask = vm["home_score"].notna() & vm["away_score"].notna()
    vm.loc[mask, "goal_difference"] = vm.loc[mask, "home_score"] - vm.loc[mask, "away_score"]
    vm.loc[mask, "result"] = np.where(
        vm.loc[mask, "goal_difference"] > 0,
        "H",
        np.where(vm.loc[mask, "goal_difference"] < 0, "A", "D"),
    )
    val_match_results = vm.loc[mask, ["match_id", "result", "goal_difference", "home_score", "away_score"]]
    val_df["match_id"] = val_df["match_id"].astype(str)
    val_df = val_df.merge(val_match_results, on="match_id", how="left")
    # Evaluate only finished matches with known labels
    if "result" in val_df.columns:
        val_df = val_df.loc[val_df["result"].notna()].copy()

    # Sanity checks
    if len(train_df) == 0:
        raise RuntimeError("No training samples found. Increase seasons_back or check data availability.")
    if len(val_df) == 0:
        raise RuntimeError("No validation samples found in the current season.")
    if "result" not in val_df.columns:
        raise RuntimeError("Validation dataset missing 'result' labels.")

    return train_df, val_df, loader, current_season


def _reset_optuna_storage(storage_url: str) -> None:
    """Reset Optuna storage to a clean initial state.

    - For SQLite URLs (sqlite:///path), deletes the DB file and ensures parent dir exists.
    - For other RDB URLs, deletes all existing studies via Optuna API.
    - Validates reset by checking that no studies remain.
    """
    if optuna is None:
        return
    try:
        if storage_url.startswith("sqlite:///"):
            db_path = Path(storage_url.replace("sqlite:///", ""))
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # Remove existing DB file
            if db_path.exists():
                db_path.unlink()
            # Force schema re-init and validate empty state
            summaries = optuna.study.get_all_study_summaries(storage_url)
            if len(summaries) != 0:
                raise RuntimeError("Database reset verification failed: studies still present.")
        else:
            # Best-effort cleanup for non-SQLite storages
            summaries = optuna.study.get_all_study_summaries(storage_url)
            for s in summaries:
                try:
                    optuna.delete_study(study_name=s.study_name, storage=storage_url)
                except Exception:
                    # Continue deleting others even if a single delete fails
                    pass
            # Validate empty
            summaries_after = optuna.study.get_all_study_summaries(storage_url)
            if len(summaries_after) != 0:
                raise RuntimeError("Database reset verification failed: studies still present.")
    except Exception as exc:
        raise RuntimeError(f"Failed to reset Optuna storage '{storage_url}': {exc}")


def _compute_workers(workers: Optional[int], n_trials: int | None = None) -> int:
    """Compute an effective worker count.

    - If `workers` is None or <= 0, use `max(1, (os.cpu_count() or 2) - 1)`.
    - If `n_trials` is specified, cap workers to `n_trials` to avoid idle workers.
    """
    auto = max(1, (os.cpu_count() or 2) - 1)
    w = int(workers or auto)
    if n_trials is not None:
        w = min(w, int(max(1, n_trials)))
    return w


def _split_trials(n_trials: int, workers: int) -> list[int]:
    """Split `n_trials` approximately evenly across `workers`.

    Distributes the remainder across the first few workers.
    """
    base, rem = divmod(int(n_trials), int(workers))
    parts = [base + (1 if i < rem else 0) for i in range(workers)]
    return [p for p in parts if p > 0]


def _parse_log_level(level: str) -> int:
    level = (level or "warning").upper()
    return getattr(logging, level, logging.WARNING)


def _set_logging_level(level: int) -> None:
    logging.basicConfig(level=level)
    logging.getLogger("optuna").setLevel(level)
    logging.getLogger("xgboost").setLevel(level)
    logging.getLogger("sklearn").setLevel(level)
    logging.getLogger("kicktipp_predictor").setLevel(level)


def _limit_threads(max_threads: int) -> None:
    """Limit per-process threads to avoid oversubscription.

    Sets common thread-related env vars so libraries like OpenMP, BLAS, and
    XGBoost respect the cap. Also used so Config picks a safe `n_jobs`.
    """
    n = max(1, int(max_threads))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        # XGBoost mostly uses n_jobs param; keep env for good measure
        "XGB_NUM_THREADS",
    ):
        os.environ[var] = str(n)


def _resolve_storage_with_timeout(storage_url: str):
    """Return an Optuna storage object with a friendly SQLite timeout.

    For SQLite URLs, creates RDBStorage with `connect_args={'timeout': 60.0}`
    to reduce 'database is locked' commit errors under concurrency.
    Otherwise returns the input URL.
    """
    if optuna is None:
        return storage_url
    try:
        if isinstance(storage_url, str) and storage_url.startswith("sqlite:///"):
            return optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={"connect_args": {"timeout": 60.0}},
            )
    except Exception:
        # Fallback to plain URL if RDBStorage construction fails
        pass
    return storage_url


# Main entry point for tuning is now in cli.py
# This module is imported by cli.py to provide tuning functionality


# ==========================================================================
# V4 Cascaded Model: Objective Function Factory
# ==========================================================================

def create_objective_function(
    model_to_tune: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fixed_config: Config | None = None,
    draw_metric: str = "roc_auc",
    multi_objective_draw: bool = True,
) -> Callable[[optuna.Trial], float | tuple[float, ...]]:
    """
    Factory that returns the correct Optuna objective function based on tuning mode.
    
    This consolidates all the duplicated logic and eliminates ~100 lines of code.
    
    Args:
        model_to_tune: 'draw', 'win', or 'both'
        train_df: Training data
        val_df: Validation data
        fixed_config: Optional config with fixed params (for cascaded tuning)
        draw_metric: Metric to use for draw tuning ('roc_auc' or 'f1')
        multi_objective_draw: If True, draw tuning optimizes both roc_auc and draw_rate_error
    
    Returns:
        Objective function for Optuna optimization
    """
    if model_to_tune == 'draw':
        # Draw-only tuning with system-level evaluation
        # Can be single-objective (roc_auc) or multi-objective (roc_auc + draw_rate_error)
        def objective(trial: optuna.Trial) -> float | tuple[float, float]:
            cfg = Config.load()
            cfg.model.n_jobs = 1
            
            # Suggest draw params
            cfg.model.draw_n_estimators = trial.suggest_int("draw_n_estimators", 100, 1500)
            cfg.model.draw_max_depth = trial.suggest_int("draw_max_depth", 3, 10)
            cfg.model.draw_learning_rate = trial.suggest_float("draw_learning_rate", 0.01, 0.3, log=True)
            cfg.model.draw_subsample = trial.suggest_float("draw_subsample", 0.6, 1.0)
            cfg.model.draw_colsample_bytree = trial.suggest_float("draw_colsample_bytree", 0.5, 1.0)
            cfg.model.draw_scale_pos_weight = trial.suggest_float("draw_scale_pos_weight", 1.0, 8.0)
            
            # Fix win params to reasonable defaults (if not provided, use current config)
            if fixed_config:
                cfg.model.win_n_estimators = fixed_config.model.win_n_estimators
                cfg.model.win_max_depth = fixed_config.model.win_max_depth
                cfg.model.win_learning_rate = fixed_config.model.win_learning_rate
                cfg.model.win_subsample = fixed_config.model.win_subsample
                cfg.model.win_colsample_bytree = fixed_config.model.win_colsample_bytree
            
            try:
                predictor = CascadedPredictor(config=cfg)
                predictor.train(train_df, verbose=False)
            except Exception as exc:
                raise optuna.TrialPruned(f"Training failed: {exc}")
            
            # Evaluate complete system performance
            preds = predictor.predict(val_df, verbose=False)
            proba = np.array(
                [
                    [
                        p.get("home_win_probability", 0.0),
                        p.get("draw_probability", 0.0),
                        p.get("away_win_probability", 0.0),
                    ]
                    for p in preds
                ],
                dtype=float,
            )
            
            # Calculate draw metric from system-level predictions
            y_val_draw = (val_df["result"].astype(str) == "D").astype(int).to_numpy()
            p_draw_system = proba[:, 1]  # Draw probability from complete system
            
            if len(np.unique(y_val_draw)) < 2:
                raise optuna.TrialPruned("Validation set lacks both draw and non-draw classes.")
            
            # Also compute overall accuracy for reference
            y_true = val_df["result"].astype(str).tolist()
            y_true_numeric = np.array([{"H": 0, "D": 1, "A": 2}[t] for t in y_true])
            accuracy = float(np.mean(np.argmax(proba, axis=1) == y_true_numeric))
            
            trial.set_user_attr("system_accuracy", accuracy)
            trial.set_user_attr("metric", draw_metric)
            trial.set_user_attr("evaluation_mode", "system")
            
            # Calculate the draw metric
            if draw_metric == "roc_auc":
                score = float(roc_auc_score(y_val_draw, p_draw_system))
            elif draw_metric == "f1":
                y_pred = (p_draw_system >= 0.5).astype(int)
                score = float(f1_score(y_val_draw, y_pred))
            else:
                raise optuna.TrialPruned(f"Unsupported draw_metric: {draw_metric}")
            
            trial.set_user_attr("score", score)
            
            # If multi-objective, also return draw_rate_error
            if multi_objective_draw:
                actual_draw_rate = 0.25  # Target draw rate
                pred_labels = np.argmax(proba, axis=1)
                predicted_draw_rate = np.mean(pred_labels == 1)
                draw_rate_error = abs(predicted_draw_rate - actual_draw_rate)
                
                trial.set_user_attr("predicted_draw_rate", predicted_draw_rate)
                trial.set_user_attr("draw_rate_error", draw_rate_error)
                
                return score, draw_rate_error  # Maximize roc_auc, minimize draw_rate_error
            
            return score
        
        return objective
    
    elif model_to_tune in ('win', 'both'):
        # Win model tuning (or both) with fixed draw params
        def objective(trial: optuna.Trial) -> tuple[float, float, float]:
            cfg = Config.load()
            cfg.model.n_jobs = 1
            
            # Apply fixed draw params
            if fixed_config:
                cfg.model.draw_n_estimators = fixed_config.model.draw_n_estimators
                cfg.model.draw_max_depth = fixed_config.model.draw_max_depth
                cfg.model.draw_learning_rate = fixed_config.model.draw_learning_rate
                cfg.model.draw_subsample = fixed_config.model.draw_subsample
                cfg.model.draw_colsample_bytree = fixed_config.model.draw_colsample_bytree
                cfg.model.draw_scale_pos_weight = fixed_config.model.draw_scale_pos_weight
            
            # Suggest win params
            cfg.model.win_n_estimators = trial.suggest_int("win_n_estimators", 100, 2000)
            cfg.model.win_max_depth = trial.suggest_int("win_max_depth", 3, 10)
            cfg.model.win_learning_rate = trial.suggest_float("win_learning_rate", 0.01, 0.3, log=True)
            cfg.model.win_subsample = trial.suggest_float("win_subsample", 0.6, 1.0)
            cfg.model.win_colsample_bytree = trial.suggest_float("win_colsample_bytree", 0.5, 1.0)
            
            try:
                predictor = CascadedPredictor(config=cfg)
                predictor.train(train_df, verbose=False)
            except Exception as exc:
                raise optuna.TrialPruned(f"Training failed: {exc}")
            
            # Get predictions from full system
            preds = predictor.predict(val_df, verbose=False)
            proba = np.array(
                [
                    [
                        p.get("home_win_probability", 0.0),
                        p.get("draw_probability", 0.0),
                        p.get("away_win_probability", 0.0),
                    ]
                    for p in preds
                ],
                dtype=float,
            )
            y_true = val_df["result"].astype(str).tolist()
            y_true_numeric = np.array([{"H": 0, "D": 1, "A": 2}[t] for t in y_true])
            
            # Calculate the three metrics
            # 1. Accuracy
            accuracy = float(np.mean(np.argmax(proba, axis=1) == y_true_numeric))
            
            # 2. Log Loss
            log_loss = float(ProbabilityMetrics.log_loss_multiclass(y_true, proba))
            
            # 3. PPG via scoreline heuristic
            # Ensure predictor uses current cfg heuristic bins
            predictor.config.model.heuristic_home_win_bins = cfg.model.heuristic_home_win_bins
            predictor.config.model.heuristic_away_win_bins = cfg.model.heuristic_away_win_bins
            predictor.config.model.heuristic_draw_bins = cfg.model.heuristic_draw_bins

            scoreline_preds = [predictor._get_scoreline_from_probs(p) for p in proba]
            pred_home_scores = [int(s[0]) for s in scoreline_preds]
            pred_away_scores = [int(s[1]) for s in scoreline_preds]

            actual_home_scores = val_df["home_score"].astype(int).tolist()
            actual_away_scores = val_df["away_score"].astype(int).tolist()

            points = compute_points(pred_home_scores, pred_away_scores, actual_home_scores, actual_away_scores)
            ppg = float(np.mean(points))
            
            # Store for analysis
            trial.set_user_attr("accuracy", accuracy)
            trial.set_user_attr("log_loss", log_loss)
            # Keep draw rate for analysis
            pred_labels = np.argmax(proba, axis=1)
            predicted_draw_rate = float(np.mean(pred_labels == 1))
            trial.set_user_attr("predicted_draw_rate", predicted_draw_rate)
            trial.set_user_attr("ppg", ppg)
            
            return accuracy, log_loss, ppg
        
        return objective
    
    else:
        raise ValueError(f"Invalid model_to_tune: {model_to_tune}")


# ==========================================================================
# V4 Cascaded Model: Sequential Tuning (Draw -> Win)
# ==========================================================================

def _save_best_params(update: dict[str, float | int | str]) -> Path:
    """Persist/merge best parameters into package config best_params.yaml.

    - Reads existing YAML if present and merges the provided update keys.
    - Writes back to `src/kicktipp_predictor/config/best_params.yaml`.
    - Falls back to JSON on YAML errors.

    Returns:
        Path to the written YAML/JSON file.
    """
    pkg_config_dir = Path(__file__).parent / "config"
    pkg_config_dir.mkdir(parents=True, exist_ok=True)
    out_path = pkg_config_dir / "best_params.yaml"
    existing: dict[str, float | int | str] = {}
    try:
        import yaml  # type: ignore
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    existing = data
        # Merge and write
        merged = dict(existing)
        merged.update(update)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, sort_keys=False, indent=2)
        # Basic validation
        with open(out_path, "r", encoding="utf-8") as f:
            _validate = yaml.safe_load(f)
        if not isinstance(_validate, dict):
            raise ValueError("Invalid YAML after write")
        return out_path
    except Exception:
        alt = out_path.with_suffix(".json")
        merged = dict(existing)
        merged.update(update)
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        return alt


def run_tuning_v4_sequential(
    n_trials: int = 100,
    seasons_back: int = 5,
    storage: Optional[str] = None,
    study_name: str = "v4_cascaded_sequential",
    timeout: Optional[int] = None,
    model_to_tune: str = "both",  # 'draw' | 'win' | 'both'
    draw_metric: str = "roc_auc",  # 'roc_auc' | 'f1'
    win_metric: str = "accuracy",  # 'accuracy' | 'log_loss'
    reset_storage: bool = False,
) -> None:
    """Sequential tuning for V4 CascadedPredictor.

    This is a thin wrapper around run_tuning_v4_parallel with workers=1.
    All tuning now uses the same unified implementation.
    """
    # Simply delegate to parallel with workers=1
    run_tuning_v4_parallel(
        n_trials=n_trials,
        seasons_back=seasons_back,
        storage=storage,
        study_name=study_name,
        timeout=timeout,
        model_to_tune=model_to_tune,
        draw_metric=draw_metric,
        win_metric=win_metric,
        workers=1,  # Sequential mode
        bench_trials=None,
        log_level="warning",
        reset_storage=reset_storage,
    )


# ==========================================================================
# V4 Cascaded Model: Parallel Tuning (Draw -> Win)
# ==========================================================================

def _worker_initializer():
    """Initialize worker process - called once when process starts.
    
    This runs BEFORE any code in the worker function and sets environment
    variables so that imported libraries (numpy/scipy/sklearn) respect them.
    """
    import os
    # Set to 1 thread per worker to avoid OpenMP thread exhaustion
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "XGB_NUM_THREADS",
    ):
        os.environ[var] = "1"
    import logging
    logging.basicConfig(level=logging.INFO, format="Worker PID %(process)d: %(message)s")
    logging.info("Worker process initialized with OMP_NUM_THREADS=1")


def _worker_optimize_draw(
    storage_url: str,
    study_name: str,
    n_trials_worker: int,
    timeout: Optional[int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    draw_metric: str,
    log_level: int,
) -> tuple[int, Optional[str], Optional[dict]]:
    """Worker process to optimize the draw model using system-level evaluation.

    Uses the unified objective function factory - always evaluates the full cascaded system.
    Returns (n_completed_trials, error_message_if_any, best_params_or_none).
    """
    # CRITICAL: Set thread limits FIRST, before any library initialization
    # This prevents OpenMP from creating too many threads
    env_omp = os.environ.get("OMP_NUM_THREADS", "NOT_SET")
    logging.info(f"Worker PID {os.getpid()}: OMP_NUM_THREADS={env_omp}")
    _limit_threads(int(env_omp) if env_omp != "NOT_SET" else 1)
    logging.info(f"Worker PID {os.getpid()}: Set thread limits")
    _set_logging_level(log_level)
    storage = _resolve_storage_with_timeout(storage_url)
    
    # Multi-objective optimization for draw (maximize roc_auc, minimize draw_rate_error)
    sampler_draw = NSGAIISampler()
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage,
        sampler=sampler_draw,
        load_if_exists=True,
    )

    # Use unified objective function factory with system-level evaluation and multi-objective
    # This always evaluates the draw model as part of the complete cascaded system
    objective_draw = create_objective_function('draw', train_df, val_df, None, draw_metric, multi_objective_draw=True)

    before = len(study.trials)
    error: Optional[str] = None
    try:
        study.optimize(objective_draw, n_trials=int(n_trials_worker), timeout=timeout, catch=(Exception,))
    except Exception as exc:  # pragma: no cover - safeguard
        error = str(exc)
    after = len(study.trials)
    
    # Return best params (will be written by main process only)
    best_params = None
    if after > before:
        try:
            best_trial = study.best_trial
            best_params = dict(best_trial.params)
        except Exception:
            pass
    
    return max(0, after - before), error, best_params


def _worker_optimize_win(
    storage_url: str,
    study_name: str,
    n_trials_worker: int,
    timeout: Optional[int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    win_metric: str,
    cfg_fixed: Config,
    log_level: int,
) -> tuple[int, Optional[str], Optional[dict]]:
    """Worker process to optimize the win model.

    Uses the unified objective function factory.
    Returns (n_completed_trials, error_message_if_any, best_params_or_none).
    """
    # CRITICAL: Set thread limits FIRST, before any library initialization
    # This prevents OpenMP from creating too many threads
    env_omp = os.environ.get("OMP_NUM_THREADS", "NOT_SET")
    logging.info(f"Worker PID {os.getpid()}: OMP_NUM_THREADS={env_omp}")
    _limit_threads(int(env_omp) if env_omp != "NOT_SET" else 1)
    logging.info(f"Worker PID {os.getpid()}: Set thread limits")
    _set_logging_level(log_level)
    storage = _resolve_storage_with_timeout(storage_url)
    sampler = NSGAIISampler()
    
    # Simply load the study - it should already exist from main process
    # Use load_if_exists to avoid race conditions with many workers
    study = optuna.create_study(
        directions=["maximize", "minimize", "maximize"],
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    # Use unified objective function factory
    objective_win = create_objective_function('win', train_df, val_df, cfg_fixed)

    before = len(study.trials)
    error: Optional[str] = None
    try:
        study.optimize(objective_win, n_trials=int(n_trials_worker), timeout=timeout, catch=(Exception,))
    except Exception as exc:  # pragma: no cover - safeguard
        error = str(exc)
    after = len(study.trials)
    
    # Return best params (will be written by main process only)
    best_params = None
    if after > before:
        try:
            best_trial = study.best_trial
            best_params = dict(best_trial.params)
        except Exception:
            pass
    
    return max(0, after - before), error, best_params


def _bench_sequential(
    objective: Callable[[optuna.Trial], float],
    storage_url: str,
    study_name: str,
    n_trials: int,
    direction: str,
) -> float:
    """Run a small sequential benchmark to estimate per-trial time.

    Returns average seconds per trial.
    """
    storage = _resolve_storage_with_timeout(storage_url)
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=int(n_trials), catch=(Exception,))
    dur = max(1e-9, time.perf_counter() - start)
    return float(dur / max(1, n_trials))


def run_tuning_v4_parallel(
    n_trials: int = 100,
    seasons_back: int = 5,
    storage: Optional[str] = None,
    study_name: str = "v4_cascaded_parallel",
    timeout: Optional[int] = None,
    model_to_tune: str = "both",  # 'draw' | 'win' | 'both'
    draw_metric: str = "roc_auc",  # 'roc_auc' | 'f1'
    win_metric: str = "accuracy",  # 'accuracy' | 'log_loss'
    workers: Optional[int] = None,
    bench_trials: Optional[int] = None,
    log_level: str = "warning",
    reset_storage: bool = False,
) -> None:
    """Parallel tuning for V4 CascadedPredictor using multiple workers.

    - Dynamically scales workers to available CPUs (or user-provided).
    - Uses Optuna with shared storage to distribute trials safely.
    - Rich progress UI with unified bars and color-coded statuses.
    - Optional sequential benchmark to estimate speedup.
    """
    if optuna is None:
        raise RuntimeError(
            "Optuna not installed. Install with `pip install \"kicktipp-predictor[tuning]\"` or `pip install optuna`."
        )

    console = Console()
    lvl = _parse_log_level(log_level)
    _set_logging_level(lvl)

    # Prepare datasets
    train_df, val_df, loader, current_season = _prepare_datasets(seasons_back)
    cfg_base = loader.config

    # Default storage (SQLite) under data dir
    default_storage = f"sqlite:///{cfg_base.paths.data_dir / 'optuna_studies.db'}"
    storage_url = storage or default_storage

    # Optional storage reset for reproducibility
    if reset_storage:
        _reset_optuna_storage(storage_url)

    # Compute workers and split trials
    w = _compute_workers(workers, n_trials)
    
    # Warn if too many workers specified
    cpu_count = os.cpu_count() or 8
    if w > cpu_count * 2:
        console.print(
            f"[yellow]WARNING:[/yellow] Using {w} workers on {cpu_count} CPU cores. "
            f"Consider reducing to avoid resource exhaustion. Suggest: --workers {min(cpu_count, n_trials)}"
        )
    
    parts = _split_trials(n_trials, w)

    # Set per-worker threads to 1 (fixed) to avoid libgomp thread creation failures
    # With many workers, each worker should use only 1 thread to avoid oversubscription
    # Set environment variables BEFORE spawning workers so they're inherited
    tpw = 1  # Fixed at 1 thread per worker
    # Pre-set env vars in main process so workers inherit them
    os.environ["OMP_NUM_THREADS"] = str(tpw)
    os.environ["OPENBLAS_NUM_THREADS"] = str(tpw)
    os.environ["MKL_NUM_THREADS"] = str(tpw)
    os.environ["NUMEXPR_NUM_THREADS"] = str(tpw)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(tpw)
    os.environ["XGB_NUM_THREADS"] = str(tpw)
    _limit_threads(tpw)
    storage_for_optuna = _resolve_storage_with_timeout(storage_url)

    console.rule("[bold cyan]OPTUNA TUNING (V4 Cascaded)[/bold cyan]")
    console.print(
        f"[dim]Training:[/dim] [cyan]{current_season - seasons_back}..{current_season - 1}[/cyan] | "
        f"[dim]Validation:[/dim] [yellow]{current_season}[/yellow] | "
        f"[dim]Workers:[/dim] [green]{w}[/green] | "
        f"[dim]Storage:[/dim] [blue]{Path(storage_url).name}[/blue]"
    )
    try:
        console.print(f"[dim]Start method:[/dim] [magenta]{mp.get_start_method() or 'default'}[/magenta]")
    except Exception:
        pass

    # Progress UI
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    # ---------------------- Phase 1: Draw Tuning ----------------------
    if model_to_tune in ("draw", "both"):
        # Create multi-objective study for draw model (roc_auc + draw_rate_error)
        # This ensures draw predictions are both accurate and realistic
        sampler_draw = NSGAIISampler()
        optuna.create_study(
            directions=["maximize", "minimize"],  # Maximize roc_auc, minimize draw_rate_error
            study_name=f"{study_name}_draw",
            storage=storage_for_optuna,
            sampler=sampler_draw,
            load_if_exists=True,
        )

        bench_avg_draw: Optional[float] = None
        if bench_trials and bench_trials > 0:
            # Local sequential bench to estimate per-trial baseline
            def bench_obj(trial: optuna.Trial) -> float:
                cfg = Config.load()
                cfg.model.draw_n_estimators = trial.suggest_int("draw_n_estimators", 100, 1500)
                cfg.model.draw_max_depth = trial.suggest_int("draw_max_depth", 3, 10)
                cfg.model.draw_learning_rate = trial.suggest_float("draw_learning_rate", 0.01, 0.3, log=True)
                cfg.model.draw_subsample = trial.suggest_float("draw_subsample", 0.6, 1.0)
                cfg.model.draw_colsample_bytree = trial.suggest_float("draw_colsample_bytree", 0.5, 1.0)
                cfg.model.draw_scale_pos_weight = trial.suggest_float("draw_scale_pos_weight", 1.0, 8.0)
                predictor = CascadedPredictor(config=cfg)
                predictor.train(train_df, verbose=False)
                X_val = predictor._prepare_features(val_df)
                y_val_draw = (val_df["result"].astype(str) == "D").astype(int).to_numpy()
                draw_proba = predictor.draw_model.predict_proba(X_val)
                draw_label = int(predictor.draw_label_encoder.transform([1])[0])
                idx = int(np.where(predictor.draw_model.classes_ == draw_label)[0][0])
                p_draw = draw_proba[:, idx]
                if draw_metric == "roc_auc":
                    return float(roc_auc_score(y_val_draw, p_draw))
                y_pred = (p_draw >= 0.5).astype(int)
                return float(f1_score(y_val_draw, y_pred))

            bench_avg_draw = _bench_sequential(
                bench_obj,
                storage_url,
                f"{study_name}_draw_bench",
                int(bench_trials),
                "maximize",
            )

        task_draw = progress.add_task(
            "[cyan]Phase 1:[/cyan] Draw Classifier",
            total=int(n_trials)
        )
        start_draw = time.perf_counter()
        failures_draw = 0

        with progress:
            # Use spawn to avoid inheriting pre-initialized OpenMP state from parent (fixes libgomp errors on Linux)
            spawn_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=w, initializer=_worker_initializer, mp_context=spawn_ctx) as ex:
                futures = [
                    ex.submit(
                        _worker_optimize_draw,
                        storage_url,
                        f"{study_name}_draw",
                        int(n_tw),
                        timeout,
                        train_df,
                        val_df,
                        draw_metric,
                        lvl,
                    )
                    for n_tw in parts
                ]

                while futures:
                    done, not_done = wait(futures, timeout=0.5)
                    
                    # Update progress (best effort, don't block on DB locks)
                    try:
                        s = optuna.load_study(study_name=f"{study_name}_draw", storage=storage_for_optuna)
                        n_completed = len(s.trials)
                        progress.update(task_draw, completed=min(n_completed, int(n_trials)))
                    except Exception:
                        # Silently skip progress update on DB lock/timeout
                        pass
                    
                    # Process completed futures immediately
                    for f in done:
                        try:
                            n_done, err, _ = f.result()
                            if err:
                                failures_draw += 1
                                console.print(f"[yellow]Worker error:[/yellow] {err}")
                        except Exception as exc:  # pragma: no cover - defensive
                            failures_draw += 1
                            console.print(f"[yellow]Worker raised:[/yellow] {exc}")
                    
                    futures = list(not_done)
                    
                    # Safety check: if no futures are done and all might be stuck, check progress
                    if not done and len(futures) > 0:
                        try:
                            # Quick check if we've hit the trial limit
                            s = optuna.load_study(study_name=f"{study_name}_draw", storage=storage_for_optuna)
                            if len(s.trials) >= int(n_trials):
                                # Manually cancel remaining futures
                                for f in futures:
                                    f.cancel()
                                futures = []
                        except Exception:
                            pass

        dur_draw = time.perf_counter() - start_draw
        study_draw = optuna.load_study(study_name=f"{study_name}_draw", storage=storage_for_optuna)
        try:
            # Multi-objective selection: prioritize roc_auc
            pareto_front = study_draw.best_trials
            if pareto_front:
                best_draw = max(pareto_front, key=lambda t: t.values[0])  # Maximize roc_auc
            else:
                best_draw = study_draw.best_trial
        except Exception as exc:
            best_draw = None

        saved_path = None
        if best_draw is not None:
            params_d = dict(best_draw.params)
            saved_path = _save_best_params(
                {
                    "draw_n_estimators": int(params_d.get("draw_n_estimators", cfg_base.model.draw_n_estimators)),
                    "draw_max_depth": int(params_d.get("draw_max_depth", cfg_base.model.draw_max_depth)),
                    "draw_learning_rate": float(params_d.get("draw_learning_rate", cfg_base.model.draw_learning_rate)),
                    "draw_subsample": float(params_d.get("draw_subsample", cfg_base.model.draw_subsample)),
                    "draw_colsample_bytree": float(params_d.get("draw_colsample_bytree", cfg_base.model.draw_colsample_bytree)),
                    "draw_scale_pos_weight": float(params_d.get("draw_scale_pos_weight", cfg_base.model.draw_scale_pos_weight)),
                }
            )

    # Summary table (multi-objective)
    tbl = Table(title="[cyan]Phase 1 Complete: Draw Classifier (Multi-Objective)[/cyan]", show_lines=False, box=None)
    tbl.add_column("Metric", style="dim")
    tbl.add_column("Value", style="bold")
    if best_draw and hasattr(best_draw, 'values'):
        tbl.add_row("ROC-AUC", f"[green]{best_draw.values[0]:.4f}[/green]" if len(best_draw.values) >= 1 else "-")
        tbl.add_row("Draw Rate Error", f"[yellow]{best_draw.values[1]:.4f}[/yellow]" if len(best_draw.values) >= 2 else "-")
    else:
        tbl.add_row("Best score", "-" if best_draw is None else f"[green]{float(best_draw.value) if hasattr(best_draw, 'value') else 0.0:.4f}[/green]")
    tbl.add_row("Trials", f"{len(study_draw.trials)}")
    tbl.add_row("Duration", f"[cyan]{dur_draw:.1f}s[/cyan]")
    if bench_avg_draw is not None:
        est_seq_total = bench_avg_draw * n_trials
        speedup = est_seq_total / max(1e-9, dur_draw)
        tbl.add_row("Avg/trial", f"{bench_avg_draw:.3f}s")
        tbl.add_row("Estimated seq", f"{est_seq_total:.1f}s")
        tbl.add_row("Speedup", f"[green]{speedup:.2f}x[/green]")
    status = "[green]✓ OK" if failures_draw == 0 else f"[red]✗ Failed ({failures_draw})[/red]"
    tbl.add_row("Status", status)
    console.print(tbl)

    # Save compact summary YAML
    summary_dir = cfg_base.paths.data_dir / "optuna"
    os.makedirs(summary_dir, exist_ok=True)
    try:
        import yaml  # type: ignore
        with open(summary_dir / f"{study_name}_draw_summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "study_name": f"{study_name}_draw",
                    "storage": storage_url,
                    "n_trials": len(study_draw.trials),
                    "multi_objective": True,
                    "best_trial": None if best_draw is None else {
                        "number": best_draw.number,
                        "values": best_draw.values if hasattr(best_draw, 'values') else None,
                        "roc_auc": float(best_draw.values[0]) if (hasattr(best_draw, 'values') and len(best_draw.values) >= 1) else None,
                        "draw_rate_error": float(best_draw.values[1]) if (hasattr(best_draw, 'values') and len(best_draw.values) >= 2) else None,
                        "predicted_draw_rate": float(best_draw.user_attrs.get("predicted_draw_rate", 0.0)) if best_draw.user_attrs.get("predicted_draw_rate") else None,
                        "params": best_draw.params,
                        "metric": draw_metric,
                    },
                    "updated_file": None if saved_path is None else str(saved_path),
                    "duration_seconds": float(dur_draw),
                    "workers": int(w),
                },
                f,
                sort_keys=False,
                indent=2,
            )
    except Exception:
        pass

    # ---------------------- Phase 2: Win Tuning ----------------------
    if model_to_tune in ("win", "both"):
        cfg_fixed = Config.load()
        sampler_win = NSGAIISampler()
        
        # Create study if it doesn't exist, with correct directions
        # Use load_if_exists to avoid errors if study already exists
        try:
            existing_study = optuna.load_study(study_name=f"{study_name}_win", storage=storage_for_optuna)
            directions = existing_study.directions
            if len(directions) != 3 or directions != [optuna.study.StudyDirection.MAXIMIZE, optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MAXIMIZE]:
                console.print(f"[yellow]Warning: Existing study has incorrect directions. Deleting and recreating with multi-objective directions.[/yellow]")
                optuna.delete_study(study_name=f"{study_name}_win", storage=storage_for_optuna)
                optuna.create_study(
                    directions=["maximize", "minimize", "maximize"],
                    study_name=f"{study_name}_win",
                    storage=storage_for_optuna,
                    sampler=sampler_win,
                    load_if_exists=False,
                )
            else:
                console.print(f"[green]Found existing study with correct multi-objective directions.[/green]")
        except KeyError:
            # Study doesn't exist, create it
            console.print(f"[cyan]Creating new multi-objective study.[/cyan]")
            optuna.create_study(
                directions=["maximize", "minimize", "maximize"],
                study_name=f"{study_name}_win",
                storage=storage_for_optuna,
                sampler=sampler_win,
            )

        bench_avg_win: Optional[float] = None
        if bench_trials and bench_trials > 0:
            def bench_obj_w(trial: optuna.Trial) -> float:
                cfg = Config.load()
                cfg.model.draw_n_estimators = cfg_fixed.model.draw_n_estimators
                cfg.model.draw_max_depth = cfg_fixed.model.draw_max_depth
                cfg.model.draw_learning_rate = cfg_fixed.model.draw_learning_rate
                cfg.model.draw_subsample = cfg_fixed.model.draw_subsample
                cfg.model.draw_colsample_bytree = cfg_fixed.model.draw_colsample_bytree
                cfg.model.draw_scale_pos_weight = cfg_fixed.model.draw_scale_pos_weight
                cfg.model.win_n_estimators = trial.suggest_int("win_n_estimators", 100, 2000)
                cfg.model.win_max_depth = trial.suggest_int("win_max_depth", 3, 10)
                cfg.model.win_learning_rate = trial.suggest_float("win_learning_rate", 0.01, 0.3, log=True)
                cfg.model.win_subsample = trial.suggest_float("win_subsample", 0.6, 1.0)
                cfg.model.win_colsample_bytree = trial.suggest_float("win_colsample_bytree", 0.5, 1.0)
                predictor = CascadedPredictor(config=cfg)
                predictor.train(train_df, verbose=False)
                preds = predictor.predict(val_df, verbose=False)
                proba = np.array(
                    [
                        [
                            p.get("home_win_probability", 0.0),
                            p.get("draw_probability", 0.0),
                            p.get("away_win_probability", 0.0),
                        ]
                        for p in preds
                    ],
                    dtype=float,
                )
                y_true = val_df["result"].astype(str).tolist()
                if win_metric == "accuracy":
                    acc = float(
                        np.mean(
                            np.argmax(proba, axis=1)
                            == np.array([{ "H": 0, "D": 1, "A": 2 }[t] for t in y_true])
                        )
                    )
                    return acc
                return float(ProbabilityMetrics.log_loss_multiclass(y_true, proba))

            bench_avg_win = _bench_sequential(
                bench_obj_w,
                storage_url,
                f"{study_name}_win_bench",
                int(bench_trials),
                "maximize" if win_metric == "accuracy" else "minimize",
            )

        task_win = progress.add_task(
            "[magenta]Phase 2:[/magenta] Win Classifier",
            total=int(n_trials)
        )
        start_win = time.perf_counter()
        failures_win = 0

        with progress:
            # Use spawn to avoid inheriting pre-initialized OpenMP state from parent (fixes libgomp errors on Linux)
            spawn_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=w, initializer=_worker_initializer, mp_context=spawn_ctx) as ex:
                futures = [
                    ex.submit(
                        _worker_optimize_win,
                        storage_url,
                        f"{study_name}_win",
                        int(n_tw),
                        timeout,
                        train_df,
                        val_df,
                        win_metric,
                        cfg_fixed,
                        lvl,
                    )
                    for n_tw in parts
                ]

                while futures:
                    done, not_done = wait(futures, timeout=0.5)
                    
                    # Update progress (best effort, don't block on DB locks)
                    try:
                        s = optuna.load_study(study_name=f"{study_name}_win", storage=storage_for_optuna)
                        n_completed = len(s.trials)
                        progress.update(task_win, completed=min(n_completed, int(n_trials)))
                    except Exception:
                        # Silently skip progress update on DB lock/timeout
                        pass
                    
                    # Process completed futures immediately
                    for f in done:
                        try:
                            n_done, err, _ = f.result()
                            if err:
                                failures_win += 1
                                console.print(f"[yellow]Worker error:[/yellow] {err}")
                        except Exception as exc:
                            failures_win += 1
                            console.print(f"[yellow]Worker raised:[/yellow] {exc}")
                    
                    futures = list(not_done)
                    
                    # Safety check: if no futures are done and all might be stuck, check progress
                    if not done and len(futures) > 0:
                        try:
                            # Quick check if we've hit the trial limit
                            s = optuna.load_study(study_name=f"{study_name}_win", storage=storage_for_optuna)
                            if len(s.trials) >= int(n_trials):
                                # Manually cancel remaining futures
                                for f in futures:
                                    f.cancel()
                                futures = []
                        except Exception:
                            pass

        dur_win = time.perf_counter() - start_win
        study_win = optuna.load_study(study_name=f"{study_name}_win", storage=storage_for_optuna)
        
        # Check if we have completed trials with values
        completed_trials = [t for t in study_win.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
        
        if not completed_trials:
            console.print("[bold red]Error: No completed trials with values found. All trials may have failed.[/bold red]")
            console.print(f"  Total trials: {len(study_win.trials)}")
            state_counts = {}
            for trial in study_win.trials:
                state_counts[trial.state.name] = state_counts.get(trial.state.name, 0) + 1
            console.print(f"  Trial states: {state_counts}")
            return
        
        # Multi-objective selection: prioritize PPG with sanity accuracy filter
        pareto_front = study_win.best_trials
        
        # 1. Optional sanity filter for minimum acceptable accuracy
        sane_trials = [t for t in pareto_front if t.values and len(t.values) >= 1 and float(t.values[0]) > 0.38]
        if not sane_trials:
            sane_trials = pareto_front

        if not sane_trials:
            console.print("[bold red]Error: No completed trials to select from. Aborting.[/bold red]")
            return
        
        # 2. Select by highest PPG (objective index 2)
        try:
            best_win = max(sane_trials, key=lambda t: float(t.values[2]) if (t.values and len(t.values) >= 3) else -1e9)
            console.print(f"\n[green]Selected Trial #{best_win.number} from the Pareto front.[/green]")
            console.print(f"  Accuracy: {best_win.values[0]:.4f}")
            console.print(f"  Log Loss: {best_win.values[1]:.4f}")
            console.print(f"  PPG: {best_win.values[2]:.4f}")
            console.print(f"  Predicted Draw Rate: {best_win.user_attrs.get('predicted_draw_rate', 0.0):.2%}")
        except Exception:
            best_win = None

        saved_path = None
        if best_win is not None:
            params_w = dict(best_win.params)
            saved_path = _save_best_params(
                {
                    "win_n_estimators": int(params_w.get("win_n_estimators", cfg_base.model.win_n_estimators)),
                    "win_max_depth": int(params_w.get("win_max_depth", cfg_base.model.win_max_depth)),
                    "win_learning_rate": float(params_w.get("win_learning_rate", cfg_base.model.win_learning_rate)),
                    "win_subsample": float(params_w.get("win_subsample", cfg_base.model.win_subsample)),
                    "win_colsample_bytree": float(params_w.get("win_colsample_bytree", cfg_base.model.win_colsample_bytree)),
                }
            )

    tbl = Table(title="[magenta]Phase 2 Complete: Win Classifier (Multi-Objective)[/magenta]", show_lines=False, box=None)
    tbl.add_column("Metric", style="dim")
    tbl.add_column("Value", style="bold")
    if best_win is not None:
        tbl.add_row("Accuracy", f"[green]{best_win.user_attrs.get('accuracy', 0.0):.4f}[/green]")
        tbl.add_row("Log Loss", f"[yellow]{best_win.user_attrs.get('log_loss', 0.0):.4f}[/yellow]")
        tbl.add_row("PPG", f"[green]{best_win.user_attrs.get('ppg', best_win.values[2] if best_win.values and len(best_win.values) >= 3 else 0.0):.4f}[/green]")
        tbl.add_row("Pred. Draw Rate", f"[cyan]{best_win.user_attrs.get('predicted_draw_rate', 0.0):.2%}[/cyan]")
    tbl.add_row("Trials", f"{len(study_win.trials)}")
    tbl.add_row("Duration", f"[cyan]{dur_win:.1f}s[/cyan]")
    if bench_avg_win is not None:
        est_seq_total = bench_avg_win * n_trials
        speedup = est_seq_total / max(1e-9, dur_win)
        tbl.add_row("Avg/trial", f"{bench_avg_win:.3f}s")
        tbl.add_row("Estimated seq", f"{est_seq_total:.1f}s")
        tbl.add_row("Speedup", f"[green]{speedup:.2f}x[/green]")
    status = "[green]✓ OK" if failures_win == 0 else f"[red]✗ Failed ({failures_win})[/red]"
    tbl.add_row("Status", status)
    console.print(tbl)

    summary_dir = cfg_base.paths.data_dir / "optuna"
    os.makedirs(summary_dir, exist_ok=True)
    try:
        import yaml  # type: ignore
        with open(summary_dir / f"{study_name}_win_summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "study_name": f"{study_name}_win",
                    "storage": storage_url,
                    "n_trials": len(study_win.trials),
                    "best_trial": None if best_win is None else {
                        "number": best_win.number,
                        "values": best_win.values if hasattr(best_win, 'values') else None,
                        "accuracy": float(best_win.user_attrs.get("accuracy", 0.0)),
                        "log_loss": float(best_win.user_attrs.get("log_loss", 0.0)),
                        "ppg": float(best_win.values[2]) if hasattr(best_win, 'values') and len(best_win.values) >= 3 else None,
                        "predicted_draw_rate": float(best_win.user_attrs.get("predicted_draw_rate", 0.0)),
                        "params": best_win.params,
                        "multi_objective": True,
                    },
                    "updated_file": None if saved_path is None else str(saved_path),
                    "duration_seconds": float(dur_win),
                    "workers": int(w),
                },
                f,
                sort_keys=False,
                indent=2,
            )
    except Exception:
        pass

    console.rule("[bold green]TUNING COMPLETE[/bold green]")