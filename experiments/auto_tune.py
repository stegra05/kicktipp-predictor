#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning with selectable objectives (PPG, logloss, brier, etc.).
Saves per-objective best params (config/best_params_<objective>.yaml) and the winner to config/best_params.yaml.
Removes the Optuna SQLite storage database after run to start fresh each time.
"""

import argparse
import functools
import os
import sys
import time

# Hard-cap BLAS/OpenMP threads before importing numpy/xgboost to avoid fork/thread storms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XGBOOST_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rich import box

# Rich console imports
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Ensure project root is on sys.path so `src.*` imports work when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from urllib.parse import unquote, urlparse

from kicktipp_predictor.config import get_config, reset_config
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.evaluate import (
    brier_score_multiclass,
    log_loss_multiclass,
    ranked_probability_score_3c,
)
from kicktipp_predictor.metrics import compute_points
from kicktipp_predictor.predictor import MatchPredictor

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception as _e:  # pragma: no cover
    optuna = None


def _retry_on_database_lock(max_retries: int = 10, delay: float = 0.1):
    """Decorator to retry database operations on SQLite lock errors with improved backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Check if it's a database lock error
                    if (
                        "database is locked" in str(e).lower()
                        or "sqlite3.OperationalError" in str(type(e))
                        or "StorageInternalError" in str(type(e))
                        or "OperationalError" in str(type(e))
                        or "sqlite3.DatabaseError" in str(type(e))
                        or "sqlite3.IntegrityError" in str(type(e))
                    ):
                        if attempt < max_retries - 1:
                            # More aggressive backoff with jitter to reduce thundering herd
                            import random

                            jitter = random.uniform(0.1, 0.5)
                            backoff_delay = delay * (2.0**attempt) + jitter
                            time.sleep(min(backoff_delay, 30.0))  # Cap at 30 seconds
                            continue
                    # If it's not a database lock error, re-raise immediately
                    raise
            # If we've exhausted all retries, raise the last exception
            raise last_exception

        return wrapper

    return decorator


def _objective_direction(objective: str) -> str:
    obj = (objective or "").lower()
    if obj in ("logloss", "brier", "rps"):
        return "minimize"
    return "maximize"


def _sqlite_fs_path(storage: str | None) -> str | None:
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


def _configure_sqlite_for_concurrency(storage: str) -> str:
    """Configure SQLite storage URL for better concurrency with multiple workers."""
    if not storage or not storage.startswith("sqlite:"):
        return storage

    # Parse the URL to add/update parameters
    parsed = urlparse(storage)
    query_params = {}

    # Parse existing query parameters
    if parsed.query:
        for param in parsed.query.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                query_params[key] = value

    # Set optimal parameters for high concurrency
    query_params.update(
        {
            "timeout": "600",  # 10 minutes timeout for high concurrency
            "check_same_thread": "false",  # Allow different threads
            "isolation_level": "IMMEDIATE",  # Better for concurrent writes
            "cache_size": "10000",  # Larger cache
            "synchronous": "NORMAL",  # Balance between safety and speed
            "journal_mode": "WAL",  # Write-Ahead Logging for better concurrency
            "temp_store": "MEMORY",  # Use memory for temp tables
            "mmap_size": "268435456",  # 256MB memory-mapped I/O
        }
    )

    # Rebuild the URL
    new_query = "&".join(f"{k}={v}" for k, v in query_params.items())
    new_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

    return new_url


def _enable_sqlite_wal_mode(storage: str) -> None:
    """Enable WAL mode for SQLite database to improve concurrency."""
    if not storage or not storage.startswith("sqlite:"):
        return

    try:
        import sqlite3
        from urllib.parse import unquote, urlparse

        # Extract database path
        parsed = urlparse(storage)
        db_path = unquote(parsed.path or "")
        if not db_path:
            return

        # Handle Windows drive letter like /C:/path.db
        if len(db_path) >= 3 and db_path[0] == "/" and db_path[2] == ":":
            db_path = db_path[1:]
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Enable WAL mode and optimize for high concurrency
        with sqlite3.connect(db_path, timeout=60) as conn:
            # Set WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.execute(
                "PRAGMA wal_autocheckpoint=1000"
            )  # Checkpoint every 1000 pages
            conn.execute("PRAGMA busy_timeout=600000")  # 10 minutes busy timeout
            conn.execute(
                "PRAGMA foreign_keys=OFF"
            )  # Disable foreign key checks for speed
            conn.execute("PRAGMA locking_mode=NORMAL")  # Allow shared locks
            conn.commit()
    except Exception:
        # If WAL mode setup fails, continue without it
        pass


# Deprecated: use compute_points from metrics for vectorized points


def _apply_params_to_config(params: dict[str, float]) -> None:
    """Mutate the global Config instance with trial parameters."""
    cfg = get_config()
    # Core
    if "draw_boost" in params:
        cfg.model.draw_boost = float(params["draw_boost"])
    if "min_lambda" in params:
        cfg.model.min_lambda = float(params["min_lambda"])
    if "time_decay_half_life_days" in params:
        cfg.model.time_decay_half_life_days = float(params["time_decay_half_life_days"])
        cfg.model.use_time_decay = True
    # Optional feature knobs (recognized if present)
    if "form_last_n" in params:
        try:
            cfg.model.form_last_n = int(params["form_last_n"])
        except Exception:
            cfg.model.form_last_n = int(float(params["form_last_n"]))
    if "momentum_decay" in params:
        cfg.model.momentum_decay = float(params["momentum_decay"])
    # New value weighting and decision knobs
    if "value_weight_2pt" in params:
        cfg.model.value_weight_2pt = float(params["value_weight_2pt"])
    if "value_weight_3pt" in params:
        cfg.model.value_weight_3pt = float(params["value_weight_3pt"])
    if "value_weight_4pt" in params:
        cfg.model.value_weight_4pt = float(params["value_weight_4pt"])
    if "confidence_shift_threshold" in params:
        cfg.model.confidence_shift_threshold = float(
            params["confidence_shift_threshold"]
        )
    if "confidence_shift_prob_ratio" in params:
        cfg.model.confidence_shift_prob_ratio = float(
            params["confidence_shift_prob_ratio"]
        )
    # Outcome classifier
    if "outcome_n_estimators" in params:
        cfg.model.outcome_n_estimators = int(params["outcome_n_estimators"])
    if "outcome_max_depth" in params:
        cfg.model.outcome_max_depth = int(params["outcome_max_depth"])
    if "outcome_learning_rate" in params:
        cfg.model.outcome_learning_rate = float(params["outcome_learning_rate"])
    if "outcome_subsample" in params:
        cfg.model.outcome_subsample = float(params["outcome_subsample"])
    if "outcome_colsample_bytree" in params:
        cfg.model.outcome_colsample_bytree = float(params["outcome_colsample_bytree"])
    if "outcome_reg_alpha" in params:
        cfg.model.outcome_reg_alpha = float(params["outcome_reg_alpha"])
    if "outcome_reg_lambda" in params:
        cfg.model.outcome_reg_lambda = float(params["outcome_reg_lambda"])
    if "outcome_gamma" in params:
        cfg.model.outcome_gamma = float(params["outcome_gamma"])
    if "outcome_min_child_weight" in params:
        cfg.model.outcome_min_child_weight = float(params["outcome_min_child_weight"])

    # Post-processing probabilities
    if "proba_temperature" in params:
        cfg.model.proba_temperature = float(params["proba_temperature"])
    if "prior_blend_alpha" in params:
        cfg.model.prior_blend_alpha = float(params["prior_blend_alpha"])
    if "prob_source" in params:
        cfg.model.prob_source = str(params["prob_source"]).strip().lower()
    if "hybrid_poisson_weight" in params:
        cfg.model.hybrid_poisson_weight = float(params["hybrid_poisson_weight"])
    if "hybrid_scheme" in params:
        cfg.model.hybrid_scheme = str(params["hybrid_scheme"]).strip().lower()
    if "hybrid_entropy_w_min" in params:
        cfg.model.hybrid_entropy_w_min = float(params["hybrid_entropy_w_min"])
    if "hybrid_entropy_w_max" in params:
        cfg.model.hybrid_entropy_w_max = float(params["hybrid_entropy_w_max"])
    if "poisson_joint" in params:
        cfg.model.poisson_joint = str(params["poisson_joint"]).strip().lower()
    if "dixon_coles_rho" in params:
        cfg.model.dixon_coles_rho = float(params["dixon_coles_rho"])
    if "proba_grid_max_goals" in params:
        cfg.model.proba_grid_max_goals = int(params["proba_grid_max_goals"])
    if "poisson_draw_rho" in params:
        cfg.model.poisson_draw_rho = float(params["poisson_draw_rho"])

    # Goal regressors
    if "goals_n_estimators" in params:
        cfg.model.goals_n_estimators = int(params["goals_n_estimators"])
    if "goals_max_depth" in params:
        cfg.model.goals_max_depth = int(params["goals_max_depth"])
    if "goals_learning_rate" in params:
        cfg.model.goals_learning_rate = float(params["goals_learning_rate"])
    if "goals_subsample" in params:
        cfg.model.goals_subsample = float(params["goals_subsample"])
    if "goals_colsample_bytree" in params:
        cfg.model.goals_colsample_bytree = float(params["goals_colsample_bytree"])
    if "goals_reg_alpha" in params:
        cfg.model.goals_reg_alpha = float(params["goals_reg_alpha"])
    if "goals_reg_lambda" in params:
        cfg.model.goals_reg_lambda = float(params["goals_reg_lambda"])
    if "goals_gamma" in params:
        cfg.model.goals_gamma = float(params["goals_gamma"])
    if "goals_min_child_weight" in params:
        cfg.model.goals_min_child_weight = float(params["goals_min_child_weight"])


def _objective_builder(
    base_features_df,
    all_matches,
    folds: list[tuple[np.ndarray, np.ndarray]],
    objective_name: str,
    direction: str,
    omp_threads: int,
    verbose: bool,
    console: Console,
):
    # Cache features by feature-knob tuple to avoid recomputation across trials
    features_cache: dict[tuple, any] = {}

    def obj_fn(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
        # Limit BLAS/OMP threads to avoid oversubscription
        if omp_threads and omp_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(omp_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
            os.environ["MKL_NUM_THREADS"] = str(omp_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)
        # Track worker and timing for per-trial metadata
        try:
            wid = int(os.environ.get("OPTUNA_WORKER_ID", "0"))
        except Exception:
            wid = 0
        t_start = time.time()
        # Search space (with retry logic for database locks)
        @_retry_on_database_lock(max_retries=3, delay=0.5)
        def suggest_params():
            return {
                "draw_boost": trial.suggest_float("draw_boost", 0.8, 5.0, step=0.1),
                "time_decay_half_life_days": trial.suggest_float(
                    "time_decay_half_life_days", 15.0, 365.0
                ),
                "outcome_n_estimators": trial.suggest_int(
                    "outcome_n_estimators", 100, 1500, step=50
                ),
                "outcome_gamma": trial.suggest_float("outcome_gamma", 1e-8, 10.0, log=True),
                "outcome_min_child_weight": trial.suggest_float(
                    "outcome_min_child_weight", 0.1, 10.0, log=True
                ),
                "goals_min_child_weight": trial.suggest_float(
                    "goals_min_child_weight", 0.1, 10.0, log=True
                ),
                "momentum_decay": trial.suggest_float("momentum_decay", 0.50, 0.99, step=0.01),
                "hybrid_poisson_weight": trial.suggest_float("hybrid_poisson_weight", 0.0, 0.25),
            }

        params: dict[str, float] = suggest_params()

        fold_metrics: list[float] = []

        # Determine which feature set to use for this trial
        # Apply params to config now so DataLoader sees knobs
        reset_config()
        _apply_params_to_config(params)

        # Cache key by (form_last_n, momentum_decay)
        cfg = get_config()
        key = (
            int(getattr(cfg.model, "form_last_n", 5)),
            float(round(getattr(cfg.model, "momentum_decay", 0.9), 3)),
        )

        if key not in features_cache:
            # Recompute features for this knob combo
            dl = DataLoader()
            if verbose:
                _safe_print(
                    console,
                    f"[FEATS] Building features for knobs form_last_n={key[0]} momentum_decay={key[1]}...",
                    f"[dim][FEATS] Building features for knobs form_last_n={key[0]} momentum_decay={key[1]}...[/dim]",
                )
            feats_df = dl.create_features_from_matches(all_matches)
            if "date" in feats_df.columns:
                feats_df = feats_df.sort_values("date").reset_index(drop=True)
            features_cache[key] = feats_df
        features_df = features_cache[key]

        for train_idx, test_idx in folds:
            # Reset and apply params
            reset_config()
            _apply_params_to_config(params)

            # Prepare data
            train_df = features_df.iloc[train_idx]
            test_df = features_df.iloc[test_idx]
            test_feats = test_df.drop(
                columns=["home_score", "away_score", "goal_difference", "result"],
                errors="ignore",
            )

            predictor = MatchPredictor(quiet=not verbose)
            predictor.train(train_df)
            preds = predictor.predict(test_feats)

            # Build per-match points vector
            ph = np.array(
                [int(p.get("predicted_home_score", 0)) for p in preds], dtype=int
            )
            pa = np.array(
                [int(p.get("predicted_away_score", 0)) for p in preds], dtype=int
            )
            ah = np.asarray(test_df["home_score"], dtype=int)
            aa = np.asarray(test_df["away_score"], dtype=int)
            points_vec = compute_points(ph, pa, ah, aa).astype(float)

            # Recency weights for validation fold
            if "date" in test_df.columns:
                fold_dates = pd.to_datetime(test_df["date"])
                days_old = (fold_dates.max() - fold_dates).dt.days.astype(float)
                half_life = float(params["time_decay_half_life_days"])
                decay_rate = np.log(2.0) / max(1.0, half_life)
                weights = np.exp(-decay_rate * days_old.values)
            else:
                weights = np.ones_like(points_vec, dtype=float)

            # Build proba matrix and labels
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
            y_true = test_df["result"].tolist()
            y_pred_idx = np.argmax(proba, axis=1) if len(proba) else np.array([])
            idx_to_label = {0: "H", 1: "D", 2: "A"}
            y_pred = [idx_to_label.get(int(i), "H") for i in y_pred_idx]

            # Force optimization toward weighted PPG as primary target
            obj = "ppg"
            if obj == "ppg":
                metric_val = float(
                    np.sum(points_vec * weights) / max(1.0, np.sum(weights))
                )
            elif obj == "ppg_unweighted":
                metric_val = float(np.mean(points_vec)) if len(points_vec) else 0.0
            elif obj == "logloss":
                metric_val = log_loss_multiclass(y_true, proba)
            elif obj == "brier":
                metric_val = brier_score_multiclass(y_true, proba)
            elif obj == "rps":
                metric_val = ranked_probability_score_3c(y_true, proba)
            elif obj == "balanced_accuracy":
                try:
                    metric_val = float(
                        balanced_accuracy_score(y_true, y_pred, sample_weight=weights)
                    )
                except Exception:
                    metric_val = float("nan")
            elif obj == "accuracy":
                try:
                    metric_val = float(
                        accuracy_score(y_true, y_pred, sample_weight=weights)
                    )
                except Exception:
                    metric_val = float("nan")
            else:
                metric_val = float(
                    np.sum(points_vec * weights) / max(1.0, np.sum(weights))
                )

            if verbose:
                if obj in ("ppg", "ppg_unweighted"):
                    w_ppg = float(
                        np.sum(points_vec * weights) / max(1.0, np.sum(weights))
                    )
                    u_ppg = float(np.mean(points_vec)) if len(points_vec) else 0.0
                    _safe_print(
                        console,
                        f"[FOLD] ppg_w={w_ppg:.4f} ppg={u_ppg:.4f} n={len(points_vec)}",
                        f"[dim][FOLD] ppg_w={w_ppg:.4f} ppg={u_ppg:.4f} n={len(points_vec)}[/dim]",
                    )
                else:
                    _safe_print(
                        console,
                        f"[FOLD] {obj}={metric_val:.6f} n={len(points_vec)}",
                        f"[dim][FOLD] {obj}={metric_val:.6f} n={len(points_vec)}[/dim]",
                    )

            fold_metrics.append(metric_val)

        # Objective: average of fold metrics
        if not fold_metrics:
            # Record trial metadata even on empty folds
            try:
                trial.set_user_attr("worker_id", wid)
                trial.set_user_attr("duration_sec", max(0.0, time.time() - t_start))
                trial.set_user_attr("finished_at", int(time.time()))
            except Exception:
                pass
            return float("inf") if direction == "minimize" else float("-inf")
        # Record per-trial metadata for dashboard
        try:
            trial.set_user_attr("worker_id", wid)
            trial.set_user_attr("duration_sec", max(0.0, time.time() - t_start))
            trial.set_user_attr("finished_at", int(time.time()))
        except Exception:
            pass
        return float(np.nanmean(fold_metrics))

    return obj_fn


def _safe_print(console, message: str, rich_message: str = None):
    """Print message using Rich console if available, otherwise simple print."""
    if console is not None:
        console.print(rich_message if rich_message else message)
    else:
        print(message)


def main():
    # Check if we're in multi-worker mode by looking at environment variables
    # When using --workers > 1, each worker process will have its own console
    is_coordinated = os.environ.get("KTP_TUNE_COORDINATED", "0") == "1"
    has_worker_id = os.environ.get("OPTUNA_WORKER_ID") is not None
    is_multiworker = is_coordinated or has_worker_id

    # Add staggered startup for multi-worker mode to reduce database contention
    if is_multiworker:
        worker_id = int(os.environ.get("OPTUNA_WORKER_ID", "0"))
        total_workers = int(os.environ.get("OPTUNA_TOTAL_WORKERS", "1"))
        if worker_id > 0:  # Don't delay the first worker
            # Stagger startup: each worker waits a bit longer
            import random

            delay = random.uniform(0.1, 0.5) + (worker_id * 0.1)
            time.sleep(delay)

    # Initialize Rich console only for single-worker mode
    if not is_multiworker:
        console = Console()
        # Display welcome banner
        console.print(
            Panel.fit(
                "[bold blue]Kicktipp Predictor Hyperparameter Tuning[/bold blue]\n"
                "[dim]Optuna-based optimization with Rich console output[/dim]",
                border_style="blue",
            )
        )
    else:
        # Suppress ALL output for multi-worker mode to prevent console chaos
        # The CLI will handle progress monitoring and final output
        console = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials")
    parser.add_argument("--n-splits", type=int, default=3, help="TimeSeriesSplit folds")
    parser.add_argument(
        "--omp-threads", type=int, default=1, help="Threads per worker for BLAS/OMP"
    )
    # Removed final training options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs from inner training loop",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///study.db) for multi-process tuning",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (used with --storage)",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["none", "median", "hyperband"],
        default="median",
        help="Enable trial pruning strategy",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=20,
        help="Trials before enabling pruning (median pruner)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=[
            "ppg",
            "ppg_unweighted",
            "logloss",
            "brier",
            "balanced_accuracy",
            "accuracy",
            "rps",
        ],
        default="ppg",
        help="Tuning objective",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["auto", "maximize", "minimize"],
        default="auto",
        help="Study direction; auto selects based on objective",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Comma-separated list of objectives to compare; when set, --objective is ignored",
    )
    args = parser.parse_args()

    # Propagate verbosity to submodules via environment variable
    os.environ["KTP_VERBOSE"] = "1" if args.verbose else "0"

    if optuna is None:
        _safe_print(
            console,
            "❌ Optuna is not installed. Please install optuna to run tuning.",
            "[red]❌ Optuna is not installed. Please install optuna to run tuning.[/red]",
        )
        sys.exit(1)

    # Limit threads in parent
    if args.omp_threads and args.omp_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.omp_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.omp_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.omp_threads))

    # Load data with appropriate progress display
    if not is_multiworker:
        _safe_print(
            console, "\n📊 Data Loading Phase", "\n[bold]📊 Data Loading Phase[/bold]"
        )

    if console is not None:
        # Rich progress for single-worker mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Load historical data
            task1 = progress.add_task("Loading historical data...", total=None)
            data_loader = DataLoader()
            current_season = data_loader.get_current_season()
            start_season = current_season - 2
            all_matches = data_loader.fetch_historical_seasons(
                start_season, current_season
            )
            progress.update(task1, description=f"✅ Loaded {len(all_matches)} matches")

            # Create features
            task2 = progress.add_task("Creating features...", total=None)
            features_df = data_loader.create_features_from_matches(all_matches)
            progress.update(task2, description=f"✅ Created {len(features_df)} samples")
    else:
        # Silent loading for multi-worker mode to prevent console chaos
        data_loader = DataLoader()
        current_season = data_loader.get_current_season()
        start_season = current_season - 2
        all_matches = data_loader.fetch_historical_seasons(start_season, current_season)
        features_df = data_loader.create_features_from_matches(all_matches)

    # Only show data loading summary in single-worker mode
    if not is_multiworker:
        _safe_print(
            console,
            f"✓ Data loaded: {len(all_matches)} matches → {len(features_df)} samples",
            f"[green]✓[/green] Data loaded: [bold]{len(all_matches)}[/bold] matches → [bold]{len(features_df)}[/bold] samples",
        )

    # Build and run study
    def _fmt_dur(sec: float) -> str:
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    total_trials = int(max(0, args.n_trials))
    total_cv_trainings = total_trials * int(max(1, args.n_splits))

    # Verbosity reflects CLI flag only (quiet by default, even with multiple jobs)
    effective_verbose = bool(args.verbose) and not is_multiworker

    # Ensure chronological order for time-series CV
    if "date" in features_df.columns:
        features_df = features_df.sort_values("date").reset_index(drop=True)
    # Precompute CV folds once to ensure identical splits across objectives
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = list(tscv.split(features_df))

    # Determine objectives to run
    if args.compare:
        objectives_to_run = [o.strip() for o in args.compare.split(",") if o.strip()]
    else:
        objectives_to_run = [args.objective]

    # Display configuration only in single-worker mode
    if not is_multiworker:
        _safe_print(console, "\n⚙️ Configuration", "\n[bold]⚙️ Configuration[/bold]")

        if console is not None:
            # Rich table for single-worker mode
            config_table = Table(title="Tuning Configuration", box=box.ROUNDED)
            config_table.add_column("Parameter", style="cyan", no_wrap=True)
            config_table.add_column("Value", style="magenta")

            config_table.add_row("Total Trials", str(total_trials))
            config_table.add_row("CV Splits", str(args.n_splits))
            config_table.add_row("Total CV Trainings", str(total_cv_trainings))
            config_table.add_row("OMP Threads", str(args.omp_threads or 1))
            config_table.add_row("Pruner", args.pruner)
            config_table.add_row("Storage", args.storage or "Memory")
            config_table.add_row("Objectives", ", ".join(objectives_to_run))
            config_table.add_row("Mode", "Compare" if args.compare else "Single")

            console.print(config_table)

        if args.compare:
            _safe_print(
                console,
                f"⚠️ Compare mode will take approximately {len(objectives_to_run)}× the time of a single run.",
                f"[yellow]⚠️[/yellow] Compare mode will take approximately [bold]{len(objectives_to_run)}×[/bold] the time of a single run.",
            )

    # Configure pruner
    pruner = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(max(0, args.pruner_startup_trials))
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()

    # Track best per objective
    per_objective_best: dict[str, dict[str, object]] = {}
    storage_fs_path = _sqlite_fs_path(args.storage)

    try:
        for obj in objectives_to_run:
            study_direction = (
                args.direction
                if args.direction != "auto"
                else _objective_direction(obj)
            )
            objective_fn = _objective_builder(
                features_df,
                all_matches,
                folds,
                obj,
                study_direction,
                args.omp_threads,
                effective_verbose,
                console,
            )

            # Create study, optionally with storage (with retry logic for database locks)
            @_retry_on_database_lock(max_retries=5, delay=2.0)
            def create_study():
                if args.storage:
                    # Configure SQLite for better concurrency
                    storage_url = _configure_sqlite_for_concurrency(args.storage)
                    # Enable WAL mode for better concurrency
                    _enable_sqlite_wal_mode(storage_url)
                    return optuna.create_study(
                        direction=study_direction,
                        storage=storage_url,
                        study_name=((args.study_name or "kicktipp-tune") + f"-{obj}"),
                        load_if_exists=True,
                        pruner=pruner,
                    )
                else:
                    return optuna.create_study(direction=study_direction, pruner=pruner)

            study = create_study()

            # Rich progress tracking
            progress_data = {
                "start": time.time(),
                "completed": 0,
                "best_value": float("-inf")
                if study_direction == "maximize"
                else float("inf"),
                "best_trial": None,
            }

            def _progress_cb(
                study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
            ) -> None:  # type: ignore[name-defined]
                progress_data["completed"] += 1
                try:
                    progress_data["best_value"] = float(study.best_value)
                    progress_data["best_trial"] = int(study.best_trial.number)
                except Exception:
                    pass

            # Only show objective info in single-worker mode
            if not is_multiworker:
                _safe_print(
                    console,
                    f"\n🎯 Optimizing Objective: {obj}",
                    f"\n[bold]🎯 Optimizing Objective: [cyan]{obj}[/cyan][/bold]",
                )
                _safe_print(
                    console,
                    f"Direction: {study_direction} | Trials: {args.n_trials} | CV Splits: {args.n_splits}",
                    f"[dim]Direction: {study_direction} | Trials: {args.n_trials} | CV Splits: {args.n_splits}[/dim]",
                )

            start = time.time()

            # Run optimization with simple progress display (with retry logic for database locks)
            @_retry_on_database_lock(max_retries=3, delay=1.0)
            def run_optimization():
                study.optimize(
                    objective_fn,
                    n_trials=args.n_trials,
                    n_jobs=1,
                    callbacks=[_progress_cb],
                    show_progress_bar=False,
                )

            run_optimization()

            duration = time.time() - start
            # Only show completion messages in single-worker mode
            if not is_multiworker:
                try:
                    _safe_print(
                        console,
                        f"✅ Study complete in {duration:.1f}s. Best value: {study.best_value:.6f} ({study_direction})",
                        f"[green]✅[/green] Study complete in [bold]{duration:.1f}s[/bold]. Best value: [bold]{study.best_value:.6f}[/bold] ({study_direction})",
                    )
                except Exception:
                    _safe_print(
                        console,
                        f"⚠️ Study complete in {duration:.1f}s. No completed trials.",
                        f"[yellow]⚠️[/yellow] Study complete in [bold]{duration:.1f}s[/bold]. No completed trials.",
                    )

            try:
                completed_trials = study.get_trials(
                    deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
                )  # type: ignore[attr-defined]
            except Exception:
                completed_trials = []
            if not completed_trials:
                if not is_multiworker:
                    _safe_print(
                        console,
                        f"⚠️ No completed trials for objective '{obj}'; skipping save.",
                        f"[yellow]⚠️[/yellow] No completed trials for objective '{obj}'; skipping save.",
                    )
                continue

            best_params = dict(study.best_params)
            per_objective_best[obj] = {
                "best_value": float(study.best_value),
                "direction": study_direction,
                "params": best_params,
            }

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            cfg_dir = os.path.join(project_root, "config")
            os.makedirs(cfg_dir, exist_ok=True)
            if yaml is not None:
                with open(
                    os.path.join(cfg_dir, f"best_params_{obj}.yaml"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    yaml.safe_dump(best_params, f, sort_keys=True)
            else:
                import json

                with open(
                    os.path.join(cfg_dir, f"best_params_{obj}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(best_params, f, indent=2)
            # Only show save messages in single-worker mode
            if not is_multiworker:
                _safe_print(
                    console,
                    f"💾 Best params saved to config/best_params_{obj}.yaml",
                    f"[green]💾[/green] Best params saved to [bold]config/best_params_{obj}.yaml[/bold]",
                )

        # If we have results, re-evaluate on identical folds and choose winner by weighted PPG
        if not per_objective_best:
            if not is_multiworker:
                _safe_print(
                    console,
                    "❌ No completed trials across objectives; exiting.",
                    "[red]❌[/red] No completed trials across objectives; exiting.",
                )
            return

        summaries: dict[str, dict[str, float]] = {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for obj, info in per_objective_best.items():
            params = info["params"]  # type: ignore[assignment]
            reset_config()
            _apply_params_to_config(params)  # type: ignore[arg-type]
            # Recompute features for this params combo (ensures feature knobs take effect)
            dl = DataLoader()
            feats_df = dl.create_features_from_matches(all_matches)
            if "date" in feats_df.columns:
                feats_df = feats_df.sort_values("date").reset_index(drop=True)

            ppg_w_list: list[float] = []
            ppg_u_list: list[float] = []
            acc_w_list: list[float] = []
            bacc_w_list: list[float] = []
            brier_list: list[float] = []
            logloss_list: list[float] = []
            rps_list: list[float] = []

            for train_idx, test_idx in folds:
                reset_config()
                _apply_params_to_config(params)  # type: ignore[arg-type]
                train_df = feats_df.iloc[train_idx]
                test_df = feats_df.iloc[test_idx]
                test_feats = test_df.drop(
                    columns=["home_score", "away_score", "goal_difference", "result"],
                    errors="ignore",
                )
                predictor = MatchPredictor(quiet=not effective_verbose)
                predictor.train(train_df)
                preds = predictor.predict(test_feats)

                ph = np.array(
                    [int(p.get("predicted_home_score", 0)) for p in preds], dtype=int
                )
                pa = np.array(
                    [int(p.get("predicted_away_score", 0)) for p in preds], dtype=int
                )
                ah = np.asarray(test_df["home_score"], dtype=int)
                aa = np.asarray(test_df["away_score"], dtype=int)
                points_vec = compute_points(ph, pa, ah, aa).astype(float)

                if "date" in test_df.columns:
                    fold_dates = pd.to_datetime(test_df["date"])
                    days_old = (fold_dates.max() - fold_dates).dt.days.astype(float)
                    half_life = float(params.get("time_decay_half_life_days", 90.0))  # type: ignore[union-attr]
                    decay_rate = np.log(2.0) / max(1.0, half_life)
                    weights = np.exp(-decay_rate * days_old.values)
                else:
                    weights = np.ones_like(points_vec, dtype=float)

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
                y_true = test_df["result"].tolist()
                y_pred_idx = np.argmax(proba, axis=1) if len(proba) else np.array([])
                idx_to_label = {0: "H", 1: "D", 2: "A"}
                y_pred = [idx_to_label.get(int(i), "H") for i in y_pred_idx]

                ppg_w_list.append(
                    float(np.sum(points_vec * weights) / max(1.0, np.sum(weights)))
                )
                ppg_u_list.append(
                    float(np.mean(points_vec)) if len(points_vec) else 0.0
                )
                try:
                    acc_w_list.append(
                        float(accuracy_score(y_true, y_pred, sample_weight=weights))
                    )
                except Exception:
                    acc_w_list.append(float("nan"))
                try:
                    bacc_w_list.append(
                        float(
                            balanced_accuracy_score(
                                y_true, y_pred, sample_weight=weights
                            )
                        )
                    )
                except Exception:
                    bacc_w_list.append(float("nan"))
                brier_list.append(brier_score_multiclass(y_true, proba))
                logloss_list.append(log_loss_multiclass(y_true, proba))
                rps_list.append(ranked_probability_score_3c(y_true, proba))

            summaries[obj] = {
                "ppg_weighted": float(np.nanmean(ppg_w_list))
                if ppg_w_list
                else float("nan"),
                "ppg_unweighted": float(np.nanmean(ppg_u_list))
                if ppg_u_list
                else float("nan"),
                "accuracy_weighted": float(np.nanmean(acc_w_list))
                if acc_w_list
                else float("nan"),
                "balanced_accuracy_weighted": float(np.nanmean(bacc_w_list))
                if bacc_w_list
                else float("nan"),
                "brier": float(np.nanmean(brier_list)) if brier_list else float("nan"),
                "log_loss": float(np.nanmean(logloss_list))
                if logloss_list
                else float("nan"),
                "rps": float(np.nanmean(rps_list)) if rps_list else float("nan"),
            }

        # Display results summary only in single-worker mode
        if not is_multiworker:
            _safe_print(
                console, "\n📊 Results Summary", "\n[bold]📊 Results Summary[/bold]"
            )

            if console is not None:
                # Rich table for single-worker mode
                results_table = Table(
                    title="Objective Comparison Results", box=box.ROUNDED
                )
                results_table.add_column("Objective", style="cyan", no_wrap=True)
                results_table.add_column("PPG Weighted", justify="right", style="green")
                results_table.add_column(
                    "PPG Unweighted", justify="right", style="green"
                )
                results_table.add_column("Accuracy", justify="right", style="blue")
                results_table.add_column("Balanced Acc", justify="right", style="blue")
                results_table.add_column("Brier Score", justify="right", style="red")
                results_table.add_column("Log Loss", justify="right", style="red")
                results_table.add_column("RPS", justify="right", style="red")

                for obj, summ in summaries.items():
                    results_table.add_row(
                        obj,
                        f"{summ['ppg_weighted']:.4f}",
                        f"{summ['ppg_unweighted']:.4f}",
                        f"{summ['accuracy_weighted']:.4f}",
                        f"{summ['balanced_accuracy_weighted']:.4f}",
                        f"{summ['brier']:.4f}",
                        f"{summ['log_loss']:.4f}",
                        f"{summ['rps']:.4f}",
                    )

                console.print(results_table)

        # Choose winner by highest weighted PPG
        winner = None
        best_ppg = float("-inf")
        for obj, summ in summaries.items():
            ppgw = summ.get("ppg_weighted", float("-inf"))
            if ppgw is not None and ppgw > best_ppg:
                best_ppg = ppgw
                winner = obj

        cfg_dir = os.path.join(project_root, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        if winner:
            win_params = per_objective_best[winner]["params"]  # type: ignore[index]
            if yaml is not None:
                with open(
                    os.path.join(cfg_dir, "best_params.yaml"), "w", encoding="utf-8"
                ) as f:
                    yaml.safe_dump(win_params, f, sort_keys=True)
            else:
                import json

                with open(
                    os.path.join(cfg_dir, "best_params.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(win_params, f, indent=2)
            # Only show winner announcement in single-worker mode
            if not is_multiworker:
                _safe_print(
                    console,
                    f"\n🏆 Winner: {winner}",
                    f"\n[bold green]🏆 Winner: [cyan]{winner}[/cyan][/bold green]",
                )
                _safe_print(
                    console,
                    "💾 Best parameters saved to config/best_params.yaml",
                    "[green]💾[/green] Best parameters saved to [bold]config/best_params.yaml[/bold]",
                )

        # Save summary artifacts
        out_dir = os.path.join(project_root, "data", "predictions")
        os.makedirs(out_dir, exist_ok=True)
        try:
            import json

            with open(
                os.path.join(out_dir, "metrics_tuning.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {"summaries": summaries, "per_objective_best": per_objective_best},
                    f,
                    indent=2,
                )
            header = "objective,ppg_weighted,ppg_unweighted,accuracy_weighted,balanced_accuracy_weighted,brier,log_loss,rps\n"
            lines = [header]
            for obj, summ in summaries.items():
                lines.append(
                    f"{obj},{summ['ppg_weighted']:.6f},{summ['ppg_unweighted']:.6f},{summ['accuracy_weighted']:.6f},{summ['balanced_accuracy_weighted']:.6f},{summ['brier']:.6f},{summ['log_loss']:.6f},{summ['rps']:.6f}\n"
                )
            with open(
                os.path.join(out_dir, "metrics_table_tuning.txt"), "w", encoding="utf-8"
            ) as f:
                f.writelines(lines)
        except Exception as _e:  # pragma: no cover
            if not is_multiworker:
                _safe_print(
                    console,
                    f"⚠️ Warning: Failed to write comparison metrics: {_e}",
                    f"[yellow]⚠️[/yellow] Warning: Failed to write comparison metrics: {_e}",
                )

    finally:
        # Delete SQLite storage file, if applicable, unless coordinated by external CLI
        try:
            coordinated = os.environ.get("KTP_TUNE_COORDINATED", "0") == "1"
            if (
                (not coordinated)
                and storage_fs_path
                and os.path.exists(storage_fs_path)
            ):
                os.remove(storage_fs_path)
                if not is_multiworker:
                    _safe_print(
                        console,
                        f"🗑️ Deleted Optuna storage DB at {storage_fs_path}",
                        f"[green]🗑️[/green] Deleted Optuna storage DB at [bold]{storage_fs_path}[/bold]",
                    )
        except Exception as _e:
            if not is_multiworker:
                _safe_print(
                    console,
                    f"⚠️ Warning: Failed to delete Optuna storage DB at {storage_fs_path}: {_e}",
                    f"[yellow]⚠️[/yellow] Warning: Failed to delete Optuna storage DB at {storage_fs_path}: {_e}",
                )

    # Final completion message - only show in single-worker mode
    if console is not None:
        console.print(
            Panel.fit(
                "[bold green]🎉 Hyperparameter Tuning Complete![/bold green]\n"
                "[dim]Check the config/ directory for best parameters and data/predictions/ for detailed results.[/dim]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
