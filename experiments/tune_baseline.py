#!/usr/bin/env python3
"""
Baseline Optuna tuning on a fixed train/test split focusing on PPG.

- Train: N seasons back up to the season before current
- Test: most recent full season (current_season - 1)
- Objective: maximize avg_points (PPG)
- Probability source: classifier-only (no calibration, no prior anchoring)
- Saves Optuna parameter importances and prints top levers

This script is designed to be launched via the CLI with single or multi-worker
coordination similar to experiments/auto_tune.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Limit BLAS/OpenMP threads to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XGBOOST_NUM_THREADS", "1")

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    from optuna.visualization import plot_param_importances
except Exception:
    optuna = None  # type: ignore

from urllib.parse import unquote, urlparse

from kicktipp_predictor.config import get_config, reset_config
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import MatchPredictor


def _is_primary_worker() -> bool:
    wid = os.environ.get("OPTUNA_WORKER_ID")
    if wid is None:
        return True
    try:
        return int(wid) == 0
    except Exception:
        return str(wid).strip() in ("0", "primary")


def _log(msg: str) -> None:
    if _is_primary_worker():
        print(msg)


# --- SQLite concurrency helpers (lightweight copies from auto_tune) ---
def _retry_on_database_lock(max_retries: int = 8, delay: float = 0.2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    s = str(e).lower()
                    if "database is locked" in s or "operationalerror" in s:
                        if attempt < max_retries - 1:
                            import random

                            time.sleep(
                                min(delay * (2**attempt) + random.random() * 0.3, 20.0)
                            )
                            continue
                    last = e
                    break
            if last is not None:
                raise last

        return wrapper

    return decorator


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
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def _configure_sqlite_for_concurrency(storage: str) -> str:
    if not storage or not storage.startswith("sqlite:"):
        return storage
    parsed = urlparse(storage)
    params: dict[str, str] = {}
    if parsed.query:
        for part in parsed.query.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
    params.update(
        {
            "timeout": "600",
            "check_same_thread": "false",
            "isolation_level": "IMMEDIATE",
            "cache_size": "10000",
            "synchronous": "NORMAL",
            "journal_mode": "WAL",
            "temp_store": "MEMORY",
            "mmap_size": "268435456",
        }
    )
    new_query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"


def _enable_sqlite_wal_mode(storage: str) -> None:
    if not storage or not storage.startswith("sqlite:"):
        return
    try:
        import sqlite3

        parsed = urlparse(storage)
        db_path = unquote(parsed.path or "")
        if not db_path:
            return
        if len(db_path) >= 3 and db_path[0] == "/" and db_path[2] == ":":
            db_path = db_path[1:]
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path, timeout=60) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.execute("PRAGMA wal_autocheckpoint=1000")
            conn.execute("PRAGMA busy_timeout=600000")
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("PRAGMA locking_mode=NORMAL")
            conn.commit()
    except Exception:
        pass


# --- Parameter application ---
def _apply_params_to_config(params: dict) -> None:
    cfg = get_config()
    # Baseline forcing
    cfg.model.prob_source = "classifier"

    # Core levers
    if "draw_boost" in params:
        cfg.model.draw_boost = float(
            params["draw_boost"]
        )  # class weight boost for draws

    if "max_goals" in params:
        cfg.model.max_goals = int(params["max_goals"])  # scoreline search grid
    if "time_decay_half_life_days" in params:
        cfg.model.time_decay_half_life_days = float(params["time_decay_half_life_days"])
        cfg.model.use_time_decay = True
    if "form_last_n" in params:
        cfg.model.form_last_n = int(params["form_last_n"])  # feature knob
    if "momentum_decay" in params:
        cfg.model.momentum_decay = float(params["momentum_decay"])  # feature knob

    # Classifier hyperparameters
    for k in (
        "outcome_n_estimators",
        "outcome_max_depth",
        "outcome_learning_rate",
        "outcome_subsample",
        "outcome_reg_lambda",
        "outcome_min_child_weight",
    ):
        if k in params:
            setattr(cfg.model, k, params[k])

    # Goals regressors hyperparameters
    for k in (
        "goals_n_estimators",
        "goals_max_depth",
        "goals_learning_rate",
        "goals_subsample",
        "goals_reg_lambda",
        "goals_min_child_weight",
    ):
        if k in params:
            setattr(cfg.model, k, params[k])

    # Probability shaping for classifier-only
    if "proba_temperature" in params:
        cfg.model.proba_temperature = float(
            params["proba_temperature"]
        )  # temperature scaling
    if "prior_blend_alpha" in params:
        cfg.model.prior_blend_alpha = float(
            params["prior_blend_alpha"]
        )  # anchor to prior


# --- Objective builder for fixed split ---
def build_objective(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features_cache: dict[tuple[int, float], tuple[pd.DataFrame, pd.DataFrame]],
    verbose: bool,
):
    def obj(trial: optuna.trial.Trial) -> float:
        # Thread limiting per trial
        os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
        os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
        os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")
        # Track worker and timing metadata
        try:
            wid = int(os.environ.get("OPTUNA_WORKER_ID", "0"))
        except Exception:
            wid = 0
        t_start = time.time()
        # Suggest parameter space focused on top-7 high-impact levers
        params = {
            # Class weighting & time-decay
            "draw_boost": trial.suggest_float("draw_boost", 0.8, 5.0, step=0.1),
            "time_decay_half_life_days": trial.suggest_float(
                "time_decay_half_life_days", 15.0, 365.0
            ),
            # Outcome XGB (selected levers)
            "outcome_n_estimators": trial.suggest_int(
                "outcome_n_estimators", 100, 1500, step=50
            ),
            "outcome_min_child_weight": trial.suggest_float(
                "outcome_min_child_weight", 0.1, 10.0, log=True
            ),
            # Goals XGB (selected representative lever)
            "goals_min_child_weight": trial.suggest_float(
                "goals_min_child_weight", 0.1, 10.0, log=True
            ),
            # Feature engineering knob
            "momentum_decay": trial.suggest_float(
                "momentum_decay", 0.50, 0.99, step=0.01
            ),
        }

        # Reset and apply parameters
        reset_config()
        _apply_params_to_config(params)

        # Rebuild features if feature knobs changed
        cfg = get_config()
        key = (
            int(getattr(cfg.model, "form_last_n", 5)),
            float(round(getattr(cfg.model, "momentum_decay", 0.9), 3)),
        )
        if key not in features_cache:
            dl = DataLoader()
            feats_tr = dl.create_features_from_matches(train_matches)
            feats_te = dl.create_features_from_matches(test_matches)
            # Keep chronological order if date present
            if "date" in feats_tr.columns:
                feats_tr = feats_tr.sort_values("date").reset_index(drop=True)
            if "date" in feats_te.columns:
                feats_te = feats_te.sort_values("date").reset_index(drop=True)
            features_cache[key] = (feats_tr, feats_te)
        feats_train, feats_test = features_cache[key]

        # Train & evaluate
        predictor = MatchPredictor(quiet=not verbose)
        predictor.train(feats_train)
        metrics = predictor.evaluate(feats_test)
        ppg = float(metrics.get("avg_points", float("nan")))
        if not (ppg == ppg):  # NaN check
            raise optuna.TrialPruned("NaN PPG")
        # Record per-trial metadata for dashboard
        try:
            trial.set_user_attr("worker_id", wid)
            trial.set_user_attr("duration_sec", max(0.0, time.time() - t_start))
            trial.set_user_attr("finished_at", int(time.time()))
        except Exception:
            pass
        return ppg

    # Keep references to matches in closure to avoid recompute per trial
    dl_inner = DataLoader()
    current_season = dl_inner.get_current_season()
    test_season = current_season - 1
    start_season = max(test_season - seasons_back + 1, 2005)
    train_end_season = test_season - 1
    global train_matches, test_matches
    train_matches = dl_inner.fetch_historical_seasons(start_season, train_end_season)
    test_matches = dl_inner.fetch_season_matches(test_season)

    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Optuna tuning on fixed split (PPG objective)"
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Total Optuna trials")
    parser.add_argument(
        "--seasons-back",
        type=int,
        default=5,
        help="Number of seasons back for training window",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL for multi-worker (e.g., sqlite:////abs/path/study.db)",
    )
    parser.add_argument(
        "--study-name", type=str, default="baseline-ppg", help="Optuna study name"
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["none", "median", "hyperband"],
        default="median",
        help="Trial pruner",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=15,
        help="Trials before enabling pruning (median)",
    )
    parser.add_argument(
        "--omp-threads", type=int, default=1, help="Threads per worker for BLAS/OMP"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose training logs"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save param importances plot (HTML)",
    )
    args = parser.parse_args()

    if optuna is None:
        print("Optuna not installed. Install optuna to run tuning.")
        sys.exit(1)

    # Silence Optuna logs unless explicitly debugging
    try:
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass

    # Limit threads
    if args.omp_threads and args.omp_threads > 0:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "XGBOOST_NUM_THREADS",
        ):
            os.environ[var] = str(args.omp_threads)

    # Build initial fixed split features cache holder
    features_cache: dict[tuple[int, float], object] = {}

    # Objective
    global seasons_back
    seasons_back = int(max(1, args.seasons_back))
    objective = build_objective(
        train_df=None, test_df=None, features_cache=features_cache, verbose=args.verbose
    )

    # Configure pruner
    pruner = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(max(0, args.pruner_startup_trials))
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()

    # Create study
    @_retry_on_database_lock(max_retries=5, delay=1.0)
    def create_study():
        if args.storage:
            storage_url = _configure_sqlite_for_concurrency(args.storage)
            _enable_sqlite_wal_mode(storage_url)
            return optuna.create_study(
                direction="maximize",
                storage=storage_url,
                study_name=args.study_name or "baseline-ppg",
                load_if_exists=True,
                pruner=pruner,
            )
        else:
            return optuna.create_study(direction="maximize", pruner=pruner)

    study = create_study()

    start = time.time()
    _log(
        f"Objective: avg_points (PPG) | Trials: {args.n_trials} | Fixed split baseline"
    )

    # Run optimization
    @_retry_on_database_lock(max_retries=3, delay=1.0)
    def run_opt():
        study.optimize(
            objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=False
        )

    run_opt()

    dur = int(time.time() - start)
    # Count only completed trials for summaries/guards
    try:
        completed_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
    except Exception:
        completed_trials = len(study.trials)

    _log(f"Completed {completed_trials} trials in {dur // 60}m {dur % 60}s")

    # If no trials ran (e.g., CLI initialization run), skip best/importance outputs
    if completed_trials == 0:
        _log("No trials executed; skipping best metrics and importances.")
        # Clean up storage when not coordinated by CLI
        fs_path = _sqlite_fs_path(args.storage)
        try:
            coordinated = os.environ.get("KTP_TUNE_COORDINATED", "0") == "1"
            if (not coordinated) and fs_path and os.path.exists(fs_path):
                os.remove(fs_path)
                _log("Removed Optuna SQLite storage file")
        except Exception:
            pass
        return

    _log(f"Best PPG: {study.best_value:.6f} | Trial #{study.best_trial.number}")

    # Save importances (primary worker only)
    if _is_primary_worker():
        try:
            from optuna.importance import get_param_importances

            importances = get_param_importances(study)
            # Rank and print top 5–7
            ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            topk = ranked[:7]
            _log("Top Hyperparameters by Importance:")
            for i, (p, imp) in enumerate(topk, 1):
                _log(f"{i}. {p}: {imp:.4f}")
            # Plot
            fig = plot_param_importances(study)
            out_dir = PROJECT_ROOT / "data" / "optuna"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = (
                Path(args.save_plot)
                if args.save_plot
                else out_dir / "baseline_param_importances.html"
            )
            try:
                fig.write_html(str(out_path))
                _log(f"Param importances saved to {out_path}")
            except Exception:
                _log("Warning: Failed to save Plotly HTML for importances.")
        except Exception as e:
            _log(f"Warning: Could not compute/save param importances: {e}")

    # Save best params for reproducibility (primary worker only, end of run)
    if _is_primary_worker():
        try:
            import yaml  # type: ignore

            cfg_dir = PROJECT_ROOT / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            best = study.best_trial.params
            with open(cfg_dir / "best_params.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(best, f, sort_keys=True)
            _log(f"Best parameters saved to {cfg_dir / 'best_params.yaml'}")
        except Exception as e:
            _log(f"Warning: Failed to save best params: {e}")

    # Clean up storage when not coordinated by CLI
    fs_path = _sqlite_fs_path(args.storage)
    try:
        coordinated = os.environ.get("KTP_TUNE_COORDINATED", "0") == "1"
        if (not coordinated) and fs_path and os.path.exists(fs_path):
            os.remove(fs_path)
            _log("Removed Optuna SQLite storage file")
    except Exception:
        pass


if __name__ == "__main__":
    main()
