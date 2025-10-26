from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import optuna  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

from .config import Config, get_config
from .data import DataLoader
from .metrics import ProbabilityMetrics, ConfusionMetrics
from .predictor import GoalDifferencePredictor


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
    val_match_results = vm.loc[mask, ["match_id", "result", "goal_difference"]]
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


def run_tuning(
    n_trials: int = 100,
    seasons_back: int = 5,
    storage: Optional[str] = None,
    study_name: str = "gd_v3_tuning",
    timeout: Optional[int] = None,
) -> None:
    """Run an Optuna multi-objective study to tune V3 goal-difference model.

    Optimizes for:
    - accuracy (maximize)
    - log_loss (minimize)

    Saves selected parameters to `config/best_params.yaml`.
    """
    if optuna is None:
        raise RuntimeError(
            "Optuna not installed. Install with `pip install \"kicktipp-predictor[tuning]\"` or `pip install optuna`."
        )

    train_df, val_df, loader, current_season = _prepare_datasets(seasons_back)

    # Objective uses closure over prepared datasets
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        cfg = Config.load()  # fresh config per trial
        # --- Tune core XGB params ---
        cfg.model.gd_n_estimators = trial.suggest_int("gd_n_estimators", 100, 2000)
        cfg.model.gd_max_depth = trial.suggest_int("gd_max_depth", 3, 10)
        cfg.model.gd_learning_rate = trial.suggest_float("gd_learning_rate", 0.01, 0.3, log=True)
        cfg.model.gd_reg_lambda = trial.suggest_float("gd_reg_lambda", 1e-6, 10.0, log=True)
        cfg.model.gd_min_child_weight = trial.suggest_float("gd_min_child_weight", 0.1, 10.0, log=True)
        # --- Stochasticity ---
        cfg.model.gd_subsample = trial.suggest_float("gd_subsample", 0.6, 1.0)
        cfg.model.gd_colsample_bytree = trial.suggest_float("gd_colsample_bytree", 0.6, 1.0)
        # --- Architecture-specific uncertainty ---
        cfg.model.gd_uncertainty_stddev = trial.suggest_float("gd_uncertainty_stddev", 0.5, 3.0)
        # Keep gamma at default (not tuned)
        cfg.model.gd_gamma = 0.0

        predictor = GoalDifferencePredictor(config=cfg)
        try:
            predictor.train(train_df)
        except Exception as exc:
            # Training failed (e.g., invalid params or insufficient data); prune trial
            raise optuna.TrialPruned(f"Training failed: {exc}")

        # Predict on validation season
        preds = predictor.predict(val_df)
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

        # Metrics
        stats = ConfusionMetrics.confusion_matrix_stats(y_true, proba)
        acc = float(stats.get("accuracy", float("nan")))
        ll = float(ProbabilityMetrics.log_loss_multiclass(y_true, proba))

        if not np.isfinite(acc) or not np.isfinite(ll):
            raise optuna.TrialPruned("Non-finite evaluation metrics.")

        # Record for analysis
        trial.set_user_attr("accuracy", acc)
        trial.set_user_attr("log_loss", ll)
        trial.set_user_attr("n_val", int(len(y_true)))

        return acc, ll

    # Storage (SQLite) default to project data directory
    cfg = loader.config
    default_storage = f"sqlite:///{cfg.paths.data_dir / 'optuna_studies.db'}"
    storage_url = storage or default_storage

    # --- Reset storage before each execution run ---
    try:
        _reset_optuna_storage(storage_url)
    except Exception as exc:
        raise RuntimeError(f"Database reset failed: {exc}")

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    print("=" * 80)
    print("OPTUNA TUNING (V3 Goal-Difference)")
    print("=" * 80)
    print(f"Training seasons: {current_season - seasons_back}..{current_season - 1}")
    print(f"Validation season: {current_season}")
    print(f"Trials: {n_trials} | Storage: {storage_url}")
    print()

    study.optimize(objective, n_trials=int(n_trials), timeout=timeout)

    pareto = study.best_trials
    completed = [t for t in pareto if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials to select from.")
        return

    # Selection strategy: top-25% accuracy, then lowest log-loss
    accs = np.array([t.values[0] for t in completed], dtype=float)
    if len(accs) > 1:
        threshold = float(np.quantile(accs, 0.75))
        candidates = [t for t in completed if t.values[0] >= threshold]
    else:
        candidates = completed
    selected = min(candidates, key=lambda t: t.values[1]) if candidates else completed[0]

    params = dict(selected.params)
    out = {
        # Tuned core params
        "gd_n_estimators": int(params.get("gd_n_estimators", cfg.model.gd_n_estimators)),
        "gd_max_depth": int(params.get("gd_max_depth", cfg.model.gd_max_depth)),
        "gd_learning_rate": float(params.get("gd_learning_rate", cfg.model.gd_learning_rate)),
        "gd_subsample": float(params.get("gd_subsample", cfg.model.gd_subsample)),
        "gd_reg_lambda": float(params.get("gd_reg_lambda", cfg.model.gd_reg_lambda)),
        "gd_min_child_weight": float(params.get("gd_min_child_weight", cfg.model.gd_min_child_weight)),
        "gd_colsample_bytree": float(params.get("gd_colsample_bytree", cfg.model.gd_colsample_bytree)),
        # Not tuned; explicit default retained for clarity
        "gd_gamma": float(cfg.model.gd_gamma),
        # Architecture uncertainty
        "gd_uncertainty_stddev": float(params.get("gd_uncertainty_stddev", cfg.model.gd_uncertainty_stddev)),
    }

    # Persist to YAML under package config directory for easy loading
    pkg_config_dir = Path(__file__).parent / "config"
    pkg_config_dir.mkdir(parents=True, exist_ok=True)
    out_path = pkg_config_dir / "best_params.yaml"
    try:
        import yaml  # type: ignore
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False, indent=2)
        # Validate YAML syntax by reading back
        with open(out_path, "r", encoding="utf-8") as f:
            _validate_params = yaml.safe_load(f)
        if not isinstance(_validate_params, dict):
            raise ValueError("Invalid YAML: expected a mapping of parameters")
        print(f"Saved best parameters to {out_path}")
    except Exception:
        alt = out_path.with_suffix(".json")
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"YAML unavailable; saved JSON to {alt}")

    # Save a compact study summary for later analysis
    summary_dir = cfg.paths.data_dir / "optuna"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = summary_dir / f"{study_name}_summary.yaml"
    summary = {
        "study_name": study_name,
        "storage": storage_url,
        "n_trials": len(study.trials),
        "pareto_trials": [
            {
                "number": t.number,
                "values": list(map(float, t.values)),
                "params": t.params,
                "accuracy": float(t.user_attrs.get("accuracy", float("nan"))),
                "log_loss": float(t.user_attrs.get("log_loss", float("nan"))),
            }
            for t in pareto
        ],
        "selected_trial": {
            "number": selected.number,
            "values": list(map(float, selected.values)),
            "params": selected.params,
        },
    }
    try:
        import yaml  # type: ignore
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False, indent=2)
        # Validate YAML syntax by reading back
        with open(summary_path, "r", encoding="utf-8") as f:
            _validate_summary = yaml.safe_load(f)
        if not isinstance(_validate_summary, dict):
            raise ValueError("Invalid YAML: summary must be a mapping")
        print(f"Saved study summary to {summary_path}")
    except Exception:
        alt = summary_path.with_suffix(".json")
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"YAML unavailable; saved JSON to {alt}")

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)
    print(
        f"Selected Trial #{selected.number}: accuracy={selected.values[0]:.4f}, "
        f"log_loss={selected.values[1]:.4f}"
    )


if __name__ == "__main__":
    # Default quick run; for custom runs use CLI integration.
    run_tuning()