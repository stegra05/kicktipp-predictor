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
from .predictor import GoalDifferencePredictor, CascadedPredictor
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
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
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
        # --- Architecture-specific uncertainty (static baseline) ---
        cfg.model.gd_uncertainty_stddev = trial.suggest_float("gd_uncertainty_stddev", 0.5, 3.0)
        # Force static uncertainty during tuning to reduce complexity
        cfg.model.gd_uncertainty_base_stddev = cfg.model.gd_uncertainty_stddev
        cfg.model.gd_uncertainty_scale = 0.0
        # Keep min/max bounds from config; they will only clamp if needed
        if cfg.model.gd_uncertainty_max_stddev <= cfg.model.gd_uncertainty_min_stddev:
            raise optuna.TrialPruned("Invalid stddev bounds: max <= min")
        # Draw margin tuning
        cfg.model.draw_margin = trial.suggest_float("draw_margin", 0.1, 1.0)
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
        # Draw stats
        pred_labels = np.argmax(proba, axis=1)
        predicted_draw_rate = float(np.mean(pred_labels == 1)) if len(pred_labels) else float("nan")
        mean_draw_prob = float(np.mean(proba[:, 1])) if proba.size else float("nan")
        trial.set_user_attr("predicted_draw_rate", predicted_draw_rate)
        trial.set_user_attr("mean_draw_prob", mean_draw_prob)
        # Uncertainty diagnostics: correlation between |predicted GD| and stddev
        try:
            gd_abs = np.array([abs(float(p.get("predicted_goal_difference", 0.0))) for p in preds], dtype=float)
            stddevs = np.array([float(p.get("uncertainty_stddev", float("nan"))) for p in preds], dtype=float)
            mask = np.isfinite(gd_abs) & np.isfinite(stddevs)
            if mask.sum() >= 2 and np.std(gd_abs[mask]) > 0 and np.std(stddevs[mask]) > 0:
                corr = float(np.corrcoef(gd_abs[mask], stddevs[mask])[0, 1])
            else:
                corr = float("nan")
            trial.set_user_attr("uncertainty_corr_abs_predgd_stddev", corr)
            if stddevs[mask].size:
                trial.set_user_attr("uncertainty_stddev_min", float(np.min(stddevs[mask])))
                trial.set_user_attr("uncertainty_stddev_mean", float(np.mean(stddevs[mask])))
                trial.set_user_attr("uncertainty_stddev_max", float(np.max(stddevs[mask])))
                trial.set_user_attr("uncertainty_stddev_std", float(np.std(stddevs[mask])))
            else:
                trial.set_user_attr("uncertainty_stddev_min", float("nan"))
                trial.set_user_attr("uncertainty_stddev_mean", float("nan"))
                trial.set_user_attr("uncertainty_stddev_max", float("nan"))
                trial.set_user_attr("uncertainty_stddev_std", float("nan"))
        except Exception:
            trial.set_user_attr("uncertainty_corr_abs_predgd_stddev", float("nan"))
        # Balanced accuracy objective (soft draw nudge)
        target_draw_rate = 0.25
        draw_balance_factor = 1.0 - abs(predicted_draw_rate - target_draw_rate)
        balanced_accuracy = 0.8 * acc + 0.2 * draw_balance_factor
        trial.set_user_attr("balanced_accuracy", balanced_accuracy)
        trial.set_user_attr("draw_balance_factor", draw_balance_factor)
        trial.set_user_attr("target_draw_rate", target_draw_rate)

        # Draw rate stability across splits
        try:
            rates: list[float] = []
            if "matchday" in val_df.columns:
                md = pd.to_numeric(val_df["matchday"], errors="coerce").astype(float).values
                order = np.argsort(md)
                splits = np.array_split(order, 3)
            elif "date" in val_df.columns:
                dates = pd.to_datetime(val_df["date"], errors="coerce").values
                order = np.argsort(dates)
                splits = np.array_split(order, 3)
            else:
                splits = np.array_split(np.arange(len(pred_labels)), 3)
            for idxs in splits:
                if len(idxs) > 0:
                    rates.append(float(np.mean(pred_labels[idxs] == 1)))
            draw_rate_std = float(np.std(rates)) if len(rates) >= 2 else float("nan")
            trial.set_user_attr("draw_rate_splits", rates)
            trial.set_user_attr("draw_rate_std", draw_rate_std)
            trial.set_user_attr(
                "draw_rate_splits_target_ok",
                bool(all((0.15 <= r <= 0.30) for r in rates if np.isfinite(r))),
            )
        except Exception:
            pass

        return balanced_accuracy, ll

    # Storage (SQLite) default to project data directory
    cfg = loader.config
    default_storage = f"sqlite:///{cfg.paths.data_dir / 'optuna_studies.db'}"
    storage_url = storage or default_storage

    # --- Reset storage before each execution run ---
    try:
        _reset_optuna_storage(storage_url)
    except Exception as exc:
        raise RuntimeError(f"Database reset failed: {exc}")

    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
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

    # Selection: choose highest balanced_accuracy (objective #0) from Pareto front
    try:
        selected = max(completed, key=lambda t: float(t.values[0]))
    except Exception:
        print("Selection fallback: no completed trials; using study.best_trials or study.trials.")
        pool = [t for t in study.best_trials if t.state == optuna.trial.TrialState.COMPLETE] or \
               [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        selected = max(pool, key=lambda t: float(t.values[0])) if pool else study.best_trial

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
        # Uncertainty parameters (static baseline; dynamic scale fixed to 0)
        "gd_uncertainty_stddev": float(params.get("gd_uncertainty_stddev", cfg.model.gd_uncertainty_stddev)),
        "gd_uncertainty_base_stddev": float(params.get("gd_uncertainty_stddev", cfg.model.gd_uncertainty_stddev)),
        "gd_uncertainty_scale": 0.0,
        "gd_uncertainty_min_stddev": float(cfg.model.gd_uncertainty_min_stddev),
        "gd_uncertainty_max_stddev": float(cfg.model.gd_uncertainty_max_stddev),
        # Draw margin
        "draw_margin": float(params.get("draw_margin", cfg.model.draw_margin)),
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
                "balanced_accuracy": float(t.user_attrs.get("balanced_accuracy", float("nan"))),
                "draw_balance_factor": float(t.user_attrs.get("draw_balance_factor", float("nan"))),
                "log_loss": float(t.user_attrs.get("log_loss", float("nan"))),
                "predicted_draw_rate": float(t.user_attrs.get("predicted_draw_rate", float("nan"))),
                "mean_draw_prob": float(t.user_attrs.get("mean_draw_prob", float("nan"))),
                "draw_rate_std": float(t.user_attrs.get("draw_rate_std", float("nan"))),
                "draw_rate_splits": [float(r) for r in t.user_attrs.get("draw_rate_splits", [])],
                "draw_rate_splits_target_ok": bool(t.user_attrs.get("draw_rate_splits_target_ok", False)),
                "uncertainty_corr_abs_predgd_stddev": float(t.user_attrs.get("uncertainty_corr_abs_predgd_stddev", float("nan"))),
                "uncertainty_stddev_min": float(t.user_attrs.get("uncertainty_stddev_min", float("nan"))),
                "uncertainty_stddev_mean": float(t.user_attrs.get("uncertainty_stddev_mean", float("nan"))),
                "uncertainty_stddev_max": float(t.user_attrs.get("uncertainty_stddev_max", float("nan"))),
                "uncertainty_stddev_std": float(t.user_attrs.get("uncertainty_stddev_std", float("nan"))),
            }
            for t in pareto
        ],
        "selected_trial": {
            "number": selected.number,
            "values": list(map(float, selected.values)),
            "params": selected.params,
        },
    }
    # Compute trial-level analytics
    try:
        balanced_accuracies = [float(t.user_attrs.get("balanced_accuracy", float("nan"))) for t in pareto]
        raw_accuracies = [float(t.user_attrs.get("accuracy", float("nan"))) for t in pareto]
        draw_factors = [float(t.user_attrs.get("draw_balance_factor", float("nan"))) for t in pareto]
        log_losses = [float(t.user_attrs.get("log_loss", float("nan"))) for t in pareto]
        draw_rates = [float(t.user_attrs.get("predicted_draw_rate", float("nan"))) for t in pareto]
        ba = np.array(balanced_accuracies, dtype=float); aa = np.array(raw_accuracies, dtype=float)
        df = np.array(draw_factors, dtype=float); ll = np.array(log_losses, dtype=float)
        mask_ba = np.isfinite(ba) & np.isfinite(ll)
        mask_df = np.isfinite(df) & np.isfinite(ll)
        if mask_ba.sum() >= 2 and np.std(ba[mask_ba]) > 0 and np.std(ll[mask_ba]) > 0:
            corr_ba_ll = float(np.corrcoef(ba[mask_ba], ll[mask_ba])[0, 1])
        else:
            corr_ba_ll = float("nan")
        if mask_df.sum() >= 2 and np.std(df[mask_df]) > 0 and np.std(ll[mask_df]) > 0:
            corr_df_ll = float(np.corrcoef(df[mask_df], ll[mask_df])[0, 1])
        else:
            corr_df_ll = float("nan")
        dr = np.array(draw_rates, dtype=float)
        dr_f = dr[np.isfinite(dr)]
        draw_rate_stats = {
            "count": int(dr_f.size),
            "min": float(np.min(dr_f)) if dr_f.size else float("nan"),
            "max": float(np.max(dr_f)) if dr_f.size else float("nan"),
            "mean": float(np.mean(dr_f)) if dr_f.size else float("nan"),
            "std": float(np.std(dr_f)) if dr_f.size else float("nan"),
        }
        summary["correlations"] = {
            "balanced_accuracy_vs_log_loss": corr_ba_ll,
            "draw_balance_factor_vs_log_loss": corr_df_ll,
        }
        summary["predicted_draw_rate_stats"] = draw_rate_stats
    except Exception:
        pass
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
    sel_dr = float(selected.user_attrs.get("predicted_draw_rate", float("nan")))
    ba = float(selected.user_attrs.get("balanced_accuracy", float("nan"))) if hasattr(selected, "user_attrs") else float("nan")
    print(
        f"Selected Trial #{selected.number}: balanced_accuracy={selected.values[0]:.4f}, "
        f"log_loss={selected.values[1]:.4f}, predicted_draw_rate={sel_dr:.3f}"
    )


if __name__ == "__main__":
    # Default quick run; for custom runs use CLI integration.
    run_tuning()


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
) -> None:
    """Sequential tuning for V4 CascadedPredictor.

    Phase 1: tune draw_* exclusively using draw-focused metric.
    Phase 2: tune win_* with draw_* fixed, optimizing combined outcome quality.
    """
    if optuna is None:
        raise RuntimeError(
            "Optuna not installed. Install with `pip install \"kicktipp-predictor[tuning]\"` or `pip install optuna`."
        )

    # Prepare datasets
    train_df, val_df, loader, current_season = _prepare_datasets(seasons_back)
    cfg_base = loader.config

    # Default storage (SQLite) under data dir
    default_storage = f"sqlite:///{cfg_base.paths.data_dir / 'optuna_studies.db'}"
    storage_url = storage or default_storage

    # Reset storage before each run for reproducibility
    try:
        _reset_optuna_storage(storage_url)
    except Exception as exc:
        raise RuntimeError(f"Database reset failed: {exc}")

    # ---------------------- Phase 1: Draw Tuning ----------------------
    if model_to_tune in ("draw", "both"):
        print("=" * 80)
        print("OPTUNA TUNING (V4 Cascaded) - Phase 1: Draw Model")
        print("=" * 80)
        print(f"Training seasons: {current_season - seasons_back}..{current_season - 1}")
        print(f"Validation season: {current_season}")
        print(f"Trials: {n_trials} | Storage: {storage_url}")
        print(f"Objective: maximize {draw_metric}")
        print()

        # Define objective for draw-only metric
        def objective_draw(trial: optuna.Trial) -> float:
            cfg = Config.load()
            # Suggest only draw_* params
            cfg.model.draw_n_estimators = trial.suggest_int("draw_n_estimators", 100, 1500)
            cfg.model.draw_max_depth = trial.suggest_int("draw_max_depth", 3, 10)
            cfg.model.draw_learning_rate = trial.suggest_float("draw_learning_rate", 0.01, 0.3, log=True)
            cfg.model.draw_subsample = trial.suggest_float("draw_subsample", 0.6, 1.0)
            cfg.model.draw_colsample_bytree = trial.suggest_float("draw_colsample_bytree", 0.5, 1.0)
            cfg.model.draw_scale_pos_weight = trial.suggest_float("draw_scale_pos_weight", 1.0, 8.0)

            predictor = CascadedPredictor(config=cfg)
            try:
                predictor.train(train_df)
            except Exception as exc:
                raise optuna.TrialPruned(f"Training failed: {exc}")

            # Prepare validation features aligned to training columns
            X_val = predictor._prepare_features(val_df)
            y_val_draw = (val_df["result"].astype(str) == "D").astype(int).to_numpy()
            # Must have both classes for meaningful AUC/F1
            if len(np.unique(y_val_draw)) < 2:
                raise optuna.TrialPruned("Validation set lacks both draw and non-draw classes.")

            # Compute draw probabilities
            try:
                draw_proba = predictor.draw_model.predict_proba(X_val)
                # Map encoder to model classes index
                draw_label = int(predictor.draw_label_encoder.transform([1])[0])
                idx = int(np.where(predictor.draw_model.classes_ == draw_label)[0][0])
                p_draw = draw_proba[:, idx]
            except Exception as exc:
                raise optuna.TrialPruned(f"Probability computation failed: {exc}")

            # Metric calculation
            if draw_metric == "roc_auc":
                score = float(roc_auc_score(y_val_draw, p_draw))
            elif draw_metric == "f1":
                y_pred = (p_draw >= 0.5).astype(int)
                score = float(f1_score(y_val_draw, y_pred))
            else:
                raise optuna.TrialPruned(f"Unsupported draw_metric: {draw_metric}")

            trial.set_user_attr("metric", draw_metric)
            trial.set_user_attr("score", score)
            trial.set_user_attr("n_val", int(len(y_val_draw)))
            trial.set_user_attr("mean_p_draw", float(np.mean(p_draw)))
            return score  # maximize

        study_draw = optuna.create_study(
            direction="maximize",
            study_name=f"{study_name}_draw",
            storage=storage_url,
            load_if_exists=True,
        )
        study_draw.optimize(objective_draw, n_trials=int(n_trials), timeout=timeout)

        # Select best trial and persist draw_* params
        best_draw = study_draw.best_trial
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
        # Save a compact study summary
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
                        "best_trial": {
                            "number": best_draw.number,
                            "value": float(best_draw.value),
                            "params": best_draw.params,
                            "metric": draw_metric,
                        },
                        "updated_file": str(saved_path),
                    },
                    f,
                    sort_keys=False,
                    indent=2,
                )
        except Exception:
            pass

    # ---------------------- Phase 2: Win Tuning ----------------------
    if model_to_tune in ("win", "both"):
        print("=" * 80)
        print("OPTUNA TUNING (V4 Cascaded) - Phase 2: Win Model")
        print("=" * 80)
        print(f"Trials: {n_trials} | Storage: {storage_url}")
        print(f"Objective: {'maximize' if win_metric == 'accuracy' else 'minimize'} {win_metric}")
        print()

        # Load fixed draw params from best_params.yaml (if present)
        cfg_fixed = Config.load()

        def objective_win(trial: optuna.Trial) -> float:
            cfg = Config.load()
            # Fix draw_* to previously saved (cfg_fixed already applied from YAML)
            cfg.model.draw_n_estimators = cfg_fixed.model.draw_n_estimators
            cfg.model.draw_max_depth = cfg_fixed.model.draw_max_depth
            cfg.model.draw_learning_rate = cfg_fixed.model.draw_learning_rate
            cfg.model.draw_subsample = cfg_fixed.model.draw_subsample
            cfg.model.draw_colsample_bytree = cfg_fixed.model.draw_colsample_bytree
            cfg.model.draw_scale_pos_weight = cfg_fixed.model.draw_scale_pos_weight

            # Suggest only win_* params
            cfg.model.win_n_estimators = trial.suggest_int("win_n_estimators", 100, 2000)
            cfg.model.win_max_depth = trial.suggest_int("win_max_depth", 3, 10)
            cfg.model.win_learning_rate = trial.suggest_float("win_learning_rate", 0.01, 0.3, log=True)
            cfg.model.win_subsample = trial.suggest_float("win_subsample", 0.6, 1.0)
            cfg.model.win_colsample_bytree = trial.suggest_float("win_colsample_bytree", 0.5, 1.0)

            predictor = CascadedPredictor(config=cfg)
            try:
                predictor.train(train_df)
            except Exception as exc:
                raise optuna.TrialPruned(f"Training failed: {exc}")

            # Combined predictions
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

            # Metric
            if win_metric == "accuracy":
                acc = float(
                    np.mean(
                        np.argmax(proba, axis=1)
                        == np.array([{ "H": 0, "D": 1, "A": 2 }[t] for t in y_true])
                    )
                )
                score = acc
            elif win_metric == "log_loss":
                score = float(ProbabilityMetrics.log_loss_multiclass(y_true, proba))
            else:
                raise optuna.TrialPruned(f"Unsupported win_metric: {win_metric}")

            trial.set_user_attr("metric", win_metric)
            trial.set_user_attr("score", score)
            trial.set_user_attr("n_val", int(len(y_true)))
            return score  # max or min depending on study

        study_win = optuna.create_study(
            direction="maximize" if win_metric == "accuracy" else "minimize",
            study_name=f"{study_name}_win",
            storage=storage_url,
            load_if_exists=True,
        )
        study_win.optimize(objective_win, n_trials=int(n_trials), timeout=timeout)

        best_win = study_win.best_trial
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
        # Save summary
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
                        "best_trial": {
                            "number": best_win.number,
                            "value": float(best_win.value),
                            "params": best_win.params,
                            "metric": win_metric,
                        },
                        "updated_file": str(saved_path),
                    },
                    f,
                    sort_keys=False,
                    indent=2,
                )
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("SEQUENTIAL TUNING COMPLETE (V4 Cascaded)")
    print("=" * 80)