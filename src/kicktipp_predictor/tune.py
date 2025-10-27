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


if __name__ == "__main__":
    # Default quick run for V4 sequential tuning
    run_tuning_v4_sequential()


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