"""
Script: run_shap_analysis.py
Purpose: Compute and visualize SHAP values for the GoalDifferencePredictor.
Author: Kicktipp Predictor Team
Date: 2025-10-27

This script loads the trained model and project data, computes SHAP values,
produces summary and dependence plots, and writes a markdown report with key
findings. It supports large datasets via chunked processing and will retrain a
model if none is found.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed

from kicktipp_predictor.config import get_config
from kicktipp_predictor.data import DataLoader
from kicktipp_predictor.predictor import GoalDifferencePredictor


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_dataframe(df: pd.DataFrame, name: str) -> dict:
    """Run basic validation checks and return a summary dict."""
    summary = {
        "name": name,
        "n_rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "n_cols": int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0,
        "null_counts": {},
        "dtypes": {},
        "head_preview": None,
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[WARN] DataFrame '{name}' is empty or invalid.")
        return summary
    summary["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    summary["dtypes"] = {c: str(dt) for c, dt in df.dtypes.items()}
    try:
        summary["head_preview"] = df.head(3).to_dict(orient="records")
    except Exception:
        summary["head_preview"] = None
    # Human-readable logs
    print(f"[INFO] {name}: {summary['n_rows']} rows, {summary['n_cols']} cols")
    nulls_total = sum(summary["null_counts"].values())
    print(f"[INFO] {name}: total nulls = {nulls_total}")
    return summary


def load_or_train_model(
    predictor: GoalDifferencePredictor, train_df: pd.DataFrame
) -> None:
    """Load a trained model; if unavailable, train and save one."""
    try:
        predictor.load_model()
        print("[INFO] Loaded existing model and metadata.")
        return
    except FileNotFoundError:
        print("[WARN] No model found on disk. Training a new model...")
    except Exception as exc:
        print(f"[WARN] Failed to load model ({exc}). Retraining...")

    # Fallback: train a model
    predictor.train(train_df)
    predictor.save_model()
    print("[INFO] Model trained and saved.")


def _chunks(n: int, chunk_size: int) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        bounds.append((start, end))
        start = end
    return bounds


def compute_shap_values_tree(
    model, X: pd.DataFrame, n_jobs: int, chunk_size: int
) -> tuple[np.ndarray, float]:
    """Compute SHAP values using XGBoost's native pred_contribs for tree models.

    Returns (shap_values, base_value).
    """
    booster = model.get_booster()
    # Use DMatrix for efficient prediction with contributions
    # Process in chunks to control memory usage
    bounds = _chunks(len(X), chunk_size)

    def _compute_chunk(s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
        dmat = xgb.DMatrix(X.iloc[s:e].values, feature_names=X.columns.tolist())
        contribs = booster.predict(
            dmat, pred_contribs=True
        )  # shape: (rows, n_features+1)
        return contribs[:, :-1], contribs[:, -1]

    parts = (
        Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(_compute_chunk)(s, e) for (s, e) in bounds
        )
        if bounds
        else []
    )

    if not parts:
        return np.empty((0, X.shape[1])), 0.0

    shap_values = np.vstack([p[0] for p in parts])
    base_values = np.concatenate([p[1] for p in parts])
    base_value = float(np.mean(base_values)) if len(base_values) > 0 else 0.0
    return shap_values, base_value


def compute_shap_values_kernel(
    model, X: pd.DataFrame, n_jobs: int, chunk_size: int
) -> tuple[np.ndarray, float]:
    """Compute SHAP values using KernelExplainer for non-tree models."""
    # Background: use kmeans for a compact summary
    bg = shap.kmeans(X, min(100, len(X)))
    explainer = shap.KernelExplainer(model.predict, bg)
    n = len(X)
    bounds = _chunks(n, chunk_size)

    def _compute_chunk(s: int, e: int) -> np.ndarray:
        return np.array(explainer.shap_values(X.iloc[s:e], nsamples="auto"))

    parts: list[np.ndarray] = (
        Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(_compute_chunk)(s, e) for (s, e) in bounds
        )
        if bounds
        else []
    )

    shap_values = np.vstack(parts) if parts else np.empty((0, X.shape[1]))
    # KernelExplainer expected_value typically scalar
    base_value = explainer.expected_value
    try:
        base_value = float(base_value)
    except Exception:
        base_value = 0.0
    return shap_values, base_value


def feature_importance(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    imp_df.reset_index(drop=True, inplace=True)
    return imp_df


def save_array_artifacts(
    out_dir: Path,
    shap_values: np.ndarray,
    base_value: float,
    feature_names: list[str],
    index: pd.Index,
) -> None:
    npz_path = out_dir / "shap_values.npz"
    np.savez_compressed(
        npz_path,
        shap_values=shap_values,
        base_value=np.array([base_value], dtype=float),
        feature_names=np.array(feature_names, dtype=object),
        row_index=np.array(index.astype(str)),
    )
    print(f"[INFO] Saved SHAP arrays to {npz_path}")


def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    png = out_dir / "summary_beeswarm.png"
    svg = out_dir / "summary_beeswarm.svg"
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved summary plots: {png.name}, {svg.name}")


def plot_dependence_for_top_features(
    shap_values: np.ndarray, X: pd.DataFrame, top_features: list[str], out_dir: Path
) -> None:
    for feat in top_features:
        try:
            plt.figure(figsize=(9, 6))
            shap.dependence_plot(feat, shap_values, X, show=False)
            out_png = out_dir / f"dependence_{feat}.png"
            out_svg = out_dir / f"dependence_{feat}.svg"
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            plt.savefig(out_svg, bbox_inches="tight")
            plt.close()
            print(
                f"[INFO] Saved dependence plot for '{feat}' -> {out_png.name}, {out_svg.name}"
            )
        except Exception as exc:
            print(f"[WARN] Failed dependence plot for '{feat}': {exc}")


def analyze_tanh_tamed_elo(
    shap_values: np.ndarray, X: pd.DataFrame, out_dir: Path
) -> dict:
    results: dict = {"feature": "tanh_tamed_elo"}
    feat = "tanh_tamed_elo"
    if feat not in X.columns:
        results["present"] = False
        print("[WARN] 'tanh_tamed_elo' not found in feature set.")
        return results
    results["present"] = True
    j = X.columns.get_loc(feat)
    x = pd.to_numeric(X[feat], errors="coerce").fillna(0.0).values
    s = shap_values[:, j]
    # Metrics: Pearson and Spearman correlations
    try:
        pearson = float(np.corrcoef(x, s)[0, 1]) if len(x) > 1 else float("nan")
    except Exception:
        pearson = float("nan")
    try:
        spearman = float(pd.Series(x).corr(pd.Series(s), method="spearman"))
    except Exception:
        spearman = float("nan")
    results["pearson_corr"] = pearson
    results["spearman_corr"] = spearman
    # Directionality heuristic
    results["direction"] = "positive" if pearson >= 0 else "negative"
    # Save dependence plot explicitly
    try:
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(feat, shap_values, X, show=False)
        out_png = out_dir / "dependence_tanh_tamed_elo.png"
        out_svg = out_dir / "dependence_tanh_tamed_elo.svg"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.savefig(out_svg, bbox_inches="tight")
        plt.close()
        print(
            f"[INFO] Saved dependence plot for 'tanh_tamed_elo' -> {out_png.name}, {out_svg.name}"
        )
    except Exception as exc:
        print(f"[WARN] Failed special dependence plot for tanh_tamed_elo: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for GoalDifferencePredictor"
    )
    parser.add_argument(
        "--seasons-back",
        type=int,
        default=5,
        help="Number of past seasons for training data",
    )
    parser.add_argument(
        "--max-samples", type=int, default=5000, help="Max samples for SHAP computation"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000, help="Chunk size for SHAP computation"
    )
    parser.add_argument(
        "--plots-top-n", type=int, default=7, help="Top-N features for dependence plots"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for artifacts"
    )
    parser.add_argument(
        "--explainer",
        type=str,
        default="auto",
        choices=["auto", "tree", "kernel"],
        help="Explainer backend",
    )
    args = parser.parse_args()

    cfg = get_config()
    default_out = cfg.paths.data_dir / "analysis" / "shap"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    _ensure_dir(out_dir)

    print("=" * 80)
    print("SHAP ANALYSIS")
    print("=" * 80)
    print(f"Output: {out_dir}")

    # Data loading
    loader = DataLoader()
    current_season = loader.get_current_season()
    start_season = current_season - int(args.seasons_back)
    print(f"[INFO] Fetching seasons {start_season}..{current_season}")
    all_matches = loader.fetch_historical_seasons(start_season, current_season)
    print(f"[INFO] Loaded {len(all_matches)} matches")

    # Features for training
    features_df = loader.create_features_from_matches(all_matches)
    validate_dataframe(features_df, "training_features")

    # Initialize/load predictor
    predictor = GoalDifferencePredictor()
    load_or_train_model(predictor, features_df)

    # Align feature matrix
    X_all = features_df.reindex(columns=predictor.feature_columns).fillna(0.0)
    if len(X_all) == 0:
        raise RuntimeError("No data available after feature alignment.")

    # Sample to control runtime
    if len(X_all) > args.max_samples:
        X = X_all.sample(n=int(args.max_samples), random_state=cfg.model.random_state)
    else:
        X = X_all
    print(f"[INFO] Using {len(X)} samples for SHAP.")

    # Compute SHAP
    backend = args.explainer
    is_tree = hasattr(predictor.model, "get_booster")  # XGBRegressor
    if backend == "auto":
        backend = "tree" if is_tree else "kernel"
    print(f"[INFO] Explainer backend: {backend}")

    if backend == "tree":
        shap_values, base_value = compute_shap_values_tree(
            predictor.model, X, cfg.model.n_jobs, args.chunk_size
        )
    else:
        shap_values, base_value = compute_shap_values_kernel(
            predictor.model, X, cfg.model.n_jobs, args.chunk_size
        )

    # Save arrays
    save_array_artifacts(out_dir, shap_values, base_value, list(X.columns), X.index)

    # Feature importance
    imp_df = feature_importance(shap_values, list(X.columns))
    imp_path = out_dir / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"[INFO] Saved feature importance to {imp_path}")

    # Plots
    plot_summary(shap_values, X, out_dir)
    top_feats = imp_df["feature"].head(int(args.plots_top_n)).tolist()
    plot_dependence_for_top_features(shap_values, X, top_feats, out_dir)

    # Special focus: tanh_tamed_elo
    tanh_stats = analyze_tanh_tamed_elo(shap_values, X, out_dir)

    # Findings markdown
    metadata = {
        "model_path": str(cfg.paths.gd_model_path),
        "n_features": int(len(X.columns)),
        "n_samples": int(len(X)),
        "explainer": backend,
        "base_value": float(base_value),
        "config": {
            "n_estimators": int(cfg.model.gd_n_estimators),
            "max_depth": int(cfg.model.gd_max_depth),
            "learning_rate": float(cfg.model.gd_learning_rate),
            "subsample": float(cfg.model.gd_subsample),
            "colsample_bytree": float(cfg.model.gd_colsample_bytree),
            "draw_margin": float(cfg.model.draw_margin),
        },
    }
    # write_findings_md(
    #     out_dir, imp_df, tanh_stats, top_k=int(args.plots_top_n), metadata=metadata
    # )

    print("\nDone.")


if __name__ == "__main__":
    main()
