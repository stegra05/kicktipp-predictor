from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..metrics import ensure_dir

# Optional heavy deps (loaded lazily at runtime)
shap = None  # type: ignore
plt = None  # type: ignore

# Rich is a core dependency in this project, but keep a soft fallback to prints
try:  # pragma: no cover
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore
    box = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore

console = Console() if Console is not None else None


def _print(msg: str) -> None:
    if console is not None:
        console.print(msg)
    else:
        print(msg)


@contextmanager
def _status(message: str):
    if console is not None and Progress is not None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as progress:
            task_id = progress.add_task(message, total=None)
            try:
                yield
            finally:
                progress.update(task_id, completed=True)
    else:
        _print(message)
        yield


def _ensure_shap_matplotlib_loaded() -> tuple[bool, str]:
    """Load SHAP and matplotlib.pyplot on demand.

    Returns (ok, error_message).
    """
    global shap, plt
    if shap is not None and plt is not None:
        return True, ""
    try:
        import importlib

        if shap is None:
            shap = importlib.import_module("shap")  # type: ignore
        if plt is None:
            # Import pyplot explicitly to avoid importing matplotlib at module import time
            import matplotlib.pyplot as _plt  # type: ignore

            plt = _plt
        return True, ""
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _summarize_dataframe(X) -> tuple[int, int, int]:
    """Return (rows, cols, memory_bytes) for a pandas DataFrame-like object."""
    try:
        rows, cols = X.shape
        mem = int(getattr(X, "memory_usage", lambda deep=True: [0])(deep=True).sum())
        return rows, cols, mem
    except Exception:
        try:
            rows = len(X)
        except Exception:
            rows = -1
        try:
            cols = len(getattr(X, "columns", []))
        except Exception:
            cols = -1
        return rows, cols, -1


def _try_build_explainer(
    model: Any, is_classifier: bool, background_X=None
) -> tuple[Any | None, str]:
    """Create a SHAP explainer for a model with sensible fallbacks.

    Returns (explainer, mode_description). Includes brief error detail on failure.
    """
    if shap is None:
        return None, "shap-not-installed"

    errors: list[str] = []

    # 1) New API: shap.explainers.Tree
    try:
        try:
            # Available in modern SHAP
            from shap.explainers import Tree as TreeExplainerNew  # type: ignore

            explainer = TreeExplainerNew(model)
            return explainer, "TreeExplainer(new)"
        except Exception as e1:  # noqa: F841
            # Some versions require background data
            if background_X is not None:
                from shap.explainers import Tree as TreeExplainerNew  # type: ignore

                explainer = TreeExplainerNew(model, data=background_X)
                return explainer, "TreeExplainer(new,data)"
            errors.append("new-TreeExplainer")
    except Exception as e:
        errors.append(f"new-TreeExplainer:{e}")

    # 2) Legacy API: shap.TreeExplainer
    try:
        if is_classifier:
            explainer = shap.TreeExplainer(model, model_output="probability")  # type: ignore
            return explainer, "TreeExplainer(probability)"
        explainer = shap.TreeExplainer(model)  # type: ignore
        return explainer, "TreeExplainer"
    except Exception as e:
        errors.append(f"legacy-TreeExplainer:{e}")

    # 3) Generic API with data: shap.Explainer(model, data)
    try:
        if background_X is not None:
            explainer = shap.Explainer(model, background_X)  # type: ignore
            return explainer, "Explainer(data)"
    except Exception as e:
        errors.append(f"Explainer(data):{e}")

    # 4) Function-based fallback (predict/proba) + data
    try:
        if background_X is not None:
            if is_classifier and hasattr(model, "predict_proba"):
                explainer = shap.Explainer(model.predict_proba, background_X)  # type: ignore
                return explainer, "Explainer(predict_proba)"
            if hasattr(model, "predict"):
                explainer = shap.Explainer(model.predict, background_X)  # type: ignore
                return explainer, "Explainer(predict)"
    except Exception as e:
        errors.append(f"Explainer(func):{e}")

    # Failure with details
    detail = ";".join(errors) if errors else "unknown"
    return None, f"explainer-construction-failed:{detail}"


def _compute_shap_values(explainer: Any, X) -> Any:
    """Compute SHAP values using the given explainer, handling API variants."""
    # New API returns Explanation via __call__
    try:
        return explainer.shap_values(X)
    except Exception:
        return explainer(X)


def _save_summary_plots(
    values: Any, X, base_path: str, class_names: list[str] | None = None
) -> list[str]:
    """Save beeswarm and bar summary plots. Returns list of saved file paths.

    If `values` is a list (multiclass), use `class_names` for suffixes when provided.
    """
    saved_paths: list[str] = []
    if plt is None or shap is None:
        return saved_paths

    # Helper to save a single pair of plots
    def _save_pair(v, suffix: str) -> None:
        # Beeswarm
        plt.figure()
        shap.summary_plot(v, X, show=False)
        plt.tight_layout()
        out_path_swarm = f"{base_path}_{suffix}.png"
        plt.savefig(out_path_swarm)
        saved_paths.append(out_path_swarm)
        plt.close()

        # Bar
        plt.figure()
        shap.summary_plot(v, X, show=False, plot_type="bar")
        plt.tight_layout()
        out_path_bar = f"{base_path}_{suffix}_bar.png"
        plt.savefig(out_path_bar)
        saved_paths.append(out_path_bar)
        plt.close()

    # Multiclass list
    if isinstance(values, list):
        for idx, v in enumerate(values):
            name = None
            if class_names and idx < len(class_names):
                # sanitize filename component
                name = str(class_names[idx]).replace("/", "-").replace(" ", "_")
            suffix = name or f"cls{idx}"
            _save_pair(v, suffix)
        return saved_paths

    # Explanation object or 2D ndarray
    _save_pair(values, "summary")
    return saved_paths


def _compute_mean_abs_importance(
    values: Any, feature_names: Iterable[str]
) -> list[tuple[str, float]]:
    """Compute mean |SHAP| per feature; if multiclass list, average across classes."""

    def _per_feature_mean_abs(v: Any) -> np.ndarray:
        try:
            arr = v.values if hasattr(v, "values") else v
        except Exception:
            arr = v
        arr = np.asarray(arr)
        # If arr is 3D (n_samples, n_features, n_outputs), reduce outputs first
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return np.abs(arr).mean(axis=0)

    if isinstance(values, list):
        per_class = [_per_feature_mean_abs(v) for v in values]
        mean_abs = np.mean(np.stack(per_class, axis=0), axis=0)
    else:
        mean_abs = _per_feature_mean_abs(values)

    pairs = list(zip(list(feature_names), mean_abs.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _safe_import_pandas():
    try:  # pragma: no cover
        import importlib

        return importlib.import_module("pandas")  # type: ignore
    except Exception:
        return None


def _extract_signed_arrays(values: Any) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Return (signed_2d, per_class_2d_list|None).

    - If `values` is a list, returns average across classes as signed_2d and the per-class arrays.
    - If `values` is (n_samples, n_features[, n_outputs]), reduces outputs by mean.
    """
    if isinstance(values, list):
        per_class = []
        for v in values:
            try:
                arr = v.values if hasattr(v, "values") else v
            except Exception:
                arr = v
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            per_class.append(arr)
        stacked = np.stack(per_class, axis=0)  # (n_classes, n_samples, n_features)
        signed_2d = stacked.mean(axis=0)
        return signed_2d, per_class

    try:
        arr = values.values if hasattr(values, "values") else values
    except Exception:
        arr = values
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr, None


def _compute_featurewise_stats(
    values: Any,
    X,
    feature_names: Iterable[str],
    is_classifier: bool,
    class_names: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Compute rich per-feature SHAP statistics and overall summary.

    Returns (overall_feature_stats, per_class_feature_stats, summary_metrics)
    where each feature_stats item includes keys:
    - feature, mean_abs, std_abs, median_abs, p95_abs, mean_signed, std_signed,
      pos_fraction, neg_fraction, nonzero_fraction, importance_normalized, rank,
      corr_feature_to_shap (Pearson; NaN if unavailable)
    """
    feature_names_list = list(feature_names)
    signed_2d, per_class = _extract_signed_arrays(values)
    arr_abs = np.abs(signed_2d)

    n_samples, n_features = signed_2d.shape

    # Denominator for normalization
    mean_abs_per_feature = arr_abs.mean(axis=0)
    sum_mean_abs = float(mean_abs_per_feature.sum()) or 1.0
    importance_normalized = mean_abs_per_feature / sum_mean_abs

    # Fractions and dispersion
    pos_fraction = (signed_2d > 0).mean(axis=0)
    neg_fraction = (signed_2d < 0).mean(axis=0)
    nonzero_fraction = (signed_2d != 0).mean(axis=0)

    std_abs = arr_abs.std(axis=0)
    median_abs = np.median(arr_abs, axis=0)
    p95_abs = np.percentile(arr_abs, 95, axis=0)
    mean_signed = signed_2d.mean(axis=0)
    std_signed = signed_2d.std(axis=0)

    # Correlation between feature values and SHAP values
    corrs: list[float] = []
    try:
        # Attempt pandas-powered numeric handling first
        pd = _safe_import_pandas()
        if pd is not None and hasattr(X, "iloc"):
            for idx in range(n_features):
                try:
                    s_feat = pd.to_numeric(X.iloc[:, idx], errors="coerce")
                    s_shap = pd.Series(signed_2d[:, idx])
                    if (
                        s_feat.notna().sum() > 1
                        and s_feat.std() > 0
                        and s_shap.std() > 0
                    ):
                        corrs.append(float(s_feat.corr(s_shap)))
                    else:
                        corrs.append(float("nan"))
                except Exception:
                    corrs.append(float("nan"))
        else:
            # Pure numpy fallback
            x_values = None
            try:
                x_values = (
                    np.asarray(X.values) if hasattr(X, "values") else np.asarray(X)
                )
            except Exception:
                x_values = np.asarray(X)
            for idx in range(n_features):
                xi = x_values[:, idx]
                yi = signed_2d[:, idx]
                try:
                    if np.std(xi) > 0 and np.std(yi) > 0:
                        r = float(np.corrcoef(xi, yi)[0, 1])
                    else:
                        r = float("nan")
                except Exception:
                    r = float("nan")
                corrs.append(r)
    except Exception:
        corrs = [float("nan")] * n_features

    # Assemble per-feature dicts
    overall_stats: list[dict[str, Any]] = []
    for i, feat in enumerate(feature_names_list):
        overall_stats.append(
            {
                "feature": str(feat),
                "mean_abs": float(mean_abs_per_feature[i]),
                "std_abs": float(std_abs[i]),
                "median_abs": float(median_abs[i]),
                "p95_abs": float(p95_abs[i]),
                "mean_signed": float(mean_signed[i]),
                "std_signed": float(std_signed[i]),
                "pos_fraction": float(pos_fraction[i]),
                "neg_fraction": float(neg_fraction[i]),
                "nonzero_fraction": float(nonzero_fraction[i]),
                "importance_normalized": float(importance_normalized[i]),
                "corr_feature_to_shap": corrs[i],
            }
        )

    # Rank by mean_abs
    overall_stats.sort(key=lambda d: d["mean_abs"], reverse=True)
    for rank, row in enumerate(overall_stats, start=1):
        row["rank"] = rank

    # Per-class stats (only mean_abs + normalized + rank for compactness)
    per_class_stats: dict[str, list[dict[str, Any]]] = {}
    if is_classifier and per_class is not None:
        class_labels = class_names or [f"cls{i}" for i in range(len(per_class))]
        for c_idx, cls_arr in enumerate(per_class):
            cls_abs = np.abs(cls_arr)
            cls_mean_abs = cls_abs.mean(axis=0)
            denom = float(cls_mean_abs.sum()) or 1.0
            cls_norm = cls_mean_abs / denom
            class_rows = []
            for i, feat in enumerate(feature_names_list):
                class_rows.append(
                    {
                        "feature": str(feat),
                        "class": str(class_labels[c_idx]),
                        "mean_abs": float(cls_mean_abs[i]),
                        "importance_normalized": float(cls_norm[i]),
                    }
                )
            class_rows.sort(key=lambda d: d["mean_abs"], reverse=True)
            for rank, row in enumerate(class_rows, start=1):
                row["rank"] = rank
            per_class_stats[str(class_labels[c_idx])] = class_rows

    # Overall summary metrics
    p = importance_normalized
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    max_entropy = math.log(n_features) if n_features > 0 else 1.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    gini_impurity = float(1.0 - np.sum(p**2))

    # Coverage for common k values
    sorted_norm = np.sort(p)[::-1]
    cumulative = np.cumsum(sorted_norm)

    def cov(k: int) -> float:
        k = min(k, len(sorted_norm))
        return float(cumulative[k - 1]) if k >= 1 else 0.0

    # Smallest number of features to reach 50% mass
    half_k = int(np.searchsorted(cumulative, 0.5) + 1) if len(cumulative) else 0

    summary = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "sum_mean_abs": float(sum_mean_abs),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "gini_impurity": gini_impurity,
        "coverage_top_5": cov(5),
        "coverage_top_10": cov(10),
        "coverage_top_20": cov(20),
        "features_for_50pct": half_k,
    }

    return overall_stats, per_class_stats, summary


def _save_metrics_outputs(
    metrics_dir: str,
    base_filename: str,
    overall_stats: list[dict[str, Any]],
    per_class_stats: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    model_meta: dict[str, Any],
) -> list[str]:
    """Persist metrics as CSV/JSON. Returns list of saved file paths."""
    saved: list[str] = []
    ensure_dir(metrics_dir)

    # Overall feature stats CSV/JSON
    overall_csv = os.path.join(metrics_dir, f"{base_filename}_feature_stats.csv")
    overall_json = os.path.join(metrics_dir, f"{base_filename}_feature_stats.json")
    try:
        pd = _safe_import_pandas()
        if pd is not None:
            df_overall = pd.DataFrame(overall_stats)
            df_overall.to_csv(overall_csv, index=False)
            saved.append(overall_csv)
        else:
            # csv fallback
            if overall_stats:
                with open(overall_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(overall_stats[0].keys()))
                    writer.writeheader()
                    writer.writerows(overall_stats)
                saved.append(overall_csv)
        with open(overall_json, "w") as f:
            json.dump(overall_stats, f)
        saved.append(overall_json)
    except Exception:
        pass

    # Per-class CSV
    if per_class_stats:
        per_class_csv = os.path.join(
            metrics_dir, f"{base_filename}_feature_stats_per_class.csv"
        )
        try:
            rows: list[dict[str, Any]] = []
            for cls, lst in per_class_stats.items():
                for d in lst:
                    rows.append({"class": cls, **d})
            if rows:
                pd = _safe_import_pandas()
                if pd is not None:
                    pd.DataFrame(rows).to_csv(per_class_csv, index=False)
                else:
                    with open(per_class_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
                saved.append(per_class_csv)
        except Exception:
            pass

    # Summary JSON (include model_meta)
    try:
        summary_path = os.path.join(metrics_dir, f"{base_filename}_summary.json")
        payload = {"summary": summary, "model": model_meta}
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2)
        saved.append(summary_path)
    except Exception:
        pass

    return saved


def _rankdata_avg(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for ties (Spearman helper)."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # Handle ties by averaging ranks
    unique_vals, first_idx = np.unique(values[order], return_index=True)
    for i, v in enumerate(unique_vals):
        start = first_idx[i]
        end = first_idx[i + 1] if i + 1 < len(first_idx) else len(values)
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    return _pearson_corr(rx, ry)


def _compute_cross_model_similarity(
    model_to_importance: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    names = list(model_to_importance.keys())
    results: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_map, b_map = model_to_importance[a_name], model_to_importance[b_name]
            shared_features = sorted(set(a_map.keys()) & set(b_map.keys()))
            if not shared_features:
                results.append(
                    {
                        "model_a": a_name,
                        "model_b": b_name,
                        "n_shared": 0,
                        "pearson": float("nan"),
                        "spearman": float("nan"),
                        "jaccard_top10": float("nan"),
                        "jaccard_top20": float("nan"),
                        "overlap_top10": 0,
                        "overlap_top20": 0,
                    }
                )
                continue
            a_vec = np.array([a_map[f] for f in shared_features])
            b_vec = np.array([b_map[f] for f in shared_features])
            pearson = _pearson_corr(a_vec, b_vec)
            spearman = _spearman_corr(a_vec, b_vec)

            # Top-k set overlaps
            def topk_set(m: dict[str, float], k: int) -> set[str]:
                feats = sorted(m.items(), key=lambda kv: kv[1], reverse=True)
                return set([f for f, _ in feats[:k]])

            a_top10 = topk_set(a_map, 10)
            b_top10 = topk_set(b_map, 10)
            a_top20 = topk_set(a_map, 20)
            b_top20 = topk_set(b_map, 20)

            j10_den = len(a_top10 | b_top10) or 1
            j20_den = len(a_top20 | b_top20) or 1
            jaccard_top10 = len(a_top10 & b_top10) / j10_den
            jaccard_top20 = len(a_top20 & b_top20) / j20_den

            results.append(
                {
                    "model_a": a_name,
                    "model_b": b_name,
                    "n_shared": len(shared_features),
                    "pearson": float(pearson),
                    "spearman": float(spearman),
                    "jaccard_top10": float(jaccard_top10),
                    "jaccard_top20": float(jaccard_top20),
                    "overlap_top10": len(a_top10 & b_top10),
                    "overlap_top20": len(a_top20 & b_top20),
                }
            )
    return results


def run_shap_for_predictor(
    predictor, sample_X, out_dir: str = os.path.join("data", "predictions", "shap")
) -> str | None:
    """Compute SHAP plots and print rich diagnostics for predictor models.

    Generates beeswarm and bar summary plots for:
    - Outcome classifier (per class when applicable)
    - Home goals regressor
    - Away goals regressor

    Also prints detailed diagnostics to the console using `rich`:
    - Environment (package versions) and dataset summary
    - Model presence and types
    - Top features by mean |SHAP| for each model

    Returns the output directory path if successful, otherwise None.
    """
    ok, err = _ensure_shap_matplotlib_loaded()
    if not ok or shap is None or plt is None:
        _print(
            "[bold yellow]SHAP or matplotlib not installed or failed to import. Skipping SHAP analysis.[/bold yellow]"
        )
        if err:
            _print(f"[yellow]Detail: {err}[/yellow]")
        return None

    ensure_dir(out_dir)

    # Sampling for performance
    original_rows = len(sample_X)
    if original_rows > 2000:
        X = sample_X.sample(2000, random_state=42)
    else:
        X = sample_X.copy()

    rows, cols, mem_bytes = _summarize_dataframe(X)
    label_classes = list(
        getattr(getattr(predictor, "label_encoder", None), "classes_", [])
    )

    # Header panel
    if console is not None and Panel is not None:
        console.print(
            Panel.fit("[bold cyan]SHAP Analysis[/bold cyan]", border_style="cyan")
        )

    # Environment & inputs table
    if console is not None and Table is not None:
        env_table = Table(title="Environment & Inputs", box=box.SIMPLE_HEAVY)
        env_table.add_column("Key", style="bold")
        env_table.add_column("Value")
        env_table.add_row("SHAP version", getattr(shap, "__version__", "unknown"))
        try:
            import matplotlib  # type: ignore

            mpl_ver = getattr(matplotlib, "__version__", "unknown")
        except Exception:
            mpl_ver = "unknown"
        env_table.add_row("matplotlib version", mpl_ver)
        env_table.add_row("Output directory", str(out_dir))
        env_table.add_row("Rows (sampled)", str(rows))
        env_table.add_row("Columns", str(cols))
        env_table.add_row(
            "Memory (MB)", f"{(mem_bytes/1e6):.2f}" if mem_bytes >= 0 else "unknown"
        )
        env_table.add_row(
            "Class labels", ", ".join(map(str, label_classes)) or "<none>"
        )
        console.print(env_table)

        # Feature schema table (up to 30 feature names)
        feature_names = list(getattr(X, "columns", []))
        if feature_names:
            feat_schema = Table(
                title="Feature schema (first 30)", box=box.MINIMAL_HEAVY_HEAD
            )
            feat_schema.add_column("#", justify="right")
            feat_schema.add_column("Feature name")
            for idx, name in enumerate(feature_names[:30], start=1):
                feat_schema.add_row(str(idx), str(name))
            console.print(feat_schema)

    # Model summary table scaffold
    model_table = None
    if console is not None and Table is not None:
        model_table = Table(title="Models", box=box.SIMPLE_HEAVY)
        model_table.add_column("Model", style="bold")
        model_table.add_column("Present")
        model_table.add_column("Type")
        model_table.add_column("Explainer")
        model_table.add_column("Saved plots")
        model_table.add_column("Saved metrics")

    # Processing helper
    metrics_root_dir = os.path.join(out_dir, "metrics")
    ensure_dir(metrics_root_dir)
    model_to_importance: dict[str, dict[str, float]] = {}

    def _process_model(
        model_attr: str,
        display_name: str,
        is_classifier: bool,
        base_filename: str,
    ) -> None:
        model = getattr(predictor, model_attr, None)
        if model is None:
            if model_table is not None:
                model_table.add_row(display_name, "no", "-", "-", "-", "-")
            return

        explainer, explainer_mode = _try_build_explainer(
            model, is_classifier=is_classifier, background_X=X
        )
        if explainer is None:
            if model_table is not None:
                model_table.add_row(
                    display_name,
                    "yes",
                    type(model).__name__,
                    explainer_mode,
                    "<error>",
                    "-",
                )
            _print(f"[red]Could not create SHAP explainer for {display_name}[/red]")
            return

        with _status(f"Computing SHAP values for {display_name}..."):
            try:
                shap_values = _compute_shap_values(explainer, X)
            except Exception as e:  # pragma: no cover
                if model_table is not None:
                    model_table.add_row(
                        display_name,
                        "yes",
                        type(model).__name__,
                        explainer_mode,
                        f"error: {e}",
                    )
                _print(f"[red]Failed SHAP computation for {display_name}: {e}[/red]")
                return

        # Save plots
        safe_base = os.path.join(out_dir, base_filename)
        class_names = label_classes if is_classifier else None
        saved_plots = _save_summary_plots(
            shap_values, X, safe_base, class_names=class_names
        )

        # Compute advanced metrics and save
        try:
            feature_names_full = (
                list(X.columns)
                if hasattr(X, "columns")
                else [f"f{i}" for i in range(X.shape[1])]
            )
            overall_stats, per_class_stats, summary = _compute_featurewise_stats(
                shap_values,
                X,
                feature_names_full,
                is_classifier,
                class_names,
            )
        except Exception as e:
            overall_stats, per_class_stats, summary = [], {}, {}
            _print(
                f"[yellow]Failed to compute detailed SHAP metrics for {display_name}: {e}[/yellow]"
            )

        saved_metrics: list[str] = []
        if overall_stats:
            try:
                model_meta = {
                    "model_attr": model_attr,
                    "display_name": display_name,
                    "type": type(model).__name__,
                    "explainer": explainer_mode,
                    "rows": rows,
                    "cols": cols,
                }
                saved_metrics = _save_metrics_outputs(
                    metrics_root_dir,
                    base_filename,
                    overall_stats,
                    per_class_stats,
                    summary,
                    model_meta,
                )
            except Exception:
                pass

            # Add normalized importances for cross-model similarity
            importance_map = {
                d["feature"]: float(d["importance_normalized"]) for d in overall_stats
            }
            model_to_importance[display_name] = importance_map

            # Print metrics summary table
            if console is not None and Table is not None and summary:
                smry = Table(
                    title=f"Metrics · {display_name}", box=box.MINIMAL_HEAVY_HEAD
                )
                smry.add_column("Metric", style="bold")
                smry.add_column("Value", justify="right")

                def fmt(v: Any) -> str:
                    try:
                        if isinstance(v, (int,)):
                            return str(v)
                        if isinstance(v, float):
                            return f"{v:.6f}"
                        return str(v)
                    except Exception:
                        return str(v)

                for k in [
                    "n_samples",
                    "n_features",
                    "sum_mean_abs",
                    "entropy",
                    "normalized_entropy",
                    "gini_impurity",
                    "coverage_top_5",
                    "coverage_top_10",
                    "coverage_top_20",
                    "features_for_50pct",
                ]:
                    if k in summary:
                        smry.add_row(k, fmt(summary[k]))
                console.print(smry)

            # Print top normalized features
            if console is not None and Table is not None and overall_stats:
                top_norm = overall_stats[:20]
                norm_table = Table(
                    title=f"Top normalized importance · {display_name}",
                    box=box.MINIMAL_HEAVY_HEAD,
                )
                norm_table.add_column("Rank", justify="right")
                norm_table.add_column("Feature", style="bold")
                norm_table.add_column("Norm. Importance", justify="right")
                for item in top_norm:
                    norm_table.add_row(
                        str(item.get("rank", "")),
                        item["feature"],
                        f"{item['importance_normalized']:.6f}",
                    )
                console.print(norm_table)

            # Per-class top features (if classifier)
            if console is not None and Table is not None and per_class_stats:
                for cls_name, rows_cls in per_class_stats.items():
                    view = rows_cls[:10]
                    cls_table = Table(
                        title=f"Top features · {display_name} · class={cls_name}",
                        box=box.MINIMAL_HEAVY_HEAD,
                    )
                    cls_table.add_column("Rank", justify="right")
                    cls_table.add_column("Feature", style="bold")
                    cls_table.add_column("Mean |SHAP|", justify="right")
                    cls_table.add_column("Norm.", justify="right")
                    for r in view:
                        cls_table.add_row(
                            str(r.get("rank", "")),
                            r["feature"],
                            f"{r['mean_abs']:.6f}",
                            f"{r['importance_normalized']:.6f}",
                        )
                    console.print(cls_table)

        # Print top features by mean |SHAP|
        try:
            feature_names = (
                list(X.columns)
                if hasattr(X, "columns")
                else [f"f{i}" for i in range(X.shape[1])]
            )
            top_k = _compute_mean_abs_importance(shap_values, feature_names)[:20]
        except Exception:
            top_k = []

        if console is not None and Table is not None and top_k:
            feat_table = Table(
                title=f"Top features · {display_name}", box=box.MINIMAL_HEAVY_HEAD
            )
            feat_table.add_column("Rank", justify="right")
            feat_table.add_column("Feature", style="bold")
            feat_table.add_column("Mean |SHAP|", justify="right")
            for idx, (fname, val) in enumerate(top_k, start=1):
                feat_table.add_row(str(idx), str(fname), f"{val:.6f}")
            console.print(feat_table)

        if model_table is not None:
            model_table.add_row(
                display_name,
                "yes",
                type(model).__name__,
                explainer_mode,
                "\n".join(os.path.relpath(p, start=out_dir) for p in saved_plots)
                if saved_plots
                else "(no plots)",
                "\n".join(os.path.relpath(p, start=out_dir) for p in saved_metrics)
                if saved_metrics
                else "(no metrics)",
            )

    # Run for each model
    _process_model("outcome_model", "Outcome classifier", True, "shap_outcome")
    _process_model("home_goals_model", "Home goals regressor", False, "shap_home_goals")
    _process_model("away_goals_model", "Away goals regressor", False, "shap_away_goals")

    # Cross-model similarity metrics
    if len(model_to_importance) >= 2:
        try:
            cross = _compute_cross_model_similarity(model_to_importance)
        except Exception as e:
            cross = []
            _print(f"[yellow]Failed to compute cross-model similarity: {e}[/yellow]")

        if cross:
            # Save
            try:
                cross_csv = os.path.join(metrics_root_dir, "cross_model_similarity.csv")
                cross_json = os.path.join(
                    metrics_root_dir, "cross_model_similarity.json"
                )
                pd = _safe_import_pandas()
                if pd is not None:
                    pd.DataFrame(cross).to_csv(cross_csv, index=False)
                else:
                    with open(cross_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
                        writer.writeheader()
                        writer.writerows(cross)
                with open(cross_json, "w") as f:
                    json.dump(cross, f, indent=2)
            except Exception:
                pass

            # Console table
            if console is not None and Table is not None:
                ct = Table(title="Cross-model similarity", box=box.SIMPLE_HEAVY)
                ct.add_column("A", style="bold")
                ct.add_column("B", style="bold")
                ct.add_column("Shared", justify="right")
                ct.add_column("Pearson", justify="right")
                ct.add_column("Spearman", justify="right")
                ct.add_column("Jacc@10", justify="right")
                ct.add_column("Jacc@20", justify="right")
                ct.add_column("Overlap@10", justify="right")
                ct.add_column("Overlap@20", justify="right")
                for row in cross:

                    def f(v: Any) -> str:
                        if isinstance(v, (int,)):
                            return str(v)
                        try:
                            return f"{float(v):.4f}"
                        except Exception:
                            return str(v)

                    ct.add_row(
                        str(row.get("model_a", "")),
                        str(row.get("model_b", "")),
                        str(row.get("n_shared", "")),
                        f(row.get("pearson", float("nan"))),
                        f(row.get("spearman", float("nan"))),
                        f(row.get("jaccard_top10", float("nan"))),
                        f(row.get("jaccard_top20", float("nan"))),
                        str(row.get("overlap_top10", "")),
                        str(row.get("overlap_top20", "")),
                    )
                console.print(ct)

    if model_table is not None:
        console.print(model_table)

    _print("[green]SHAP analysis complete.[/green]")
    return out_dir
