from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# Optional plotting deps
try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore
    sns = None  # type: ignore


LABELS_ORDER: Tuple[str, str, str] = ("H", "D", "A")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_label_vector(results: Iterable[str]) -> np.ndarray:
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    return np.array([mapping.get(str(r), -1) for r in results], dtype=int)


def _proba_array(proba: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(proba), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("proba must be of shape (n_samples, 3) for H/D/A")
    # clip and renormalize for numerical safety
    arr = np.clip(arr, 1e-15, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def compute_points(pred_home: Iterable[int], pred_away: Iterable[int],
                   act_home: Iterable[int], act_away: Iterable[int]) -> np.ndarray:
    ph = np.asarray(list(pred_home), dtype=int)
    pa = np.asarray(list(pred_away), dtype=int)
    ah = np.asarray(list(act_home), dtype=int)
    aa = np.asarray(list(act_away), dtype=int)

    points = np.zeros_like(ph, dtype=int)

    exact = (ph == ah) & (pa == aa)
    points[exact] = 4

    not_exact = ~exact
    gd_ok = ((ph - pa) == (ah - aa)) & not_exact
    points[gd_ok] = 3

    rem = ~(exact | gd_ok)
    pred_w = np.where(ph > pa, "H", np.where(pa > ph, "A", "D"))
    act_w = np.where(ah > aa, "H", np.where(aa > ah, "A", "D"))
    points[(pred_w == act_w) & rem] = 2

    return points


def brier_score_multiclass(y_true: Iterable[str], proba: Iterable[Iterable[float]]) -> float:
    y = _to_label_vector(y_true)
    P = _proba_array(proba)
    n = len(y)
    Y = np.zeros_like(P)
    valid = (y >= 0)
    Y[np.arange(n)[valid], y[valid]] = 1.0
    # ignore invalid rows
    diffs = (P - Y)[valid]
    if len(diffs) == 0:
        return float("nan")
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


def log_loss_multiclass(y_true: Iterable[str], proba: Iterable[Iterable[float]]) -> float:
    y = _to_label_vector(y_true)
    P = _proba_array(proba)
    valid = (y >= 0)
    idx = np.arange(len(y))[valid]
    if len(idx) == 0:
        return float("nan")
    p_true = P[idx, y[valid]]
    return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))


def ranked_probability_score_3c(y_true: Iterable[str], proba: Iterable[Iterable[float]]) -> float:
    """
    RPS for ordered outcomes [H, D, A]. Lower is better.
    RPS = (1/(K-1)) * sum_k (CDF_pred(k) - CDF_true(k))^2 for K=3.
    """
    y = _to_label_vector(y_true)
    P = _proba_array(proba)
    valid = (y >= 0)
    if not np.any(valid):
        return float("nan")
    # Build true cumulative for each row
    # For class c, true CDF is 1 at and after c
    K = 3
    cdf_pred = np.cumsum(P, axis=1)
    # true one-hot then cumulative
    Y = np.zeros_like(P)
    Y[np.arange(len(y))[valid], y[valid]] = 1.0
    cdf_true = np.cumsum(Y, axis=1)
    diffs = (cdf_pred - cdf_true)[valid]
    rps = (1.0 / (K - 1)) * np.sum(diffs * diffs, axis=1)
    return float(np.mean(rps))


@dataclass
class CalibrationCurve:
    bins: np.ndarray  # bin centers
    mean_pred: np.ndarray
    mean_true: np.ndarray
    counts: np.ndarray


def reliability_diagram(y_true: Iterable[str], proba: Iterable[Iterable[float]],
                        cls: str, n_bins: int = 10) -> CalibrationCurve:
    y = _to_label_vector(y_true)
    P = _proba_array(proba)
    if cls not in LABELS_ORDER:
        raise ValueError("cls must be one of 'H','D','A'")
    cidx = LABELS_ORDER.index(cls)
    p = P[:, cidx]
    valid = (y >= 0)
    y01 = (y == cidx).astype(int)
    p = p[valid]
    y01 = y01[valid]
    if len(p) == 0:
        return CalibrationCurve(np.array([]), np.array([]), np.array([]), np.array([]))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    mean_pred = np.zeros(n_bins)
    mean_true = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = (inds == b)
        if np.any(mask):
            mean_pred[b] = float(np.mean(p[mask]))
            mean_true[b] = float(np.mean(y01[mask]))
            counts[b] = int(np.sum(mask))
        else:
            mean_pred[b] = np.nan
            mean_true[b] = np.nan
    centers = 0.5 * (bins[:-1] + bins[1:])
    return CalibrationCurve(centers, mean_pred, mean_true, counts)


def expected_calibration_error(y_true: Iterable[str], proba: Iterable[Iterable[float]], n_bins: int = 10) -> Dict[str, float]:
    eces: Dict[str, float] = {}
    y = _to_label_vector(y_true)
    P = _proba_array(proba)
    valid = (y >= 0)
    if not np.any(valid):
        return {lab: float("nan") for lab in LABELS_ORDER}
    weights_total = np.sum(valid)
    for cls in LABELS_ORDER:
        curve = reliability_diagram(y_true, proba, cls, n_bins=n_bins)
        # ECE: sum over bins of (|acc - conf| * bin_weight)
        acc = curve.mean_true
        conf = curve.mean_pred
        w = curve.counts.astype(float)
        mask = np.isfinite(acc) & np.isfinite(conf) & (w > 0)
        if not np.any(mask):
            eces[cls] = float("nan")
            continue
        eces[cls] = float(np.sum(np.abs(acc[mask] - conf[mask]) * (w[mask] / weights_total)))
    return eces


def plot_reliability_curve(curve: CalibrationCurve, cls: str, out_path: str) -> Optional[str]:
    if plt is None or sns is None:  # pragma: no cover - optional dependency
        return None
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(5, 5))
    # Perfect calibration
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    mask = np.isfinite(curve.mean_true) & np.isfinite(curve.mean_pred)
    plt.plot(curve.mean_pred[mask], curve.mean_true[mask], marker='o', label=f'{cls}')
    sizes = 50 * (curve.counts[mask] / max(1, np.max(curve.counts)))
    plt.scatter(curve.mean_pred[mask], curve.mean_true[mask], s=sizes, alpha=0.6)
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title(f'Reliability diagram - {cls}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def confusion_matrix_stats(y_true: Iterable[str], proba: Iterable[Iterable[float]]) -> Dict[str, object]:
    y = _to_label_vector(y_true)
    P = _proba_array(proba)
    valid = (y >= 0)
    pred = np.argmax(P, axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for i in np.where(valid)[0]:
        cm[y[i], pred[i]] += 1
    acc = float(np.trace(cm) / max(1, np.sum(cm)))
    per_class = {}
    for i, lab in enumerate(LABELS_ORDER):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        per_class[lab] = {"precision": prec, "recall": rec}
    return {"matrix": cm.tolist(), "accuracy": acc, "per_class": per_class}


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> Optional[str]:
    if plt is None or sns is None:  # pragma: no cover - optional dependency
        return None
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_ORDER, yticklabels=LABELS_ORDER)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def bin_by_confidence(conf: Iterable[float], y_true: Iterable[str], proba: Iterable[Iterable[float]],
                      points: Iterable[int], n_bins: int = 5) -> pd.DataFrame:
    c = np.asarray(list(conf), dtype=float)
    y = list(y_true)
    P = _proba_array(proba)
    pts = np.asarray(list(points), dtype=float)

    # Quantile bins
    if len(c) == 0:
        return pd.DataFrame(columns=["bin", "count", "avg_points", "accuracy", "avg_confidence"])
    quantiles = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    # Handle identical values case
    quantiles = np.unique(quantiles)
    if len(quantiles) <= 2:
        # Fallback to equal width bins
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(c, quantiles, right=True) - 1
    inds = np.clip(inds, 0, len(quantiles) - 2)

    pred_labels = [LABELS_ORDER[i] for i in np.argmax(P, axis=1)]
    acc_vec = np.array([1.0 if a == b else 0.0 for a, b in zip(y, pred_labels)], dtype=float)

    rows = []
    for b in range(len(quantiles) - 1):
        mask = inds == b
        if not np.any(mask):
            continue
        rows.append({
            "bin": f"[{quantiles[b]:.2f},{quantiles[b+1]:.2f}]",
            "count": int(np.sum(mask)),
            "avg_points": float(np.mean(pts[mask])),
            "accuracy": float(np.mean(acc_vec[mask])),
            "avg_confidence": float(np.mean(c[mask])),
        })
    return pd.DataFrame(rows)


def plot_confidence_buckets(df: pd.DataFrame, out_path: str) -> Optional[str]:
    if plt is None or sns is None:  # pragma: no cover - optional dependency
        return None
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df, x="bin", y="avg_points", color="#4C78A8")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Avg points")
    plt.title("Points by confidence bucket")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_json(data: Dict[str, object], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



