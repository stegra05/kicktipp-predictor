"""Model evaluation metrics and utilities.

This module provides:
- Probability metrics: Brier score, log loss, Ranked Probability Score (RPS)
- Calibration metrics: Expected Calibration Error (ECE)
- Confusion matrix statistics
- Confidence binning analysis
- Kicktipp points calculation
- Lightweight I/O helpers (JSON saving, directory creation)

Public functions preserve the original API and semantics while the
implementation is organized into logical classes and helpers.
"""

# === Imports ===
import json
import os
from typing import Dict, List, Tuple

import numpy as np

# === Constants ===
LABELS_ORDER: Tuple[str, str, str] = ("H", "D", "A")


# === Utilities ===
class MetricsUtils:
    """Internal helpers for label mapping and probability normalization."""

    @staticmethod
    def label_mapping() -> Dict[str, int]:
        """Return a consistent label-to-index mapping using LABELS_ORDER."""
        return {lab: i for i, lab in enumerate(LABELS_ORDER)}

    @staticmethod
    def labels_to_indices(y_true: List[str]) -> np.ndarray:
        """Map label strings to integer indices, -1 for unknown labels."""
        mapping = MetricsUtils.label_mapping()
        return np.array([mapping.get(label, -1) for label in y_true], dtype=int)

    @staticmethod
    def normalize_proba(proba: np.ndarray) -> np.ndarray:
        """Clip and row-normalize predicted probabilities safely.

        - Clips values to [1e-15, 1.0] to avoid log(0) and instability.
        - Normalizes rows so each row sums to 1.
        - Works with any number of classes; current code uses 3.
        """
        P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
        row_sums = P.sum(axis=1, keepdims=True)
        # Prevent division by zero in pathological inputs
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return P / row_sums


# === File I/O ===
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, out_path: str) -> None:
    """Save a dictionary to a JSON file.

    Ensures parent directory exists and writes UTF-8 JSON with indentation.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# === Probability Metrics ===
class ProbabilityMetrics:
    """Probability-based scoring rules for multiclass predictions."""

    @staticmethod
    def brier_score_multiclass(y_true: List[str], proba: np.ndarray) -> float:
        """Mean squared error between predicted probabilities and one-hot truth.

        Returns NaN if no valid labels.
        """
        y = MetricsUtils.labels_to_indices(y_true)
        P = MetricsUtils.normalize_proba(proba)
        n = len(y)
        Y = np.zeros_like(P)
        valid = y >= 0
        if n:
            Y[np.arange(n)[valid], y[valid]] = 1.0
        diffs = (P - Y)[valid]
        if len(diffs) == 0:
            return float("nan")
        return float(np.mean(np.sum(diffs * diffs, axis=1)))

    @staticmethod
    def log_loss_multiclass(y_true: List[str], proba: np.ndarray) -> float:
        """Multiclass logarithmic loss.

        Penalizes confident but wrong predictions more heavily; lower is better.
        Returns NaN if no valid labels.
        """
        y = MetricsUtils.labels_to_indices(y_true)
        P = MetricsUtils.normalize_proba(proba)
        valid = y >= 0
        idx = np.arange(len(y))[valid]
        if len(idx) == 0:
            return float("nan")
        p_true = P[idx, y[valid]]
        return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))

    @staticmethod
    def ranked_probability_score_3c(y_true: List[str], proba: np.ndarray) -> float:
        """Ranked Probability Score (RPS) for ordered 3-class outcomes.

        Computes squared distance between cumulative distributions (pred vs true),
        averaged over samples; lower is better. Returns NaN if no valid labels.
        """
        y = MetricsUtils.labels_to_indices(y_true)
        P = MetricsUtils.normalize_proba(proba)
        valid = y >= 0
        if not np.any(valid):
            return float("nan")
        scores: List[float] = []
        # Construct true CDF per sample from the integer class index
        for i in np.where(valid)[0]:
            cdf_pred = np.cumsum(P[i, :])
            cdf_true = np.zeros(P.shape[1], dtype=float)
            cdf_true[y[i] :] = 1.0
            scores.append(float(np.sum((cdf_pred - cdf_true) ** 2) / 2.0))
        return float(np.mean(scores))


# === Calibration Metrics ===
class CalibrationMetrics:
    """Calibration assessment utilities (ECE and reliability-style bins)."""

    @staticmethod
    def expected_calibration_error(
        y_true: List[str], proba: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute per-class Expected Calibration Error (ECE).

        ECE is the weighted average over bins of |accuracy - confidence|.
        """
        y = MetricsUtils.labels_to_indices(y_true)
        P = MetricsUtils.normalize_proba(proba)
        ece: Dict[str, float] = {}
        for class_index, label in enumerate(LABELS_ORDER):
            probs = P[:, class_index]
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_ids = np.digitize(probs, bins, right=True)
            acc_sum = 0.0
            for b in range(1, n_bins + 1):
                mask = bin_ids == b
                if not np.any(mask):
                    continue
                conf = float(np.mean(probs[mask]))
                acc = float(np.mean(y[mask] == class_index))
                w = float(np.mean(mask))
                acc_sum += w * abs(acc - conf)
            ece[label] = float(acc_sum)
        return ece


# === Confusion Matrix ===
class ConfusionMetrics:
    """Confusion matrix and derived statistics (accuracy, precision, recall)."""

    @staticmethod
    def _compute_matrix(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> np.ndarray:
        """Build a 3x3 confusion matrix from indexes for LABELS_ORDER."""
        cm = np.zeros((len(LABELS_ORDER), len(LABELS_ORDER)), dtype=int)
        for t, p in zip(y_true_idx, y_pred_idx):
            if t >= 0 and p >= 0:
                cm[t, p] += 1
        return cm

    @staticmethod
    def confusion_matrix_stats(y_true: List[str], proba: np.ndarray) -> dict:
        """Return confusion matrix, accuracy, and per-class precision/recall."""
        y_idx = MetricsUtils.labels_to_indices(y_true)
        P = MetricsUtils.normalize_proba(proba)
        y_pred = np.argmax(P, axis=1)
        valid = y_idx >= 0
        y_true_v = y_idx[valid]
        y_pred_v = y_pred[valid]
        cm = ConfusionMetrics._compute_matrix(y_true_v, y_pred_v)
        acc = float(np.mean(y_true_v == y_pred_v)) if len(y_true_v) else float("nan")
        per_class: Dict[str, Dict[str, float]] = {}
        for i, lab in enumerate(LABELS_ORDER):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
            per_class[lab] = {"precision": precision, "recall": recall}
        return {"matrix": cm.tolist(), "accuracy": acc, "per_class": per_class}


# === Confidence Binning ===
class ConfidenceAnalysis:
    """Group performance by prediction confidence bins."""

    @staticmethod
    def bin_by_confidence(
        conf: np.ndarray,
        y_true: List[str],
        proba: np.ndarray,
        points: np.ndarray,
        n_bins: int = 5,
    ):
        """Summarize average points and accuracy across confidence bins.

        Returns a pandas DataFrame with per-bin statistics.
        """
        import pandas as pd

        P = MetricsUtils.normalize_proba(proba)
        y_idx = MetricsUtils.labels_to_indices(y_true)
        conf = np.asarray(conf, dtype=float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.clip(np.digitize(conf, bins, right=True), 1, n_bins)
        rows = []
        for b in range(1, n_bins + 1):
            mask = bin_ids == b
            count = int(np.sum(mask))
            if count == 0:
                rows.append(
                    {
                        "bin": f"[{bins[b - 1]:.2f},{bins[b]:.2f})",
                        "count": 0,
                        "avg_points": float("nan"),
                        "accuracy": float("nan"),
                        "avg_confidence": float("nan"),
                    }
                )
            else:
                avg_pts = float(np.mean(points[mask]))
                acc = float(np.mean(np.argmax(P[mask], axis=1) == y_idx[mask]))
                avg_conf = float(np.mean(conf[mask]))
                rows.append(
                    {
                        "bin": f"[{bins[b - 1]:.2f},{bins[b]:.2f})",
                        "count": count,
                        "avg_points": avg_pts,
                        "accuracy": acc,
                        "avg_confidence": avg_conf,
                    }
                )
        return pd.DataFrame(rows)


# === Kicktipp Points ===
class KicktippScoring:
    """Scoring rules for Kicktipp predicted vs actual scorelines."""

    @staticmethod
    def compute_points(pred_home, pred_away, act_home, act_away):
        """Calculate points: 4 exact, 3 correct diff, 2 correct outcome, 0 else."""
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
        pred_outcome = np.where(ph > pa, "H", np.where(pa > ph, "A", "D"))
        act_outcome = np.where(ah > aa, "H", np.where(aa > ah, "A", "D"))
        points[(pred_outcome == act_outcome) & rem] = 2
        return points


# === Backward-Compatible Public API (wrappers) ===
def brier_score_multiclass(y_true: List[str], proba: np.ndarray) -> float:
    """Wrapper for ProbabilityMetrics.brier_score_multiclass."""
    return ProbabilityMetrics.brier_score_multiclass(y_true, proba)


def log_loss_multiclass(y_true: List[str], proba: np.ndarray) -> float:
    """Wrapper for ProbabilityMetrics.log_loss_multiclass."""
    return ProbabilityMetrics.log_loss_multiclass(y_true, proba)


def ranked_probability_score_3c(y_true: List[str], proba: np.ndarray) -> float:
    """Wrapper for ProbabilityMetrics.ranked_probability_score_3c."""
    return ProbabilityMetrics.ranked_probability_score_3c(y_true, proba)


def expected_calibration_error(
    y_true: List[str], proba: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Wrapper for CalibrationMetrics.expected_calibration_error."""
    return CalibrationMetrics.expected_calibration_error(y_true, proba, n_bins)


def confusion_matrix_stats(y_true: List[str], proba: np.ndarray) -> dict:
    """Wrapper for ConfusionMetrics.confusion_matrix_stats."""
    return ConfusionMetrics.confusion_matrix_stats(y_true, proba)


def bin_by_confidence(
    conf: np.ndarray,
    y_true: List[str],
    proba: np.ndarray,
    points: np.ndarray,
    n_bins: int = 5,
):
    """Wrapper for ConfidenceAnalysis.bin_by_confidence."""
    return ConfidenceAnalysis.bin_by_confidence(conf, y_true, proba, points, n_bins)


def compute_points(pred_home, pred_away, act_home, act_away):
    """Wrapper for KicktippScoring.compute_points."""
    return KicktippScoring.compute_points(pred_home, pred_away, act_home, act_away)


# NOTE:
# - expected_points_from_grid helper previously removed; EP scoreline logic not used.
