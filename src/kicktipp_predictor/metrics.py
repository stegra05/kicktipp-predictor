"""Model evaluation metrics and utilities.

This module provides a collection of functions and classes for evaluating the
performance of the prediction model. It includes metrics for assessing the
accuracy of probabilistic predictions, calibration, and the quality of the
confusion matrix. It also provides a function for calculating points based on
the Kicktipp scoring system.
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
    """Provides internal helper functions for metric calculations."""

    @staticmethod
    def label_mapping() -> Dict[str, int]:
        """Returns a consistent mapping from labels to integer indices."""
        return {lab: i for i, lab in enumerate(LABELS_ORDER)}

    @staticmethod
    def labels_to_indices(y_true: List[str]) -> np.ndarray:
        """Converts a list of label strings to an array of integer indices."""
        mapping = MetricsUtils.label_mapping()
        return np.array([mapping.get(label, -1) for label in y_true], dtype=int)

    @staticmethod
    def normalize_proba(proba: np.ndarray) -> np.ndarray:
        """Safely clips and normalizes predicted probabilities.

        This function ensures that probabilities are within the range [1e-15, 1.0]
        and that each row sums to 1.
        """
        P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
        row_sums = P.sum(axis=1, keepdims=True)
        # Prevent division by zero in pathological inputs
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return P / row_sums


# === File I/O ===
def ensure_dir(path: str) -> None:
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, out_path: str) -> None:
    """Saves a dictionary to a JSON file.

    This function ensures that the parent directory of the output path exists
    and writes the dictionary to a UTF-8 encoded JSON file with indentation.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# === Probability Metrics ===
class ProbabilityMetrics:
    """A collection of probability-based scoring rules for multiclass predictions."""

    @staticmethod
    def brier_score_multiclass(y_true: List[str], proba: np.ndarray) -> float:
        """Calculates the Brier score for multiclass predictions.

        The Brier score measures the mean squared error between the predicted
        probabilities and the actual outcomes.
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
        """Calculates the logarithmic loss for multiclass predictions.

        Log loss penalizes confident but incorrect predictions more heavily.
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
        """Calculates the Ranked Probability Score (RPS) for 3-class predictions.

        RPS is a measure of the squared distance between the cumulative
        distributions of the predicted and actual outcomes.
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
    """A collection of utilities for assessing model calibration."""

    @staticmethod
    def expected_calibration_error(
        y_true: List[str], proba: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculates the Expected Calibration Error (ECE) for each class.

        ECE measures the difference between the predicted probability and the
        actual accuracy within different confidence bins.
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
    """Provides functions for calculating confusion matrices and related stats."""

    @staticmethod
    def _compute_matrix(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> np.ndarray:
        """Builds a 3x3 confusion matrix from integer indices."""
        cm = np.zeros((len(LABELS_ORDER), len(LABELS_ORDER)), dtype=int)
        for t, p in zip(y_true_idx, y_pred_idx):
            if t >= 0 and p >= 0:
                cm[t, p] += 1
        return cm

    @staticmethod
    def confusion_matrix_stats(y_true: List[str], proba: np.ndarray) -> dict:
        """Calculates the confusion matrix, accuracy, and per-class stats.

        This function returns a dictionary containing the confusion matrix,
        overall accuracy, and precision and recall for each class.
        """
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
    """Provides functions for analyzing performance across confidence bins."""

    @staticmethod
    def bin_by_confidence(
        conf: np.ndarray,
        y_true: List[str],
        proba: np.ndarray,
        points: np.ndarray,
        n_bins: int = 5,
    ):
        """Summarizes performance metrics across different confidence bins.

        This function groups predictions into bins based on their confidence
        and calculates the average points, accuracy, and confidence for each bin.
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
    """Implements the Kicktipp scoring system."""

    @staticmethod
    def compute_points(pred_home, pred_away, act_home, act_away):
        """Calculates Kicktipp points for a set of predictions.

        The scoring is as follows:
        - 4 points for the exact score.
        - 3 points for the correct goal difference.
        - 2 points for the correct outcome (win/draw/loss).
        - 0 points otherwise.
        """
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
