"""Evaluation metrics and plotting utilities for model assessment.

This module provides functions to:
- Calculate multiclass classification metrics (Brier score, log loss, RPS).
- Assess model calibration (Expected Calibration Error, reliability diagrams).
- Compute and visualize confusion matrices.
- Analyze performance based on prediction confidence.
- Simulate Kicktipp point scoring.
"""

import json
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


LABELS_ORDER: tuple[str, str, str] = ("H", "D", "A")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, out_path: str) -> None:
    """Save a dictionary to a JSON file."""
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def brier_score_multiclass(y_true: list[str], proba: np.ndarray) -> float:
    """Calculate the Brier score for multiclass predictions.

    The Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. A lower score indicates better calibration.

    Args:
        y_true: A list of true labels ('H', 'D', 'A').
        proba: A numpy array of shape (n_samples, 3) with predicted probabilities.

    Returns:
        The calculated Brier score.
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(y)
    Y = np.zeros_like(P)
    valid = y >= 0
    if n:
        Y[np.arange(n)[valid], y[valid]] = 1.0
    diffs = (P - Y)[valid]
    if len(diffs) == 0:
        return float("nan")
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


def log_loss_multiclass(y_true: list[str], proba: np.ndarray) -> float:
    """Calculate the logarithmic loss for multiclass predictions.

    Log loss penalizes confident but incorrect predictions more heavily.
    A lower score is better.

    Args:
        y_true: A list of true labels ('H', 'D', 'A').
        proba: A numpy array of shape (n_samples, 3) with predicted probabilities.

    Returns:
        The calculated log loss.
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    valid = y >= 0
    idx = np.arange(len(y))[valid]
    if len(idx) == 0:
        return float("nan")
    p_true = P[idx, y[valid]]
    return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))


def ranked_probability_score_3c(y_true: list[str], proba: np.ndarray) -> float:
    """Calculate the Ranked Probability Score (RPS) for 3-class outcomes.

    RPS is a proper scoring rule that measures the difference between cumulative
    distribution functions of predictions and outcomes. It is sensitive to distance;
    predicting a home win when the result is a draw is better than predicting an away win.
    A lower score is better.

    Args:
        y_true: A list of true labels ('H', 'D', 'A').
        proba: A numpy array of shape (n_samples, 3) with predicted probabilities.

    Returns:
        The calculated Ranked Probability Score.
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    valid = y >= 0
    if not np.any(valid):
        return float("nan")
    scores = []
    for i in np.where(valid)[0]:
        cdf_pred = np.cumsum(P[i, :])
        cdf_true = np.zeros(3)
        cdf_true[y[i] :] = 1.0
        scores.append(np.sum((cdf_pred - cdf_true) ** 2) / 2.0)
    return float(np.mean(scores))


def expected_calibration_error(
    y_true: list[str], proba: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Calculate the Expected Calibration Error (ECE).

    ECE measures the difference between a model's confidence and its accuracy.
    It is computed by binning predictions by confidence and finding the
    weighted average of the absolute difference between accuracy and confidence
    in each bin. A lower ECE indicates better calibration.

    Args:
        y_true: A list of true labels ('H', 'D', 'A').
        proba: A numpy array of shape (n_samples, 3) with predicted probabilities.
        n_bins: The number of confidence bins to use.

    Returns:
        A dictionary mapping each class label to its ECE value.
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    ece: dict[str, float] = {}
    for li, lab in enumerate(LABELS_ORDER):
        probs = P[:, li]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(probs, bins, right=True)
        acc_sum = 0.0
        for b in range(1, n_bins + 1):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            conf = np.mean(probs[mask])
            acc = float(np.mean(y[mask] == li))
            w = np.mean(mask)
            acc_sum += w * abs(acc - conf)
        ece[lab] = float(acc_sum)
    return ece


def reliability_diagram(
    y_true: list[str], proba: np.ndarray, class_label: str, n_bins: int = 10
):
    """Generate data for a reliability diagram.

    This function bins predictions by confidence and calculates the accuracy
    and average confidence for each bin, which can then be plotted to visualize
    model calibration.

    Args:
        y_true: A list of true labels ('H', 'D', 'A').
        proba: A numpy array of shape (n_samples, 3) with predicted probabilities.
        class_label: The class to generate the diagram for.
        n_bins: The number of confidence bins.

    Returns:
        A pandas DataFrame with reliability diagram data.
    """
    import pandas as pd

    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    li = mapping[class_label]
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    probs = P[:, li]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bins, right=True), 1, n_bins)
    y_idx = np.array([mapping.get(lbl, -1) for lbl in y_true], dtype=int)
    rows = []
    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        if not np.any(mask):
            rows.append(
                {
                    "bin": f"[{bins[b - 1]:.1f},{bins[b]:.1f})",
                    "avg_confidence": float("nan"),
                    "accuracy": float("nan"),
                    "count": 0,
                }
            )
        else:
            rows.append(
                {
                    "bin": f"[{bins[b - 1]:.2f},{bins[b]:.2f})",
                    "avg_confidence": float(np.mean(probs[mask])),
                    "accuracy": float(np.mean(y_idx[mask] == li)),
                    "count": int(np.sum(mask)),
                }
            )
    return pd.DataFrame(rows)


def plot_reliability_curve(df, class_label: str, out_path: str) -> None:
    """Plot a reliability curve from reliability diagram data.

    Args:
        df: A pandas DataFrame from `reliability_diagram`.
        class_label: The class label for the plot title.
        out_path: The file path to save the plot.
    """
    if plt is None:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    xs = []
    ys = []
    for _, r in df.iterrows():
        try:
            rng = r["bin"].strip("[]").split(",")
            xs.append((float(rng[0]) + float(rng[1])) / 2)
            ys.append(r["accuracy"])
        except Exception:
            xs.append(np.nan)
            ys.append(np.nan)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.scatter(xs, ys, label=f"{class_label}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration - {class_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def confusion_matrix_stats(y_true: list[str], proba: np.ndarray) -> dict:
    """Calculate a confusion matrix and related statistics.

    Args:
        y_true: A list of true labels ('H', 'D', 'A').
        proba: A numpy array of shape (n_samples, 3) with predicted probabilities.

    Returns:
        A dictionary containing the confusion matrix, overall accuracy, and
        per-class precision and recall.
    """
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y_idx = np.array([mapping.get(lbl, -1) for lbl in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y_pred = np.argmax(P, axis=1)
    valid = y_idx >= 0
    y_true_v = y_idx[valid]
    y_pred_v = y_pred[valid]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true_v, y_pred_v):
        cm[t, p] += 1
    acc = float(np.mean(y_true_v == y_pred_v)) if len(y_true_v) else float("nan")
    per_class: dict[str, dict[str, float]] = {}
    for i, lab in enumerate(LABELS_ORDER):
        tp = cm[i, i]
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        per_class[lab] = {"precision": float(precision), "recall": float(recall)}
    return {"matrix": cm.tolist(), "accuracy": acc, "per_class": per_class}


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Plot a confusion matrix.

    Args:
        cm: A 3x3 numpy array representing the confusion matrix.
        out_path: The file path to save the plot.
    """
    if plt is None:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(list(LABELS_ORDER))
    ax.set_yticklabels(list(LABELS_ORDER))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bin_by_confidence(
    conf: np.ndarray,
    y_true: list[str],
    proba: np.ndarray,
    points: np.ndarray,
    n_bins: int = 5,
):
    """Group match data into confidence-based bins.

    This function is useful for analyzing model performance at different
    levels of prediction confidence.

    Args:
        conf: An array of confidence scores for each prediction.
        y_true: A list of true labels.
        proba: A numpy array of predicted probabilities.
        points: An array of points awarded for each match.
        n_bins: The number of confidence bins.

    Returns:
        A pandas DataFrame summarizing performance metrics for each bin.
    """
    import pandas as pd

    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y_idx = np.array([mapping.get(lbl, -1) for lbl in y_true], dtype=int)
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


def plot_confidence_buckets(df, out_path: str) -> None:
    """Plot average points scored in each confidence bucket.

    Args:
        df: A pandas DataFrame from `bin_by_confidence`.
        out_path: The file path to save the plot.
    """
    if plt is None:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(7, 4))
    xs = list(range(len(df)))
    plt.bar(xs, df["avg_points"], color="#4C78A8")
    plt.xticks(xs, df["bin"], rotation=45, ha="right")
    plt.ylabel("Avg points")
    plt.title("Confidence buckets")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_points(pred_home, pred_away, act_home, act_away):
    """Calculate Kicktipp points for predicted versus actual scores.

    The scoring rules are:
    - 4 points: Exact scoreline match.
    - 3 points: Correct goal difference.
    - 2 points: Correct outcome (win/draw/loss).
    - 0 points: Incorrect outcome.

    Args:
        pred_home: Predicted home goals.
        pred_away: Predicted away goals.
        act_home: Actual home goals.
        act_away: Actual away goals.

    Returns:
        A numpy array of points for each match.
    """
    import numpy as np

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


def expected_points_from_grid(grid: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Compute maximum expected KickTipp points and corresponding scoreline from a joint grid.

    This helper is not used directly in the pipeline but can support tests/diagnostics.
    """
    import numpy as np

    G = grid.shape[0] - 1
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        return 0.0, (2, 1)
    total = np.sum(grid)
    if total <= 0:
        return 0.0, (2, 1)
    P = grid / total
    pH = float(np.sum(np.triu(P, k=1)))
    pD = float(np.sum(np.diag(P)))
    pA = float(np.sum(np.tril(P, k=-1)))
    sum_by_diff = {d: float(np.sum(np.diagonal(P, offset=d))) for d in range(-G, G + 1)}
    best_ep = -1.0
    best = (2, 1)
    for h in range(G + 1):
        for a in range(G + 1):
            e = float(P[h, a])
            d = h - a
            s = sum_by_diff.get(d, 0.0)
            outcome_p = pD if d == 0 else (pH if d > 0 else pA)
            ep = e + s + 2.0 * outcome_p
            if ep > best_ep:
                best_ep = ep
                best = (h, a)
    return best_ep, best
