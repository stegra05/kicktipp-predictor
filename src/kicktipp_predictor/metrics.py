import os
import json
from typing import Dict, List, Tuple

import numpy as np

try:  # optional plotting deps
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


LABELS_ORDER: Tuple[str, str, str] = ("H", "D", "A")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def brier_score_multiclass(y_true: List[str], proba: np.ndarray) -> float:
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(y)
    Y = np.zeros_like(P)
    valid = (y >= 0)
    if n:
        Y[np.arange(n)[valid], y[valid]] = 1.0
    diffs = (P - Y)[valid]
    if len(diffs) == 0:
        return float("nan")
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


def log_loss_multiclass(y_true: List[str], proba: np.ndarray) -> float:
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    valid = (y >= 0)
    idx = np.arange(len(y))[valid]
    if len(idx) == 0:
        return float("nan")
    p_true = P[idx, y[valid]]
    return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))


def ranked_probability_score_3c(y_true: List[str], proba: np.ndarray) -> float:
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    valid = (y >= 0)
    if not np.any(valid):
        return float("nan")
    scores = []
    for i in np.where(valid)[0]:
        cdf_pred = np.cumsum(P[i, :])
        cdf_true = np.zeros(3)
        cdf_true[y[i]:] = 1.0
        scores.append(np.sum((cdf_pred - cdf_true) ** 2) / 2.0)
    return float(np.mean(scores))


def expected_calibration_error(y_true: List[str], proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y = np.array([mapping.get(label, -1) for label in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    ece: Dict[str, float] = {}
    for li, lab in enumerate(LABELS_ORDER):
        probs = P[:, li]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(probs, bins, right=True)
        acc_sum = 0.0
        for b in range(1, n_bins + 1):
            mask = (bin_ids == b)
            if not np.any(mask):
                continue
            conf = np.mean(probs[mask])
            acc = float(np.mean((y[mask] == li)))
            w = np.mean(mask)
            acc_sum += w * abs(acc - conf)
        ece[lab] = float(acc_sum)
    return ece


def reliability_diagram(y_true: List[str], proba: np.ndarray, class_label: str, n_bins: int = 10):
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
        mask = (bin_ids == b)
        if not np.any(mask):
            rows.append({
                'bin': f"[{bins[b-1]:.1f},{bins[b]:.1f})",
                'avg_confidence': float('nan'),
                'accuracy': float('nan'),
                'count': 0,
            })
        else:
            rows.append({
                'bin': f"[{bins[b-1]:.2f},{bins[b]:.2f})",
                'avg_confidence': float(np.mean(probs[mask])),
                'accuracy': float(np.mean(y_idx[mask] == li)),
                'count': int(np.sum(mask)),
            })
    return pd.DataFrame(rows)


def plot_reliability_curve(df, class_label: str, out_path: str) -> None:
    if plt is None:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    xs = []
    ys = []
    for _, r in df.iterrows():
        try:
            rng = r['bin'].strip('[]').split(',')
            xs.append((float(rng[0]) + float(rng[1])) / 2)
            ys.append(r['accuracy'])
        except Exception:
            xs.append(np.nan)
            ys.append(np.nan)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.scatter(xs, ys, label=f'{class_label}')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration - {class_label}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def confusion_matrix_stats(y_true: List[str], proba: np.ndarray) -> Dict:
    mapping = {lab: i for i, lab in enumerate(LABELS_ORDER)}
    y_idx = np.array([mapping.get(lbl, -1) for lbl in y_true], dtype=int)
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y_pred = np.argmax(P, axis=1)
    valid = (y_idx >= 0)
    y_true_v = y_idx[valid]
    y_pred_v = y_pred[valid]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true_v, y_pred_v):
        cm[t, p] += 1
    acc = float(np.mean(y_true_v == y_pred_v)) if len(y_true_v) else float('nan')
    per_class: Dict[str, Dict[str, float]] = {}
    for i, lab in enumerate(LABELS_ORDER):
        tp = cm[i, i]
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        per_class[lab] = {'precision': float(precision), 'recall': float(recall)}
    return {'matrix': cm.tolist(), 'accuracy': acc, 'per_class': per_class}


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    if plt is None:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(list(LABELS_ORDER))
    ax.set_yticklabels(list(LABELS_ORDER))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bin_by_confidence(conf: np.ndarray, y_true: List[str], proba: np.ndarray, points: np.ndarray, n_bins: int = 5):
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
        mask = (bin_ids == b)
        count = int(np.sum(mask))
        if count == 0:
            rows.append({'bin': f"[{bins[b-1]:.2f},{bins[b]:.2f})", 'count': 0, 'avg_points': float('nan'), 'accuracy': float('nan'), 'avg_confidence': float('nan')})
        else:
            avg_pts = float(np.mean(points[mask]))
            acc = float(np.mean(np.argmax(P[mask], axis=1) == y_idx[mask]))
            avg_conf = float(np.mean(conf[mask]))
            rows.append({'bin': f"[{bins[b-1]:.2f},{bins[b]:.2f})", 'count': count, 'avg_points': avg_pts, 'accuracy': acc, 'avg_confidence': avg_conf})
    return pd.DataFrame(rows)


def plot_confidence_buckets(df, out_path: str) -> None:
    if plt is None:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(7, 4))
    xs = list(range(len(df)))
    plt.bar(xs, df['avg_points'], color='#4C78A8')
    plt.xticks(xs, df['bin'], rotation=45, ha='right')
    plt.ylabel('Avg points')
    plt.title('Confidence buckets')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_points(pred_home, pred_away, act_home, act_away):
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
    pred_outcome = np.where(ph > pa, 'H', np.where(pa > ph, 'A', 'D'))
    act_outcome = np.where(ah > aa, 'H', np.where(aa > ah, 'A', 'D'))
    points[(pred_outcome == act_outcome) & rem] = 2
    return points


