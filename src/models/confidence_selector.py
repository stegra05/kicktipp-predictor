import os
import math
from functools import lru_cache
from typing import Tuple, List

import numpy as np
import pandas as pd


DEFAULT_METRIC = "confidence"
CANDIDATE_COLUMNS: List[str] = [
    "confidence",           # combined confidence already in predictions
    "margin",               # separation between top-2 outcomes
    "entropy_confidence",   # 1 - normalized entropy
]


def _existing_paths() -> List[str]:
    paths = [
        os.path.join("data", "predictions", "debug_eval.csv"),
        os.path.join("data", "predictions", "debug_season.csv"),
    ]
    return [p for p in paths if os.path.exists(p)]


def _files_signature(paths: List[str]) -> Tuple[Tuple[str, float, int], ...]:
    sig: List[Tuple[str, float, int]] = []
    for p in paths:
        try:
            st = os.stat(p)
            sig.append((p, st.st_mtime, st.st_size))
        except Exception:
            continue
    return tuple(sig)


def _load_debug_frames(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compute_margin_from_probs(row: pd.Series) -> float:
    try:
        probs = [float(row.get("pH", np.nan)), float(row.get("pD", np.nan)), float(row.get("pA", np.nan))]
        probs = [p for p in probs if np.isfinite(p)]
        if len(probs) != 3:
            return float("nan")
        probs.sort(reverse=True)
        return float(probs[0] - probs[1])
    except Exception:
        return float("nan")


def _compute_entropy_conf_from_probs(row: pd.Series) -> float:
    try:
        probs = np.array([row.get("pH", np.nan), row.get("pD", np.nan), row.get("pA", np.nan)], dtype=float)
        if not np.all(np.isfinite(probs)):
            return float("nan")
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        entropy = float(-np.sum(probs * np.log(probs)))
        return float(1.0 - (entropy / math.log(3)))
    except Exception:
        return float("nan")


@lru_cache(maxsize=1)
def get_selected_confidence_metric_cached(files_sig: Tuple[Tuple[str, float, int], ...]) -> str:
    paths = [p for p, _, _ in files_sig]
    df = _load_debug_frames(paths)
    if df.empty or "points" not in df.columns:
        return DEFAULT_METRIC

    # Ensure candidate columns exist or can be derived
    if "margin" not in df.columns:
        df["margin"] = df.apply(_compute_margin_from_probs, axis=1)
    if "entropy_confidence" not in df.columns:
        df["entropy_confidence"] = df.apply(_compute_entropy_conf_from_probs, axis=1)

    results: List[Tuple[str, float]] = []
    for col in CANDIDATE_COLUMNS:
        if col not in df.columns:
            continue
        sub = df[[col, "points"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 30:
            continue
        try:
            # Spearman is robust to monotonic but non-linear relationships
            from scipy.stats import spearmanr  # type: ignore
            rho, _ = spearmanr(sub[col], sub["points"])  # type: ignore
            if np.isfinite(rho):
                results.append((col, float(rho)))
        except Exception:
            # Fallback to Pearson if SciPy unavailable
            try:
                corr = float(np.corrcoef(sub[col].astype(float), sub["points"].astype(float))[0, 1])
                if np.isfinite(corr):
                    results.append((col, corr))
            except Exception:
                continue

    if not results:
        return DEFAULT_METRIC

    # Choose highest absolute correlation; positive correlation preferred tie-breaker
    results.sort(key=lambda x: (abs(x[1]), x[1]), reverse=True)
    best_col = results[0][0]
    return best_col


def get_selected_confidence_metric() -> str:
    paths = _existing_paths()
    sig = _files_signature(paths)
    if not sig:
        return DEFAULT_METRIC
    return get_selected_confidence_metric_cached(sig)


def extract_display_confidence(pred: dict) -> float:
    """
    Extract the confidence value to display (0..1) from a prediction dict
    using the selected metric. Falls back to DEFAULT_METRIC and then 0.
    """
    metric = get_selected_confidence_metric()
    val = pred.get(metric)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    # Fallback to default if different metric not present
    val = pred.get(DEFAULT_METRIC)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    return 0.0


