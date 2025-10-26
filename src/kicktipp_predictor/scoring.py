"""Scoreline selection utilities for Kicktipp-style expected points.

This module provides helpers to select a predicted scoreline by maximizing
expected Kicktipp points on a Poisson probability grid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import poisson


# === Public API ===

def compute_ep_scoreline(
    home_lambda: float,
    away_lambda: float,
    max_goals: int,
) -> Tuple[int, int]:
    """Select EP-maximizing scoreline without outcome constraint.

    Computes the expected points (EP) grid and returns the scoreline
    maximizing EP with consistent tie-breaking.
    """
    # --- Input validation & setup ---
    try:
        G = int(max_goals)
    except Exception as e:
        raise ValueError("max_goals must be integer-like") from e
    if G < 0:
        G = 0

    lam_h = float(home_lambda)
    lam_a = float(away_lambda)
    if not np.isfinite(lam_h) or lam_h < 0:
        raise ValueError("home_lambda must be finite and >= 0")
    if not np.isfinite(lam_a) or lam_a < 0:
        raise ValueError("away_lambda must be finite and >= 0")

    # --- Poisson joint grid ---
    x = np.arange(G + 1)
    ph = poisson.pmf(x, lam_h).astype(float)
    pa = poisson.pmf(x, lam_a).astype(float)
    grid = np.outer(ph, pa)

    total = float(np.sum(grid))
    if not np.isfinite(total) or total <= 0.0:
        return (1, 1)
    grid = grid / total

    # --- Vectorized probability components ---
    h_idx = np.arange(G + 1)[:, None]
    a_idx = np.arange(G + 1)[None, :]
    diff = h_idx - a_idx

    prob_home = float(grid[diff > 0].sum())
    prob_draw = float(grid[diff == 0].sum())
    prob_away = float(grid[diff < 0].sum())

    pred_home_mask = (h_idx > a_idx)
    pred_draw_mask = (h_idx == a_idx)
    pred_away_mask = (h_idx < a_idx)
    P_result_equal = (
        prob_home * pred_home_mask
        + prob_draw * pred_draw_mask
        + prob_away * pred_away_mask
    )

    diff_indices = (diff + G).astype(int)
    diff_probs = np.bincount(
        diff_indices.ravel(), weights=grid.ravel(), minlength=2 * G + 1
    )
    P_diff_equal = diff_probs[(diff + G)]

    P_exact = grid
    EP = 2.0 * P_result_equal + 1.0 * P_diff_equal + 1.0 * P_exact

    mx = float(np.max(EP))
    if not np.isfinite(mx):
        return (1, 1)

    cand = np.argwhere(EP == mx)
    if cand.shape[0] == 1:
        return int(cand[0, 0]), int(cand[0, 1])

    # Tie-breakers: prefer higher grid prob, then common scorelines
    best = (int(cand[0, 0]), int(cand[0, 1]))
    best_p = float(grid[best[0], best[1]])
    for c in cand[1:]:
        p = float(grid[int(c[0]), int(c[1])])
        if p > best_p + 1e-12:
            best = (int(c[0]), int(c[1]))
            best_p = p

    common_scorelines = [
        (2, 1), (1, 0), (1, 1), (0, 1), (2, 0), (0, 0),
        (2, 2), (3, 1), (1, 2), (3, 0), (0, 3), (3, 2), (2, 3),
    ]
    candidates_set = {(int(c[0]), int(c[1])) for c in cand}
    for h, a in common_scorelines:
        if (h, a) in candidates_set and h <= G and a <= G:
            return h, a

    return best


def compute_ep_scoreline_conditional(
    home_lambda: float,
    away_lambda: float,
    max_goals: int,
    outcome: str,
) -> Tuple[int, int]:
    """Select EP-maximizing scoreline constrained to a predicted outcome.

    Restricts candidate predicted scorelines to those matching the provided
    outcome class ("H", "D", "A"). Falls back to unconstrained selection
    if the constraint yields no valid candidates.
    """
    # --- Input validation & setup ---
    try:
        G = int(max_goals)
    except Exception as e:
        raise ValueError("max_goals must be integer-like") from e
    if G < 0:
        G = 0

    lam_h = float(home_lambda)
    lam_a = float(away_lambda)
    if not np.isfinite(lam_h) or lam_h < 0:
        raise ValueError("home_lambda must be finite and >= 0")
    if not np.isfinite(lam_a) or lam_a < 0:
        raise ValueError("away_lambda must be finite and >= 0")

    # --- Poisson joint grid ---
    x = np.arange(G + 1)
    ph = poisson.pmf(x, lam_h).astype(float)
    pa = poisson.pmf(x, lam_a).astype(float)
    grid = np.outer(ph, pa)

    total = float(np.sum(grid))
    if not np.isfinite(total) or total <= 0.0:
        return (1, 1)
    grid = grid / total

    # --- Vectorized probability components ---
    h_idx = np.arange(G + 1)[:, None]
    a_idx = np.arange(G + 1)[None, :]
    diff = h_idx - a_idx

    prob_home = float(grid[diff > 0].sum())
    prob_draw = float(grid[diff == 0].sum())
    prob_away = float(grid[diff < 0].sum())

    pred_home_mask = (h_idx > a_idx)
    pred_draw_mask = (h_idx == a_idx)
    pred_away_mask = (h_idx < a_idx)
    P_result_equal = (
        prob_home * pred_home_mask
        + prob_draw * pred_draw_mask
        + prob_away * pred_away_mask
    )

    diff_indices = (diff + G).astype(int)
    diff_probs = np.bincount(
        diff_indices.ravel(), weights=grid.ravel(), minlength=2 * G + 1
    )
    P_diff_equal = diff_probs[(diff + G)]

    P_exact = grid
    EP = 2.0 * P_result_equal + 1.0 * P_diff_equal + 1.0 * P_exact

    # --- Apply outcome constraint mask ---
    outcome = (outcome or "").upper()
    if outcome == "H":
        candidate_mask = (h_idx > a_idx)
    elif outcome == "D":
        candidate_mask = (h_idx == a_idx)
    elif outcome == "A":
        candidate_mask = (h_idx < a_idx)
    else:
        # Unknown outcome label: fall back to unconstrained selection
        candidate_mask = np.ones_like(EP, dtype=bool)

    EP_masked = np.where(candidate_mask, EP, -np.inf)
    mx = float(np.max(EP_masked))

    if not np.isfinite(mx) or mx == -np.inf:
        # No valid candidates (e.g., G==0 and outcome is H/A). Fallback.
        return compute_ep_scoreline(lam_h, lam_a, G)

    cand = np.argwhere(EP_masked == mx)
    if cand.shape[0] == 1:
        return int(cand[0, 0]), int(cand[0, 1])

    # Tie-breakers: prefer higher grid prob, then common scorelines
    best = (int(cand[0, 0]), int(cand[0, 1]))
    best_p = float(grid[best[0], best[1]])
    for c in cand[1:]:
        p = float(grid[int(c[0]), int(c[1])])
        if p > best_p + 1e-12:
            best = (int(c[0]), int(c[1]))
            best_p = p

    common_scorelines = [
        (2, 1), (1, 0), (1, 1), (0, 1), (2, 0), (0, 0),
        (2, 2), (3, 1), (1, 2), (3, 0), (0, 3), (3, 2), (2, 3),
    ]
    candidates_set = {(int(c[0]), int(c[1])) for c in cand}
    for h, a in common_scorelines:
        if (h, a) in candidates_set and h <= G and a <= G:
            return h, a

    return best