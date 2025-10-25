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
    """Select the scoreline that maximizes expected Kicktipp points.

    The expected points are computed by evaluating all (home, away) scorelines
    within a Poisson joint probability grid up to ``max_goals``.

    Scoring rules:
    - 4 points: exact scoreline match
    - 3 points: correct goal difference
    - 2 points: correct outcome (home/draw/away)
    - 0 points: otherwise

    Args:
        home_lambda: Expected goals (Poisson mean) for the home team.
        away_lambda: Expected goals (Poisson mean) for the away team.
        max_goals: Maximum goals to consider per side for the grid.

    Returns:
        A tuple ``(pred_home, pred_away)`` representing the selected scoreline.
    """
    G = int(max(0, max_goals))
    lam_h = float(home_lambda)
    lam_a = float(away_lambda)

    x = np.arange(G + 1)
    ph = poisson.pmf(x, lam_h)
    pa = poisson.pmf(x, lam_a)
    grid = np.outer(ph, pa)

    total = float(np.sum(grid))
    if total <= 0.0:
        return (1, 1)
    grid = grid / total

    EP = np.zeros_like(grid, dtype=float)
    for ph_pred in range(G + 1):
        for pa_pred in range(G + 1):
            exp_pts = 0.0
            for h in range(G + 1):
                for a in range(G + 1):
                    if ph_pred == h and pa_pred == a:
                        pts = 4
                    elif (ph_pred - pa_pred) == (h - a):
                        pts = 3
                    elif (
                        (ph_pred > pa_pred and h > a)
                        or (ph_pred == pa_pred and h == a)
                        or (ph_pred < pa_pred and h < a)
                    ):
                        pts = 2
                    else:
                        pts = 0
                    exp_pts += grid[h, a] * pts
            EP[ph_pred, pa_pred] = exp_pts

    mx = float(np.max(EP))
    cand = np.argwhere(EP == mx)
    if len(cand) == 1:
        return int(cand[0][0]), int(cand[0][1])

    # Tie-break by grid probability, then by common scorelines
    best = (int(cand[0][0]), int(cand[0][1]))
    best_p = float(grid[best[0], best[1]])
    for c in cand[1:]:
        p = float(grid[int(c[0]), int(c[1])])
        if p > best_p + 1e-12:
            best = (int(c[0]), int(c[1]))
            best_p = p

    common_scorelines = [
        (2, 1),
        (1, 0),
        (1, 1),
        (0, 1),
        (2, 0),
        (0, 0),
        (2, 2),
        (3, 1),
        (1, 2),
        (3, 0),
        (0, 3),
        (3, 2),
        (2, 3),
    ]
    candidates_set = {(int(c[0]), int(c[1])) for c in cand}
    for h, a in common_scorelines:
        if (h, a) in candidates_set and h <= G and a <= G:
            return h, a
    return best