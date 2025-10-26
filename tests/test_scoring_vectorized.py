import time
import numpy as np

from kicktipp_predictor.scoring import compute_ep_scoreline
from scipy.stats import poisson


def _compute_ep_scoreline_slow(home_lambda: float, away_lambda: float, max_goals: int):
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


def test_ep_scoreline_vectorized_matches_slow_random():
    rng = np.random.default_rng(123)
    # Test a variety of lambda pairs and goal caps
    for G in [3, 5, 8, 10]:
        for _ in range(12):
            lam_h = float(rng.uniform(0.0, 3.0))
            lam_a = float(rng.uniform(0.0, 3.0))
            h_vec, a_vec = compute_ep_scoreline(lam_h, lam_a, G)
            h_slow, a_slow = _compute_ep_scoreline_slow(lam_h, lam_a, G)
            assert (h_vec, a_vec) == (h_slow, a_slow)


def test_ep_scoreline_performance_speedup():
    # Measure speed ratio on a larger grid to expose O(G^4) cost
    lam_h, lam_a, G = 1.8, 1.2, 16

    # Warmup
    compute_ep_scoreline(lam_h, lam_a, G)
    _compute_ep_scoreline_slow(lam_h, lam_a, G)

    # Timings
    t0 = time.perf_counter()
    for _ in range(3):
        _ = _compute_ep_scoreline_slow(lam_h, lam_a, G)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(200):
        _ = compute_ep_scoreline(lam_h, lam_a, G)
    t3 = time.perf_counter()

    slow_time = (t1 - t0) / 3.0
    fast_time = (t3 - t2) / 200.0
    # Expect very large speedup; sufficiently high threshold to be meaningful
    assert slow_time / max(fast_time, 1e-12) > 100.0