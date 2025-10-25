def test_compute_ep_scoreline_basic():
    from kicktipp_predictor.scoring import compute_ep_scoreline

    # Basic sanity: returns integers within the grid bounds
    h, a = compute_ep_scoreline(home_lambda=1.2, away_lambda=0.8, max_goals=5)
    assert isinstance(h, int)
    assert isinstance(a, int)
    assert 0 <= h <= 5
    assert 0 <= a <= 5


def test_compute_ep_scoreline_degenerate_grid():
    from kicktipp_predictor.scoring import compute_ep_scoreline

    # With max_goals=0 and zero lambdas, only (0,0) is available
    h, a = compute_ep_scoreline(home_lambda=0.0, away_lambda=0.0, max_goals=0)
    assert (h, a) == (0, 0)