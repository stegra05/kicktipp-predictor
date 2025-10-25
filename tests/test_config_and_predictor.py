import numpy as np
import pytest
import yaml
from pathlib import Path

from kicktipp_predictor.config import get_config, reset_config, PROJECT_ROOT
from kicktipp_predictor.predictor import MatchPredictor, compute_ep_scoreline







def test_compute_ep_scoreline_symmetry_returns_draw():
    reset_config()
    get_config()

    s = compute_ep_scoreline(0.8, 0.8, max_goals=6)
    assert isinstance(s, tuple)
    assert len(s) == 2
    # For symmetric lambdas, best EP scoreline should be a draw
    assert s[0] == s[1]
    assert 0 <= s[0] <= 6