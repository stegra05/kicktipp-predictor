import numpy as np
import pytest
import yaml
from pathlib import Path

from kicktipp_predictor.config import get_config, reset_config, PROJECT_ROOT
from kicktipp_predictor.predictor import MatchPredictor, compute_ep_scoreline


def test_config_load_time_decay_and_momentum():
    # Ensure fresh config instance
    reset_config()
    cfg = get_config()

    params_path = PROJECT_ROOT / "config" / "best_params.yaml"
    assert params_path.exists()
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Validate that tuned YAML values are reflected in the loaded config
    if "time_decay_half_life_days" in params:
        assert pytest.approx(cfg.model.time_decay_half_life_days) == params[
            "time_decay_half_life_days"
        ]
    if "momentum_decay" in params:
        assert pytest.approx(cfg.model.momentum_decay) == params["momentum_decay"]

    # Basic sanity checks
    assert isinstance(cfg.model.use_time_decay, bool)
    assert cfg.model.time_decay_half_life_days > 0
    if "form_last_n" in params:
        assert cfg.model.form_last_n == int(params["form_last_n"])


def test_hybrid_blending_fixed_weight_extremes():
    reset_config()
    cfg = get_config()

    # Configure deterministic blending behavior
    cfg.model.prob_source = "hybrid"
    cfg.model.hybrid_scheme = "fixed"
    cfg.model.hybrid_poisson_weight = 1.0  # test Poisson-only first
    cfg.model.proba_temperature = 1.0
    cfg.model.draw_boost = 1.0
    cfg.model.prior_anchor_enabled = False
    cfg.model.prior_blend_alpha = 0.0
    cfg.model.calibrator_enabled = False

    predictor = MatchPredictor(cfg)

    # One sample: classifier vs Poisson
    clf = np.array([[0.2, 0.5, 0.3]])
    hg = np.array([1.0])
    ag = np.array([1.0])

    # Weight 1 -> pure Poisson outcome
    p1, _ = predictor._derive_final_probabilities(clf, hg, ag)
    p_poisson = predictor._calculate_poisson_outcome_probs(hg, ag)
    assert np.allclose(p1, p_poisson, atol=1e-8)

    # Weight 0 -> pure classifier outcome
    cfg.model.hybrid_poisson_weight = 0.0
    p0, _ = predictor._derive_final_probabilities(clf, hg, ag)
    assert np.allclose(p0, clf, atol=1e-8)


def test_calibration_and_anchoring_identity():
    reset_config()
    cfg = get_config()
    cfg.model.calibrator_enabled = False
    cfg.model.prior_anchor_enabled = False

    predictor = MatchPredictor(cfg)

    proba = np.array([[0.3, 0.4, 0.3], [0.2, 0.2, 0.6]])

    out = predictor._apply_calibration_and_anchoring(proba)
    assert np.allclose(out, proba, atol=1e-8)


def test_compute_ep_scoreline_symmetry_returns_draw():
    reset_config()
    cfg = get_config()

    # Disable draw correlation adjustments for determinism
    cfg.model.poisson_draw_rho = 0.0
    cfg.model.dixon_coles_rho = 0.0

    s = compute_ep_scoreline(0.8, 0.8, max_goals=6, draw_rho=0.0, joint="independent", dixon_rho=0.0)
    assert isinstance(s, tuple)
    assert len(s) == 2
    # For symmetric lambdas, best EP scoreline should be a draw
    assert s[0] == s[1]
    assert 0 <= s[0] <= 6