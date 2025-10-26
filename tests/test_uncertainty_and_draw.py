import numpy as np
import pandas as pd

from kicktipp_predictor.config import Config
from kicktipp_predictor.predictor import GoalDifferencePredictor


class DummyModel:
    def __init__(self, preds: np.ndarray):
        self._preds = np.array(preds, dtype=float)

    def predict(self, X):  # noqa: N802 (sklearn-style)
        # Ignore X, return predefined goal differences
        return self._preds


def _make_features(n: int) -> pd.DataFrame:
    # Minimal numeric-only features to satisfy predictor
    return pd.DataFrame({"f1": np.arange(n, dtype=float)})


def test_dynamic_stddev_increases_with_abs_gd():
    cfg = Config.load()
    cfg.model.gd_uncertainty_base_stddev = 1.0
    cfg.model.gd_uncertainty_scale = 0.5
    cfg.model.gd_uncertainty_min_stddev = 0.1
    cfg.model.gd_uncertainty_max_stddev = 10.0
    cfg.model.draw_margin = 0.5

    predictor = GoalDifferencePredictor(config=cfg)
    predictor.model = DummyModel(np.array([0.0, 1.0, -2.0]))

    feats = _make_features(3)
    preds = predictor.predict(feats)
    stddevs = np.array([p["uncertainty_stddev"] for p in preds], dtype=float)
    # Expected: base + scale * |gd|
    expected = np.array([1.0, 1.5, 2.0], dtype=float)
    assert np.allclose(stddevs, expected, atol=1e-6)


def test_draw_probability_cdf_symmetric_zero_gd():
    cfg = Config.load()
    cfg.model.gd_uncertainty_base_stddev = 1.0
    cfg.model.gd_uncertainty_scale = 0.0  # fixed stddev
    cfg.model.gd_uncertainty_min_stddev = 0.1
    cfg.model.gd_uncertainty_max_stddev = 10.0
    cfg.model.draw_margin = 0.5

    predictor = GoalDifferencePredictor(config=cfg)
    predictor.model = DummyModel(np.array([0.0]))

    feats = _make_features(1)
    preds = predictor.predict(feats)
    p_home = preds[0]["home_win_probability"]
    p_draw = preds[0]["draw_probability"]
    p_away = preds[0]["away_win_probability"]

    # Sum to 1 by construction
    assert np.isfinite(p_home) and np.isfinite(p_draw) and np.isfinite(p_away)
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-12
    # Symmetry at zero GD: p_home == p_away
    assert abs(p_home - p_away) < 1e-6
    # Draw probability should be between the two extremes
    assert p_draw > p_home and p_draw > p_away


def test_stddev_clamped_to_max():
    cfg = Config.load()
    cfg.model.gd_uncertainty_base_stddev = 0.1
    cfg.model.gd_uncertainty_scale = 0.5
    cfg.model.gd_uncertainty_min_stddev = 0.1
    cfg.model.gd_uncertainty_max_stddev = 4.0
    cfg.model.draw_margin = 0.5

    predictor = GoalDifferencePredictor(config=cfg)
    predictor.model = DummyModel(np.array([10.0]))

    feats = _make_features(1)
    preds = predictor.predict(feats)
    stddev = preds[0]["uncertainty_stddev"]
    assert abs(stddev - 4.0) < 1e-9  # clamped to max


def test_numerical_stability_extremes():
    cfg = Config.load()
    cfg.model.gd_uncertainty_base_stddev = 1.0
    cfg.model.gd_uncertainty_scale = 0.5
    cfg.model.gd_uncertainty_min_stddev = 0.1
    cfg.model.gd_uncertainty_max_stddev = 3.0
    cfg.model.draw_margin = 0.5

    predictor = GoalDifferencePredictor(config=cfg)
    predictor.model = DummyModel(np.array([-10.0, 0.0, 10.0]))

    feats = _make_features(3)
    preds = predictor.predict(feats)

    for p in preds:
        probs = np.array([
            p["home_win_probability"],
            p["draw_probability"],
            p["away_win_probability"],
        ], dtype=float)
        assert np.all(np.isfinite(probs))
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert abs(np.sum(probs) - 1.0) < 1e-9
        stddev = float(p["uncertainty_stddev"])
        assert 0.1 - 1e-12 <= stddev <= 3.0 + 1e-12


def test_invalid_params_fallback_to_legacy_stddev():
    cfg = Config.load()
    cfg.model.gd_uncertainty_stddev = 1.5
    cfg.model.gd_uncertainty_base_stddev = -1.0  # invalid, forces fallback
    cfg.model.gd_uncertainty_scale = -2.0        # invalid, forces 0.0
    cfg.model.gd_uncertainty_min_stddev = 0.1
    cfg.model.gd_uncertainty_max_stddev = 5.0
    cfg.model.draw_margin = 0.5

    predictor = GoalDifferencePredictor(config=cfg)
    predictor.model = DummyModel(np.array([0.0]))

    feats = _make_features(1)
    preds = predictor.predict(feats)
    # Fallback to legacy stddev when base <= 0
    assert abs(preds[0]["uncertainty_stddev"] - 1.5) < 1e-9


def test_draw_rate_changes_with_margin():
    cfg = Config.load()
    cfg.model.gd_uncertainty_base_stddev = 1.0
    cfg.model.gd_uncertainty_scale = 0.0
    cfg.model.gd_uncertainty_min_stddev = 0.1
    cfg.model.gd_uncertainty_max_stddev = 10.0

    # Low margin
    cfg.model.draw_margin = 0.2
    predictor_low = GoalDifferencePredictor(config=cfg)
    predictor_low.model = DummyModel(np.zeros(3))
    preds_low = predictor_low.predict(_make_features(3))
    avg_draw_low = float(np.mean([p["draw_probability"] for p in preds_low]))
    # High margin
    cfg.model.draw_margin = 0.8
    predictor_high = GoalDifferencePredictor(config=cfg)
    predictor_high.model = DummyModel(np.zeros(3))
    preds_high = predictor_high.predict(_make_features(3))
    avg_draw_high = float(np.mean([p["draw_probability"] for p in preds_high]))

    assert avg_draw_high > avg_draw_low