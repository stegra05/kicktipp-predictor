import os
import yaml
import pytest

from kicktipp_predictor.config import get_config
from kicktipp_predictor.data import DataLoader

REMOVED_FEATURES = {
    "abs_momentum_score_diff",
    "momentum_score_difference",
    "venue_points_delta",
    "venue_goals_delta",
    "venue_conceded_delta",
}

TRANSIENT_ELO = {"home_elo", "away_elo", "elo_diff"}


def test_training_features_do_not_include_removed_columns():
    loader = DataLoader()
    current = loader.get_current_season()
    all_matches = loader.fetch_historical_seasons(current - 3, current - 1)
    df = loader.create_features_from_matches(all_matches)
    cols = set(df.columns)
    # Removed features must be absent
    assert REMOVED_FEATURES.isdisjoint(cols)
    # Transient raw elo columns must be absent
    assert TRANSIENT_ELO.isdisjoint(cols)
    # Stable signal should remain available
    assert "tanh_tamed_elo" in cols


def test_prediction_features_do_not_include_removed_columns():
    loader = DataLoader()
    current = loader.get_current_season()
    historical = loader.fetch_season_matches(current)
    # Prefer upcoming matches; fallback to matchday 1 if none
    upcoming = loader.get_upcoming_matches(days=7)
    if not upcoming:
        upcoming = loader.fetch_matchday(1)
    df = loader.create_prediction_features(upcoming, historical)
    cols = set(df.columns)
    assert REMOVED_FEATURES.isdisjoint(cols)
    assert TRANSIENT_ELO.isdisjoint(cols)
    assert "tanh_tamed_elo" in cols


def test_legacy_form_function_is_absent():
    from kicktipp_predictor import data as data_module
    assert not hasattr(data_module, "_get_form_features"), "Legacy form function should not exist"


def test_kept_features_yaml_does_not_list_removed_features():
    cfg = get_config()
    sel_path = cfg.paths.config_dir / getattr(cfg.model, "selected_features_file", "kept_features.yaml")
    assert sel_path.exists(), "kept_features.yaml should exist"
    with open(sel_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if isinstance(loaded, list):
        listed = set(map(str, loaded))
    elif isinstance(loaded, dict) and "features" in loaded:
        val = loaded.get("features")
        if isinstance(val, list):
            listed = set(map(str, val))
        elif isinstance(val, str):
            listed = set(s.strip() for s in val.splitlines() if s.strip())
        else:
            listed = set()
    elif isinstance(loaded, str):
        listed = set(s.strip() for s in loaded.splitlines() if s.strip())
    else:
        listed = set()
    assert REMOVED_FEATURES.isdisjoint(listed), "Removed features should not be listed in kept_features.yaml"