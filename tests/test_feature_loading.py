import time
import types
from pathlib import Path

import pandas as pd
import pytest

from kicktipp_predictor.data import DataLoader


def _build_dummy_df():
    # Minimal DataFrame containing several expected columns from kept_features.yaml
    return pd.DataFrame(
        {
            "match_id": ["1"],
            "matchday": [1],
            "date": [pd.Timestamp("2024-01-01")],
            "home_team": ["A"],
            "away_team": ["B"],
            "is_finished": [True],
            "home_score": [1],
            "away_score": [0],
            # Some features expected in YAML (not all need to exist)
            "tanh_tamed_elo": [0.1],
            "home_form_points_weighted_by_opponent_rank": [1.0],
            "away_form_points_weighted_by_opponent_rank": [0.5],
            "home_form_points_L3": [3.0],
            "away_form_points_L3": [2.0],
        }
    )


def test_load_selected_features_with_yaml(monkeypatch):
    dl = DataLoader()
    # Point config dir to actual package config containing kept_features.yaml
    sel_path = dl.config.paths.config_dir / "kept_features.yaml"
    assert sel_path.exists(), (
        "kept_features.yaml must exist in package config directory"
    )

    # Inject a stub 'yaml' module to simulate availability
    import kicktipp_predictor.data as data_mod

    stub = types.SimpleNamespace()

    def safe_load(text: str):
        # Minimal YAML loader that handles dash lists
        items = []
        for raw in text.splitlines():
            base = raw.split("#", 1)[0].strip()
            if base.startswith("-"):
                items.append(base[1:].strip())
        return items

    stub.safe_load = safe_load
    monkeypatch.setattr(data_mod, "yaml", stub, raising=True)

    feats = dl._load_selected_features(sel_path)
    assert isinstance(feats, list) and len(feats) > 0
    assert "tanh_tamed_elo" in feats

    # Apply selection to a dummy df
    df = _build_dummy_df()
    out = dl._apply_selected_features(df)
    assert "tanh_tamed_elo" in out.columns
    # Meta columns remain
    for col in [
        "match_id",
        "matchday",
        "date",
        "home_team",
        "away_team",
        "is_finished",
    ]:
        assert col in out.columns


def test_load_selected_features_without_yaml(monkeypatch):
    dl = DataLoader()
    sel_path = dl.config.paths.config_dir / "kept_features.yaml"
    assert sel_path.exists(), (
        "kept_features.yaml must exist in package config directory"
    )

    import kicktipp_predictor.data as data_mod

    # Simulate PyYAML unavailable
    monkeypatch.setattr(data_mod, "yaml", None, raising=True)

    feats = dl._load_selected_features(sel_path)
    assert isinstance(feats, list) and len(feats) > 0
    assert "tanh_tamed_elo" in feats


def test_missing_yaml_file_error(tmp_path, monkeypatch):
    dl = DataLoader()
    # Redirect config dir to an empty temp folder
    monkeypatch.setattr(dl.config.paths, "config_dir", tmp_path, raising=True)

    df = _build_dummy_df()
    with pytest.raises(FileNotFoundError) as exc:
        dl._apply_selected_features(df)
    assert "Selected features file not found" in str(exc.value)


def test_malformed_yaml_error(tmp_path, monkeypatch):
    dl = DataLoader()
    # Create malformed file (no dash list, no 'features' key)
    bad = tmp_path / "kept_features.yaml"
    bad.write_text("not: a: valid: yaml", encoding="utf-8")

    monkeypatch.setattr(dl.config.paths, "config_dir", tmp_path, raising=True)

    with pytest.raises(ValueError) as exc:
        dl._load_selected_features(bad)
    assert "No valid features loaded" in str(exc.value)


def test_apply_selected_features_warns_missing(monkeypatch, caplog):
    dl = DataLoader()
    # Create a YAML with one feature not present in df
    tmp = monkeypatch.chdir
    cfg_dir = Path(dl.config.paths.config_dir)
    df = _build_dummy_df()

    # Create a temporary file in a new dir and redirect config_dir
    new_dir = Path(str(cfg_dir)) / "_tmp_test"
    new_dir.mkdir(exist_ok=True)
    yaml_path = new_dir / "kept_features.yaml"
    yaml_path.write_text(
        "# test\n- tanh_tamed_elo\n- non_existing_feature\n", encoding="utf-8"
    )
    monkeypatch.setattr(dl.config.paths, "config_dir", new_dir, raising=True)

    with caplog.at_level("WARNING"):
        out = dl._apply_selected_features(df)
    # Column present
    assert "tanh_tamed_elo" in out.columns
    # Warning for missing feature was logged
    assert any("missing in dataframe" in rec.getMessage() for rec in caplog.records)


def test_feature_loading_performance():
    dl = DataLoader()
    sel_path = dl.config.paths.config_dir / "kept_features.yaml"
    start = time.perf_counter()
    feats = dl._load_selected_features(sel_path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert isinstance(feats, list) and len(feats) > 0
    # Target a reasonable threshold for small files; allow slack for CI
    assert elapsed_ms < 200.0, f"Loading took too long: {elapsed_ms:.2f} ms"
