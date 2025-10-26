import pathlib
import sys
from datetime import datetime, timedelta

import pandas as pd

# Ensure src layout is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from kicktipp_predictor.data import DataLoader


def make_match(match_id, season, date, matchday, home, away, hs, as_, finished=True):
    return {
        "match_id": str(match_id),
        "season": int(season),
        "date": pd.to_datetime(date).isoformat(),
        "matchday": int(matchday),
        "home_team": str(home),
        "away_team": str(away),
        "home_score": int(hs) if finished else None,
        "away_score": int(as_) if finished else None,
        "is_finished": bool(finished),
    }


def test_training_features_empty_input_returns_empty_df():
    dl = DataLoader()
    df = dl.create_features_from_matches([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_training_features_targets_and_diffs_consistency():
    dl = DataLoader()
    base_date = datetime(2022, 8, 1)
    matches = [
        make_match("m1", 2022, base_date + timedelta(days=1), 1, "A", "B", 2, 1, True),
        make_match("m2", 2022, base_date + timedelta(days=2), 1, "C", "D", 0, 0, True),
        make_match("m3", 2022, base_date + timedelta(days=3), 2, "B", "A", 1, 3, True),
    ]

    df = dl.create_features_from_matches(matches)
    assert not df.empty
    # Targets
    assert "goal_difference" in df.columns
    assert "result" in df.columns
    # Consistency of goal_difference
    assert (df["goal_difference"] == (df["home_score"] - df["away_score"])) .all()
    # Result mapping
    res_map = df["result"].tolist()
    assert set(res_map) <= {"H", "D", "A"}
    # Derived diffs
    h_w = df.get("home_form_points_weighted_by_opponent_rank").fillna(0)
    a_w = df.get("away_form_points_weighted_by_opponent_rank").fillna(0)
    wdiff = df.get("weighted_form_points_difference").fillna(0)
    awdiff = df.get("abs_weighted_form_points_diff").fillna(0)
    assert ((h_w - a_w) - wdiff).abs().max() < 1e-9
    assert ((h_w - a_w).abs() - awdiff).abs().max() < 1e-9
    # Elo-based feature present
    assert "tanh_tamed_elo" in df.columns


def test_prediction_features_asof_merge_and_no_targets():
    dl = DataLoader()
    base_date = datetime(2022, 8, 1)
    hist_matches = [
        make_match("h1", 2022, base_date + timedelta(days=1), 1, "A", "B", 2, 0, True),
        make_match("h2", 2022, base_date + timedelta(days=2), 1, "C", "D", 1, 0, True),
        make_match("h3", 2022, base_date + timedelta(days=3), 2, "A", "C", 1, 1, True),
        make_match("h4", 2022, base_date + timedelta(days=4), 2, "B", "D", 0, 3, True),
    ]
    upcoming_date = datetime(2023, 8, 1)
    upcoming_matches = [
        make_match("u1", 2023, upcoming_date + timedelta(days=1), 1, "A", "D", 0, 0, False),
        make_match("u2", 2023, upcoming_date + timedelta(days=2), 1, "C", "B", 0, 0, False),
    ]

    df = dl.create_prediction_features(upcoming_matches, hist_matches)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # As-of merged home/away history presence
    assert "home_form_points_weighted_by_opponent_rank" in df.columns
    assert "away_form_points_weighted_by_opponent_rank" in df.columns
    # Derived diffs
    h_w = df.get("home_form_points_weighted_by_opponent_rank").fillna(0)
    a_w = df.get("away_form_points_weighted_by_opponent_rank").fillna(0)
    wdiff = df.get("weighted_form_points_difference").fillna(0)
    awdiff = df.get("abs_weighted_form_points_diff").fillna(0)
    assert ((h_w - a_w) - wdiff).abs().max() < 1e-9
    assert ((h_w - a_w).abs() - awdiff).abs().max() < 1e-9
    # No target columns in prediction features
    assert "goal_difference" not in df.columns
    assert "result" not in df.columns
    # Elo diff present and tanh_tamed_elo computed
    assert "elo_diff" in df.columns
    assert "tanh_tamed_elo" in df.columns


def test_prediction_features_empty_context_returns_empty():
    dl = DataLoader()
    upcoming_date = datetime(2023, 8, 1)
    upcoming_matches = [
        make_match("u1", 2023, upcoming_date + timedelta(days=1), 1, "A", "D", 0, 0, False),
    ]
    df = dl.create_prediction_features(upcoming_matches, [])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_training_features_handles_string_match_ids_consistently():
    dl = DataLoader()
    base_date = datetime(2022, 8, 1)
    matches = [
        make_match("2022-001", 2022, base_date + timedelta(days=1), 1, "X", "Y", 1, 0, True),
        make_match("2022-002", 2022, base_date + timedelta(days=2), 1, "Y", "Z", 0, 2, True),
    ]
    df = dl.create_features_from_matches(matches)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Merge on match_id and teams should succeed and produce home_* and away_* history columns
    assert any(col.startswith("home_form_") for col in df.columns)
    assert any(col.startswith("away_form_") for col in df.columns)