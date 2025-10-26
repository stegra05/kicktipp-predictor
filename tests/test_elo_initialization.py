import sys
import pathlib
from datetime import datetime, timedelta

# Ensure src layout is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd
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


def test_enhanced_initialization_promoted_vs_relegated():
    dl = DataLoader()
    dl._elo_avg_k = 2  # average over top/bottom 2 teams for clearer separation

    # Season 2021: include team G for prior presence (no need for extensive matches)
    base_date = datetime(2021, 8, 1)
    matches_2021 = [
        make_match("2021-1", 2021, base_date, 1, "G", "X", 1, 1, finished=True),
    ]

    # Season 2022 standings: A > B > C > D (clear order)
    base_date = datetime(2022, 8, 1)
    matches_2022 = [
        make_match("2022-1", 2022, base_date + timedelta(days=1), 1, "A", "D", 2, 0, True),
        make_match("2022-2", 2022, base_date + timedelta(days=2), 1, "B", "D", 1, 0, True),
        make_match("2022-3", 2022, base_date + timedelta(days=3), 1, "C", "D", 1, 0, True),
        make_match("2022-4", 2022, base_date + timedelta(days=4), 2, "A", "B", 2, 1, True),
        make_match("2022-5", 2022, base_date + timedelta(days=5), 2, "A", "C", 3, 1, True),
        make_match("2022-6", 2022, base_date + timedelta(days=6), 2, "B", "C", 1, 0, True),
    ]

    elo_hist = dl._compute_elo_history(matches_2021 + matches_2022)
    by_season = dl._group_matches_by_season(matches_2021 + matches_2022)

    prev_final_elos = elo_hist["season_final"].get(2022, {})
    prev_table = dl._calculate_table(by_season.get(2022, []))

    # Upcoming season 2023 teams: keep A, B; add E, F (promoted); re-add G (relegated)
    teams_2023 = {"A", "B", "E", "F", "G"}
    prev_teams_2022 = dl._teams_in_season(by_season.get(2022, []))
    prior_presence = dl._build_prior_presence([2021], by_season)

    start_elos_2023 = dl._compute_initial_elos_for_season(
        2023,
        teams_2023,
        prev_teams_2022,
        prev_final_elos,
        prev_table,
        prior_presence,
        dl._elo_avg_k,
    )

    # Compute expected bases to compare against
    top_avg, bottom_avg = dl._compute_prev_season_bases(prev_final_elos, prev_table, dl._elo_avg_k)

    # Promoted teams get bottom_avg; relegated (G) gets top_avg; carry-over keeps own
    assert np.isclose(start_elos_2023["E"], bottom_avg, atol=1e-6)
    assert np.isclose(start_elos_2023["F"], bottom_avg, atol=1e-6)
    assert np.isclose(start_elos_2023["G"], top_avg, atol=1e-6)
    assert start_elos_2023["A"] != start_elos_2023["B"] or start_elos_2023["A"] != bottom_avg


def test_tanh_tamed_elo_present_in_training_features():
    dl = DataLoader()
    dl._elo_avg_k = 2
    base_date = datetime(2022, 8, 1)
    matches_2022 = [
        make_match("2022-1", 2022, base_date + timedelta(days=1), 1, "A", "D", 2, 0, True),
        make_match("2022-2", 2022, base_date + timedelta(days=2), 1, "B", "D", 1, 0, True),
        make_match("2022-3", 2022, base_date + timedelta(days=3), 1, "C", "D", 1, 0, True),
        make_match("2022-4", 2022, base_date + timedelta(days=4), 2, "A", "B", 2, 1, True),
        make_match("2022-5", 2022, base_date + timedelta(days=5), 2, "A", "C", 3, 1, True),
        make_match("2022-6", 2022, base_date + timedelta(days=6), 2, "B", "C", 1, 0, True),
    ]

    features_df = dl.create_features_from_matches(matches_2022)
    assert "tanh_tamed_elo" in features_df.columns
    # Elo-based feature should have variability
    assert features_df["tanh_tamed_elo"].abs().sum() > 0.0


def test_prediction_features_include_elo_diff_and_tanh():
    dl = DataLoader()
    dl._elo_avg_k = 2
    base_date = datetime(2022, 8, 1)
    hist_matches = [
        make_match("2022-1", 2022, base_date + timedelta(days=1), 1, "A", "D", 2, 0, True),
        make_match("2022-2", 2022, base_date + timedelta(days=2), 1, "B", "D", 1, 0, True),
        make_match("2022-3", 2022, base_date + timedelta(days=3), 1, "C", "D", 1, 0, True),
        make_match("2022-4", 2022, base_date + timedelta(days=4), 2, "A", "B", 2, 1, True),
        make_match("2022-5", 2022, base_date + timedelta(days=5), 2, "A", "C", 3, 1, True),
        make_match("2022-6", 2022, base_date + timedelta(days=6), 2, "B", "C", 1, 0, True),
    ]

    upcoming_date = datetime(2023, 8, 1)
    upcoming_matches = [
        make_match("2023-1", 2023, upcoming_date + timedelta(days=1), 1, "A", "E", 0, 0, finished=False),
        make_match("2023-2", 2023, upcoming_date + timedelta(days=2), 1, "B", "F", 0, 0, finished=False),
    ]

    pred_df = dl.create_prediction_features(upcoming_matches, hist_matches)
    # Ensure elo_diff exists for upcoming and tanh_tamed_elo computed
    assert "elo_diff" in pred_df.columns
    assert "tanh_tamed_elo" in pred_df.columns