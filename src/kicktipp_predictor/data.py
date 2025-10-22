"""Data loading and feature engineering for kicktipp predictor.

This module combines data fetching from OpenLigaDB API with
comprehensive feature engineering for match prediction.
"""

import os
import pickle
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .config import get_config

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class DataLoader:
    """Fetches match data and creates features for prediction."""

    def __init__(self):
        """Initialize with configuration."""
        self.config = get_config()
        self.cache_dir = str(self.config.paths.cache_dir)
        self.base_url = self.config.api.base_url
        self.league_code = self.config.api.league_code
        self.cache_ttl = self.config.api.cache_ttl
        self.timeout = self.config.api.request_timeout

    # =================================================================
    # DATA FETCHING METHODS
    # =================================================================

    def get_current_season(self) -> int:
        """Get current season year (e.g., 2024 for 2024/2025 season)."""
        now = datetime.now()
        # Season typically starts in July/August
        return now.year if now.month >= 7 else now.year - 1

    def fetch_season_matches(self, season: int | None = None) -> list[dict]:
        """Fetch all matches for a given season.

        Args:
            season: Season year (e.g., 2024 for 2024/2025). Defaults to current season.

        Returns:
            List of match dictionaries.
        """
        if season is None:
            season = self.get_current_season()

        cache_file = os.path.join(self.cache_dir, f"matches_{season}.pkl")

        # Check cache
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < self.cache_ttl:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        url = f"{self.base_url}/getmatchdata/{self.league_code}/{season}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            matches_raw = response.json()

            matches = self._parse_matches(matches_raw)

            # Cache the results
            with open(cache_file, "wb") as f:
                pickle.dump(matches, f)

            return matches

        except Exception as e:
            print(f"Error fetching matches for season {season}: {e}")
            # Try to return cached data even if expired
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            return []

    def fetch_matchday(self, matchday: int, season: int | None = None) -> list[dict]:
        """Fetch matches for a specific matchday.

        Args:
            matchday: Matchday number (1-38 for 3. Liga).
            season: Season year. Defaults to current season.

        Returns:
            List of match dictionaries.
        """
        if season is None:
            season = self.get_current_season()

        url = f"{self.base_url}/getmatchdata/{self.league_code}/{season}/{matchday}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            matches_raw = response.json()

            return self._parse_matches(matches_raw)

        except Exception as e:
            print(f"Error fetching matchday {matchday}: {e}")
            return []

    def get_current_matchday(self, season: int | None = None) -> int:
        """Get the current matchday number."""
        matches = self.fetch_season_matches(season)

        now = datetime.now()

        # Find the next unplayed matchday
        for match in matches:
            if match["date"] > now and not match["is_finished"]:
                return match["matchday"]

        # If all matches are finished, return last matchday
        if matches:
            return matches[-1]["matchday"]

        return 1

    def get_upcoming_matches(
        self, days: int = 7, season: int | None = None
    ) -> list[dict]:
        """Get matches in the next N days.

        Args:
            days: Number of days to look ahead.
            season: Season year. Defaults to current season.

        Returns:
            List of upcoming match dictionaries.
        """
        matches = self.fetch_season_matches(season)

        now = datetime.now()
        future_date = now + timedelta(days=days)

        upcoming = [
            m
            for m in matches
            if now <= m["date"] <= future_date and not m["is_finished"]
        ]

        return sorted(upcoming, key=lambda x: x["date"])

    def fetch_historical_seasons(
        self, start_season: int, end_season: int
    ) -> list[dict]:
        """Fetch multiple seasons of historical data for training.

        Args:
            start_season: First season year (e.g., 2019).
            end_season: Last season year (e.g., 2024).

        Returns:
            List of all matches from all seasons.
        """
        all_matches = []

        for season in range(start_season, end_season + 1):
            if os.getenv("KTP_VERBOSE") == "1":
                print(f"Fetching season {season}/{season + 1}...")
            matches = self.fetch_season_matches(season)
            all_matches.extend(matches)
            time.sleep(0.5)  # Be nice to the API

        return all_matches

    def _parse_matches(self, matches_raw: list[dict]) -> list[dict]:
        """Parse raw API response into standardized match dictionaries."""
        matches = []

        for match in matches_raw:
            try:
                match_dict = {
                    "match_id": match.get("matchID"),
                    "matchday": match.get("group", {}).get("groupOrderID", 0),
                    "date": datetime.fromisoformat(
                        match["matchDateTime"].replace("Z", "+00:00")
                    ),
                    "home_team": match["team1"]["teamName"],
                    "away_team": match["team2"]["teamName"],
                    "is_finished": match["matchIsFinished"],
                }

                # Add results if match is finished
                if match["matchIsFinished"] and match["matchResults"]:
                    final_result = [
                        r for r in match["matchResults"] if r["resultTypeID"] == 2
                    ]
                    if final_result:
                        result = final_result[0]
                        match_dict["home_score"] = result["pointsTeam1"]
                        match_dict["away_score"] = result["pointsTeam2"]
                    else:
                        # Fallback to any result
                        result = match["matchResults"][0]
                        match_dict["home_score"] = result["pointsTeam1"]
                        match_dict["away_score"] = result["pointsTeam2"]
                else:
                    match_dict["home_score"] = None
                    match_dict["away_score"] = None

                matches.append(match_dict)

            except Exception as e:
                print(f"Error parsing match: {e}")
                continue

        return matches

    # =================================================================
    # FEATURE ENGINEERING METHODS
    # =================================================================

    def create_features_from_matches(self, matches: list[dict]) -> pd.DataFrame:
        """Create feature dataset from match data using vectorized pandas operations.

        Args:
            matches: List of match dictionaries.

        Returns:
            DataFrame with features and target variables.
        """
        if not matches:
            return pd.DataFrame()

        # Base DataFrame (finished matches only for training)
        base = pd.DataFrame(matches)
        base = base.loc[
            base["is_finished"]
            & base["home_score"].notna()
            & base["away_score"].notna()
        ].copy()
        if base.empty:
            return pd.DataFrame()
        base["date"] = pd.to_datetime(base["date"])

        # Precompute team-long frame and vectorized history features
        long_df = self._build_team_long_df(matches)
        if long_df.empty:
            return pd.DataFrame()
        long_df["date"] = pd.to_datetime(long_df["date"])
        long_df = long_df.sort_values(["team", "date", "match_id"]).reset_index(
            drop=True
        )

        long_df = self._compute_team_history_features(long_df)

        # Precompute EWMA recency features once across the dataset and merge
        try:
            ewma_long_df = self._compute_ewma_recency_features(matches, span=5)
        except Exception:
            ewma_long_df = None

        # Assemble match-level features by merging home/away history rows for the same match_id
        # Select team history columns to attach
        hist_cols = [
            "avg_goals_for",
            "avg_goals_against",
            "avg_points",
            "form_points",
            "form_points_per_game",
            "form_wins",
            "form_draws",
            "form_losses",
            "form_goals_scored",
            "form_goals_conceded",
            "form_goal_diff",
            "form_avg_goals_scored",
            "form_avg_goals_conceded",
            "momentum_points",
            "momentum_goals",
            "momentum_conceded",
            "momentum_score",
            # venue-specific history metrics
            "points_pg_at_home",
            "goals_pg_at_home",
            "goals_conceded_pg_at_home",
            "points_pg_away",
            "goals_pg_away",
            "goals_conceded_pg_away",
        ]
        right_cols = ["match_id", "team"] + hist_cols
        home_merge = long_df[right_cols].rename(columns={"team": "home_team"})
        away_merge = long_df[right_cols].rename(columns={"team": "away_team"})

        features_df = base.copy()
        # Merge home team features
        features_df = features_df.merge(
            home_merge.add_prefix("home_"),
            left_on=["match_id", "home_team"],
            right_on=["home_match_id", "home_home_team"],
            how="left",
        )
        # Merge away team features
        features_df = features_df.merge(
            away_merge.add_prefix("away_"),
            left_on=["match_id", "away_team"],
            right_on=["away_match_id", "away_away_team"],
            how="left",
        )

        # Drop merge helper columns
        drop_cols = [
            "home_match_id",
            "home_home_team",
            "away_match_id",
            "away_away_team",
        ]
        for c in drop_cols:
            if c in features_df.columns:
                features_df.drop(columns=[c], inplace=True)

        # Attach EWMA recency features via vectorized merges
        if ewma_long_df is not None and not ewma_long_df.empty:
            # Home EWMA
            home_ewm = ewma_long_df.rename(
                columns={"team": "home_team", "match_id": "home_match_id"}
            )
            features_df = features_df.merge(
                home_ewm.add_prefix("home_"),
                left_on=["match_id", "home_team"],
                right_on=["home_home_match_id", "home_home_team"],
                how="left",
            )
            # Away EWMA
            away_ewm = ewma_long_df.rename(
                columns={"team": "away_team", "match_id": "away_match_id"}
            )
            features_df = features_df.merge(
                away_ewm.add_prefix("away_"),
                left_on=["match_id", "away_team"],
                right_on=["away_away_match_id", "away_away_team"],
                how="left",
            )
            # Clean helper columns
            for c in [
                "home_home_match_id",
                "home_home_team",
                "away_away_match_id",
                "away_away_team",
            ]:
                if c in features_df.columns:
                    features_df.drop(columns=[c], inplace=True)

            # Rename ewm columns to expected names (home_/away_ prefix placement)
            ewm_map = {
                "goals_for_ewm5": "goals_for_ewm5",
                "goals_against_ewm5": "goals_against_ewm5",
                "goal_diff_ewm5": "goal_diff_ewm5",
                "points_ewm5": "points_ewm5",
            }
            for base_col in ewm_map.values():
                hcol = f"home_{base_col}"
                acol = f"away_{base_col}"
                # Already prefixed by add_prefix; columns exist as home_<col>, away_<col>
                # Ensure they exist
                if hcol not in features_df.columns:
                    features_df[hcol] = 0.0
                if acol not in features_df.columns:
                    features_df[acol] = 0.0

        # Compute simple derived differences
        features_df["abs_form_points_diff"] = (
            features_df.get("home_form_points", 0)
            - features_df.get("away_form_points", 0)
        ).abs()
        features_df["form_points_difference"] = features_df.get(
            "home_form_points", 0
        ) - features_df.get("away_form_points", 0)
        features_df["abs_momentum_score_diff"] = (
            features_df.get("home_momentum_score", 0.0)
            - features_df.get("away_momentum_score", 0.0)
        ).abs()
        features_df["momentum_score_difference"] = features_df.get(
            "home_momentum_score", 0.0
        ) - features_df.get("away_momentum_score", 0.0)

        # Venue deltas
        if all(
            c in features_df.columns
            for c in ["home_points_pg_at_home", "away_points_pg_away"]
        ):
            features_df["venue_points_delta"] = (
                features_df["home_points_pg_at_home"]
                - features_df["away_points_pg_away"]
            )
        if all(
            c in features_df.columns
            for c in ["home_goals_pg_at_home", "away_goals_pg_away"]
        ):
            features_df["venue_goals_delta"] = (
                features_df["home_goals_pg_at_home"] - features_df["away_goals_pg_away"]
            )
        if all(
            c in features_df.columns
            for c in ["home_goals_conceded_pg_at_home", "away_goals_conceded_pg_away"]
        ):
            features_df["venue_conceded_delta"] = (
                features_df["home_goals_conceded_pg_at_home"]
                - features_df["away_goals_conceded_pg_away"]
            )

        # Interaction ratios (safe divisions)
        eps = 1e-6

        def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a.astype(float) / (b.astype(float) + eps)

        if all(
            c in features_df.columns
            for c in ["home_form_avg_goals_scored", "away_form_avg_goals_conceded"]
        ):
            features_df["attack_defense_form_ratio_home"] = _safe_div(
                features_df["home_form_avg_goals_scored"],
                features_df["away_form_avg_goals_conceded"],
            )
        if all(
            c in features_df.columns
            for c in ["away_form_avg_goals_scored", "home_form_avg_goals_conceded"]
        ):
            features_df["attack_defense_form_ratio_away"] = _safe_div(
                features_df["away_form_avg_goals_scored"],
                features_df["home_form_avg_goals_conceded"],
            )
        if all(
            c in features_df.columns
            for c in ["home_avg_goals_for", "away_avg_goals_against"]
        ):
            features_df["attack_defense_long_ratio_home"] = _safe_div(
                features_df["home_avg_goals_for"], features_df["away_avg_goals_against"]
            )
        if all(
            c in features_df.columns
            for c in ["away_avg_goals_for", "home_avg_goals_against"]
        ):
            features_df["attack_defense_long_ratio_away"] = _safe_div(
                features_df["away_avg_goals_for"], features_df["home_avg_goals_against"]
            )
        if all(
            c in features_df.columns
            for c in ["home_form_points_per_game", "away_form_points_per_game"]
        ):
            features_df["form_points_pg_ratio"] = _safe_div(
                features_df["home_form_points_per_game"],
                features_df["away_form_points_per_game"],
            )
        if all(
            c in features_df.columns
            for c in ["home_momentum_score", "away_momentum_score"]
        ):
            features_df["momentum_score_ratio"] = _safe_div(
                features_df["home_momentum_score"], features_df["away_momentum_score"]
            )
        if all(
            c in features_df.columns for c in ["home_points_ewm5", "away_points_ewm5"]
        ):
            features_df["ewm_points_ratio"] = _safe_div(
                features_df["home_points_ewm5"], features_df["away_points_ewm5"]
            )

        # Elo ratings (pre-match), computed leakage-safe from chronological results
        try:
            elo_match_df, elo_long_df = self._compute_elos_from_matches(matches)
        except Exception:
            elo_match_df, elo_long_df = None, None
        if elo_match_df is not None and not elo_match_df.empty:
            try:
                features_df = features_df.merge(
                    elo_match_df[["match_id", "home_elo", "away_elo"]],
                    on="match_id",
                    how="left",
                )
                # Derived Elo difference (home - away)
                features_df["elo_diff"] = features_df.get(
                    "home_elo", 0.0
                ) - features_df.get("away_elo", 0.0)
            except Exception:
                pass

        # Targets
        features_df["goal_difference"] = (
            features_df["home_score"] - features_df["away_score"]
        )
        features_df["result"] = np.where(
            features_df["goal_difference"] > 0,
            "H",
            np.where(features_df["goal_difference"] < 0, "A", "D"),
        )

        # Optionally reduce to selected features if config/<selected_features_file> exists
        features_df = self._apply_selected_features(features_df)

        # Ensure deterministic column order (optional)
        # Optionally reduce to selected features if config/<selected_features_file> exists
        features_df = self._apply_selected_features(features_df)

        return features_df

    def create_prediction_features(
        self, upcoming_matches: list[dict], historical_matches: list[dict]
    ) -> pd.DataFrame:
        """Create features for upcoming matches to predict using vectorized history merges.

        Args:
            upcoming_matches: List of upcoming match dictionaries.
            historical_matches: List of all historical matches for context.

        Returns:
            DataFrame with features (no target variables).
        """
        if not upcoming_matches:
            return pd.DataFrame()

        upcoming = pd.DataFrame(upcoming_matches).copy()
        upcoming["date"] = pd.to_datetime(upcoming["date"])

        # Build long_df and history features from historical (finished) matches only
        long_hist = self._build_team_long_df(historical_matches)
        if long_hist.empty:
            return pd.DataFrame()
        long_hist["date"] = pd.to_datetime(long_hist["date"])
        long_hist = long_hist.sort_values(["team", "date", "match_id"]).reset_index(
            drop=True
        )

        long_hist = self._compute_team_history_features(long_hist)

        # Columns to use for asof merge
        hist_cols = [
            "avg_goals_for",
            "avg_goals_against",
            "avg_points",
            "form_points",
            "form_points_per_game",
            "form_wins",
            "form_draws",
            "form_losses",
            "form_goals_scored",
            "form_goals_conceded",
            "form_goal_diff",
            "form_avg_goals_scored",
            "form_avg_goals_conceded",
            "momentum_points",
            "momentum_goals",
            "momentum_conceded",
            "momentum_score",
            # venue-specific history metrics
            "points_pg_at_home",
            "goals_pg_at_home",
            "goals_conceded_pg_at_home",
            "points_pg_away",
            "goals_pg_away",
            "goals_conceded_pg_away",
        ]
        hist_merge = long_hist[["team", "date"] + hist_cols].copy()

        # Prepare per-side frames for asof merge (requires sorting by date)
        upcoming = upcoming.sort_values("date")

        # Home side merge
        home_hist = hist_merge.rename(columns={"team": "home_team"})
        # merge_asof requires the right keys (on=date) to be globally sorted by 'date'
        home_hist = home_hist.sort_values(["date"])
        features_df = pd.merge_asof(
            upcoming.sort_values("date"),
            home_hist,
            left_on="date",
            right_on="date",
            left_by="home_team",
            right_by="home_team",
            direction="backward",
        )
        # Prefix home columns
        for col in hist_cols:
            if col in features_df.columns:
                features_df.rename(columns={col: f"home_{col}"}, inplace=True)

        # Away side merge
        away_hist = hist_merge.rename(columns={"team": "away_team"})
        # Ensure global sort by 'date' for merge_asof
        away_hist = away_hist.sort_values(["date"])
        features_df = pd.merge_asof(
            features_df.sort_values("date"),
            away_hist,
            left_on="date",
            right_on="date",
            left_by="away_team",
            right_by="away_team",
            direction="backward",
            suffixes=("", "_away"),
        )
        # Prefix away columns
        for col in hist_cols:
            away_src = f"{col}_away"
            if away_src in features_df.columns:
                features_df.rename(columns={away_src: f"away_{col}"}, inplace=True)
        # If no name collision occurred during merge_asof, pandas will not add the
        # '_away' suffix and the right-hand columns keep their base names. Ensure
        # those get prefixed as away_* so they match the trained feature schema.
        for col in hist_cols:
            if f"away_{col}" not in features_df.columns and col in features_df.columns:
                features_df.rename(columns={col: f"away_{col}"}, inplace=True)

        # Attach EWMA recency features from historical via asof
        try:
            ewma_long_df = self._compute_ewma_recency_features(
                historical_matches, span=5
            )
        except Exception:
            ewma_long_df = None

        if ewma_long_df is not None and not ewma_long_df.empty:
            # Prepare EWMA frames
            ewm_home = ewma_long_df.rename(columns={"team": "home_team"})[
                [
                    "home_team",
                    "date",
                    "goals_for_ewm5",
                    "goals_against_ewm5",
                    "goal_diff_ewm5",
                    "points_ewm5",
                ]
            ]
            # Ensure global sort by 'date' for merge_asof
            ewm_home = ewm_home.sort_values(["date"])
            ewm_away = ewma_long_df.rename(columns={"team": "away_team"})[
                [
                    "away_team",
                    "date",
                    "goals_for_ewm5",
                    "goals_against_ewm5",
                    "goal_diff_ewm5",
                    "points_ewm5",
                ]
            ]
            # Ensure global sort by 'date' for merge_asof
            ewm_away = ewm_away.sort_values(["date"])

            features_df = pd.merge_asof(
                features_df.sort_values("date"),
                ewm_home,
                left_on="date",
                right_on="date",
                left_by="home_team",
                right_by="home_team",
                direction="backward",
            )
            features_df = pd.merge_asof(
                features_df.sort_values("date"),
                ewm_away,
                left_on="date",
                right_on="date",
                left_by="away_team",
                right_by="away_team",
                direction="backward",
                suffixes=("", "_away"),
            )
            # Rename to home_/away_ prefixed
            for base in [
                "goals_for_ewm5",
                "goals_against_ewm5",
                "goal_diff_ewm5",
                "points_ewm5",
            ]:
                if base in features_df.columns:
                    features_df.rename(columns={base: f"home_{base}"}, inplace=True)
                away_col = f"{base}_away"
                if away_col in features_df.columns:
                    features_df.rename(columns={away_col: f"away_{base}"}, inplace=True)

        # Elo ratings as-of the match date (from historical only)
        try:
            _, elo_long_df = self._compute_elos_from_matches(historical_matches)
        except Exception:
            elo_long_df = None
        if elo_long_df is not None and not elo_long_df.empty:
            try:
                # Prepare home/away elo frames
                home_elo = elo_long_df.rename(
                    columns={"team": "home_team", "elo": "home_elo"}
                )[["home_team", "date", "home_elo"]].sort_values(["date"])
                away_elo = elo_long_df.rename(
                    columns={"team": "away_team", "elo": "away_elo"}
                )[["away_team", "date", "away_elo"]].sort_values(["date"])
                features_df = pd.merge_asof(
                    features_df.sort_values("date"),
                    home_elo,
                    left_on="date",
                    right_on="date",
                    left_by="home_team",
                    right_by="home_team",
                    direction="backward",
                )
                features_df = pd.merge_asof(
                    features_df.sort_values("date"),
                    away_elo,
                    left_on="date",
                    right_on="date",
                    left_by="away_team",
                    right_by="away_team",
                    direction="backward",
                )
                features_df["elo_diff"] = features_df.get(
                    "home_elo", 0.0
                ) - features_df.get("away_elo", 0.0)
            except Exception:
                pass

        # Derived diffs
        features_df["abs_form_points_diff"] = (
            features_df.get("home_form_points", 0)
            - features_df.get("away_form_points", 0)
        ).abs()
        features_df["form_points_difference"] = features_df.get(
            "home_form_points", 0
        ) - features_df.get("away_form_points", 0)
        features_df["abs_momentum_score_diff"] = (
            features_df.get("home_momentum_score", 0.0)
            - features_df.get("away_momentum_score", 0.0)
        ).abs()
        features_df["momentum_score_difference"] = features_df.get(
            "home_momentum_score", 0.0
        ) - features_df.get("away_momentum_score", 0.0)

        # Venue deltas
        if all(
            c in features_df.columns
            for c in ["home_points_pg_at_home", "away_points_pg_away"]
        ):
            features_df["venue_points_delta"] = (
                features_df["home_points_pg_at_home"]
                - features_df["away_points_pg_away"]
            )
        if all(
            c in features_df.columns
            for c in ["home_goals_pg_at_home", "away_goals_pg_away"]
        ):
            features_df["venue_goals_delta"] = (
                features_df["home_goals_pg_at_home"] - features_df["away_goals_pg_away"]
            )
        if all(
            c in features_df.columns
            for c in ["home_goals_conceded_pg_at_home", "away_goals_conceded_pg_away"]
        ):
            features_df["venue_conceded_delta"] = (
                features_df["home_goals_conceded_pg_at_home"]
                - features_df["away_goals_conceded_pg_away"]
            )

        # Interaction ratios (safe divisions)
        eps = 1e-6

        def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a.astype(float) / (b.astype(float) + eps)

        if all(
            c in features_df.columns
            for c in ["home_form_avg_goals_scored", "away_form_avg_goals_conceded"]
        ):
            features_df["attack_defense_form_ratio_home"] = _safe_div(
                features_df["home_form_avg_goals_scored"],
                features_df["away_form_avg_goals_conceded"],
            )
        if all(
            c in features_df.columns
            for c in ["away_form_avg_goals_scored", "home_form_avg_goals_conceded"]
        ):
            features_df["attack_defense_form_ratio_away"] = _safe_div(
                features_df["away_form_avg_goals_scored"],
                features_df["home_form_avg_goals_conceded"],
            )
        if all(
            c in features_df.columns
            for c in ["home_avg_goals_for", "away_avg_goals_against"]
        ):
            features_df["attack_defense_long_ratio_home"] = _safe_div(
                features_df["home_avg_goals_for"], features_df["away_avg_goals_against"]
            )
        if all(
            c in features_df.columns
            for c in ["away_avg_goals_for", "home_avg_goals_against"]
        ):
            features_df["attack_defense_long_ratio_away"] = _safe_div(
                features_df["away_avg_goals_for"], features_df["home_avg_goals_against"]
            )
        if all(
            c in features_df.columns
            for c in ["home_form_points_per_game", "away_form_points_per_game"]
        ):
            features_df["form_points_pg_ratio"] = _safe_div(
                features_df["home_form_points_per_game"],
                features_df["away_form_points_per_game"],
            )
        if all(
            c in features_df.columns
            for c in ["home_momentum_score", "away_momentum_score"]
        ):
            features_df["momentum_score_ratio"] = _safe_div(
                features_df["home_momentum_score"], features_df["away_momentum_score"]
            )
        if all(
            c in features_df.columns for c in ["home_points_ewm5", "away_points_ewm5"]
        ):
            features_df["ewm_points_ratio"] = _safe_div(
                features_df["home_points_ewm5"], features_df["away_points_ewm5"]
            )

        return features_df

    def _apply_selected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """If a selected features list exists, filter DataFrame to essential + selected.

        This keeps ID/meta/target columns alongside the selected feature names.
        If the selection file is absent or unreadable, returns df unchanged.
        """
        try:
            sel_filename = getattr(
                self.config.model, "selected_features_file", "kept_features.yaml"
            )
            sel_path = self.config.paths.config_dir / sel_filename
            if not sel_path.exists():
                return df

            selected: list[str] | None = None
            if yaml is not None:
                with open(sel_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, list):
                    selected = [str(c) for c in loaded]
                elif isinstance(loaded, dict) and "features" in loaded:
                    val = loaded.get("features")
                    if isinstance(val, list):
                        selected = [str(c) for c in val]
                    elif isinstance(val, str):
                        # Support newline-separated string under key
                        selected = [s.strip() for s in val.splitlines() if s.strip()]
                elif isinstance(loaded, str):
                    # Support plain newline-separated string in YAML
                    selected = [s.strip() for s in loaded.splitlines() if s.strip()]
            else:
                # Fallback: try .txt with one name per line
                txt_path = sel_path.with_suffix(".txt")
                if txt_path.exists():
                    selected = [
                        line.strip()
                        for line in txt_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]

            if not selected:
                return df

            essential = {
                "match_id",
                "matchday",
                "date",
                "home_team",
                "away_team",
                "is_finished",
                "home_score",
                "away_score",
                "goal_difference",
                "result",
            }
            keep_cols = [c for c in df.columns if (c in essential) or (c in selected)]
            # Ensure we keep at least essentials
            if not keep_cols:
                return df
            return df[keep_cols].copy()
        except Exception:
            return df

    def _build_team_long_df(self, matches: list[dict]) -> pd.DataFrame:
        """Build a long-format team-match DataFrame from match dicts.

        Each finished match contributes two rows (home and away) with goals_for/against.
        """
        rows = []
        for m in matches:
            try:
                if not m.get("is_finished"):
                    continue
                date = m.get("date")
                if date is None:
                    continue
                hs = m.get("home_score")
                as_ = m.get("away_score")
                if hs is None or as_ is None:
                    continue
                # Home row
                rows.append(
                    {
                        "match_id": m.get("match_id"),
                        "date": date,
                        "team": m.get("home_team"),
                        "goals_for": hs,
                        "goals_against": as_,
                        "at_home": True,
                    }
                )
                # Away row
                rows.append(
                    {
                        "match_id": m.get("match_id"),
                        "date": date,
                        "team": m.get("away_team"),
                        "goals_for": as_,
                        "goals_against": hs,
                        "at_home": False,
                    }
                )
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(columns=["match_id", "date", "team"])

        long_df = pd.DataFrame(rows)
        long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
        # Points per match: 3 win, 1 draw, 0 loss (from perspective of team)
        long_df["points"] = np.select(
            [
                long_df["goals_for"] > long_df["goals_against"],
                long_df["goals_for"] == long_df["goals_against"],
            ],
            [3, 1],
            default=0,
        )
        return long_df

    def _compute_elos_from_matches(
        self, matches: list[dict], K: float | int = 20, home_adv: float | int = 90
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute Elo ratings chronologically and return per-match and per-team timeseries.

        Returns:
            match_elos: DataFrame with columns ['match_id','home_elo','away_elo'] (pre-match ratings)
            elo_long:  DataFrame with columns ['team','date','elo'] giving pre-match Elo per team per match
        """
        if not matches:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(matches).copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df = df.sort_values(["date", "match_id"])  # chronological

        ratings: dict[str, float] = defaultdict(lambda: 1500.0)
        rows_match: list[dict] = []
        rows_long: list[dict] = []

        for _, m in df.iterrows():
            mid = m.get("match_id")
            date = m.get("date")
            home = m.get("home_team")
            away = m.get("away_team")
            hs = m.get("home_score")
            as_ = m.get("away_score")
            r_h = float(ratings[home])
            r_a = float(ratings[away])

            rows_match.append({"match_id": mid, "home_elo": r_h, "away_elo": r_a})
            rows_long.append({"team": home, "date": date, "elo": r_h})
            rows_long.append({"team": away, "date": date, "elo": r_a})

            if pd.notnull(hs) and pd.notnull(as_):
                s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
                exp_h = 1.0 / (1.0 + 10 ** (-(r_h + float(home_adv) - r_a) / 400.0))
                delta_h = float(K) * (s_h - exp_h)
                ratings[home] = r_h + delta_h
                ratings[away] = r_a - delta_h

        match_elos = pd.DataFrame(rows_match).drop_duplicates(
            subset=["match_id"], keep="last"
        )
        elo_long = pd.DataFrame(rows_long)
        elo_long["date"] = (
            pd.to_datetime(elo_long["date"]) if len(elo_long) else elo_long
        )
        return match_elos, elo_long

    def _compute_team_history_features(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Compute leakage-safe per-team expanding/rolling and momentum features.

        Assumes long_df has columns: ['match_id','date','team','goals_for','goals_against','goal_diff','points']
        """
        if long_df.empty:
            return long_df

        long_df = long_df.sort_values(["team", "date", "match_id"]).reset_index(
            drop=True
        )
        grp = long_df.groupby("team", group_keys=False)

        # Shifted prior series to prevent leakage
        gf_prior = grp["goals_for"].shift(1)
        ga_prior = grp["goals_against"].shift(1)
        pts_prior = grp["points"].shift(1)

        long_df["avg_goals_for"] = (
            gf_prior.expanding().mean().reset_index(level=0, drop=True)
        )
        long_df["avg_goals_against"] = (
            ga_prior.expanding().mean().reset_index(level=0, drop=True)
        )
        long_df["avg_points"] = (
            pts_prior.expanding().mean().reset_index(level=0, drop=True)
        )

        # Rolling last-N form stats (on prior rows)
        try:
            N = int(getattr(self.config.model, "form_last_n", 5))
        except Exception:
            N = 5

        win_prior = (grp["goals_for"].shift(1) > grp["goals_against"].shift(1)).astype(
            float
        )
        draw_prior = (
            grp["goals_for"].shift(1) == grp["goals_against"].shift(1)
        ).astype(float)
        loss_prior = (grp["goals_for"].shift(1) < grp["goals_against"].shift(1)).astype(
            float
        )

        long_df["form_points"] = (
            pts_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        long_df["form_points_per_game"] = long_df["form_points"] / np.minimum(
            N, grp.cumcount() + 1
        )
        long_df["form_wins"] = (
            win_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        long_df["form_draws"] = (
            draw_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        long_df["form_losses"] = (
            loss_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        gf_roll = (
            gf_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        ga_roll = (
            ga_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        long_df["form_goals_scored"] = gf_roll
        long_df["form_goals_conceded"] = ga_roll
        long_df["form_goal_diff"] = gf_roll - ga_roll
        long_df["form_avg_goals_scored"] = gf_roll / np.minimum(N, grp.cumcount() + 1)
        long_df["form_avg_goals_conceded"] = ga_roll / np.minimum(N, grp.cumcount() + 1)

        # Momentum features via EWMA on prior rows
        try:
            decay = float(getattr(self.config.model, "momentum_decay", 0.9))
        except Exception:
            decay = 0.9
        alpha = max(1e-6, 1.0 - min(max(decay, 0.0), 0.9999))

        long_df["momentum_points"] = pts_prior.groupby(long_df["team"]).transform(
            lambda s: s.ewm(alpha=alpha, adjust=False).mean()
        )
        long_df["momentum_goals"] = gf_prior.groupby(long_df["team"]).transform(
            lambda s: s.ewm(alpha=alpha, adjust=False).mean()
        )
        long_df["momentum_conceded"] = ga_prior.groupby(long_df["team"]).transform(
            lambda s: s.ewm(alpha=alpha, adjust=False).mean()
        )
        long_df["momentum_score"] = (
            (long_df["momentum_points"].fillna(0))
            + (long_df["momentum_goals"].fillna(0)) * 0.5
            - (long_df["momentum_conceded"].fillna(0)) * 0.3
        )

        # Fill initial NaNs
        long_df.fillna(0.0, inplace=True)

        # Venue-specific last-N metrics (home vs away), leakage-safe via shift(1)
        try:
            home_mask = long_df["at_home"] == True
            away_mask = long_df["at_home"] == False

            # Group within venue subsets
            grp_home = long_df[home_mask].groupby("team", group_keys=False)
            grp_away = long_df[away_mask].groupby("team", group_keys=False)

            # Points per game (mean of prior points over last N at that venue)
            long_df.loc[home_mask, "points_pg_at_home"] = (
                grp_home["points"]
                .shift(1)
                .rolling(window=N, min_periods=1)
                .mean()
                .values
            )
            long_df.loc[away_mask, "points_pg_away"] = (
                grp_away["points"]
                .shift(1)
                .rolling(window=N, min_periods=1)
                .mean()
                .values
            )

            # Goals for per game at venue
            long_df.loc[home_mask, "goals_pg_at_home"] = (
                grp_home["goals_for"]
                .shift(1)
                .rolling(window=N, min_periods=1)
                .mean()
                .values
            )
            long_df.loc[away_mask, "goals_pg_away"] = (
                grp_away["goals_for"]
                .shift(1)
                .rolling(window=N, min_periods=1)
                .mean()
                .values
            )

            # Goals conceded per game at venue
            long_df.loc[home_mask, "goals_conceded_pg_at_home"] = (
                grp_home["goals_against"]
                .shift(1)
                .rolling(window=N, min_periods=1)
                .mean()
                .values
            )
            long_df.loc[away_mask, "goals_conceded_pg_away"] = (
                grp_away["goals_against"]
                .shift(1)
                .rolling(window=N, min_periods=1)
                .mean()
                .values
            )

            # Fill remaining NaNs with zeros
            for c in [
                "points_pg_at_home",
                "goals_pg_at_home",
                "goals_conceded_pg_at_home",
                "points_pg_away",
                "goals_pg_away",
                "goals_conceded_pg_away",
            ]:
                if c in long_df.columns:
                    long_df[c] = long_df[c].fillna(0.0)
        except Exception:
            # If venue columns are missing for any reason, proceed without them
            pass

        return long_df

    def _compute_ewma_recency_features(
        self, matches: list[dict], span: int = 5
    ) -> pd.DataFrame:
        """Compute leakage-safe EWMA recency features on a long-format team-match frame.

        Returns a DataFrame with columns: ['match_id','date','team', '<metric>_ewm{span}', ...]
        where each EWMA is computed on prior values only (via groupby.shift(1)).
        """
        long_df = self._build_team_long_df(matches)
        if long_df.empty:
            return long_df

        long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

        metrics = ["goals_for", "goals_against", "goal_diff", "points"]
        for col in metrics:
            prior_col = f"{col}_prior"
            ewm_col = f"{col}_ewm{span}"
            long_df[prior_col] = long_df.groupby("team")[col].shift(1)
            # EWMA on prior values to avoid leakage
            long_df[ewm_col] = long_df.groupby("team")[prior_col].transform(
                lambda s: s.ewm(span=span, adjust=False).mean()
            )
            # Drop helper
            long_df.drop(columns=[prior_col], inplace=True)

        # Fill early NaNs with global means (season not tracked explicitly)
        for col in [f"{m}_ewm{span}" for m in metrics]:
            if col in long_df.columns:
                long_df[col] = long_df[col].fillna(long_df[col].mean())

        # Keep only lookup-relevant columns
        keep_cols = ["match_id", "date", "team"] + [f"{m}_ewm{span}" for m in metrics]
        return long_df[keep_cols]

    def _get_form_features(
        self, team: str, history: list[dict], prefix: str, last_n: int = 5
    ) -> dict:
        """Calculate team form over last N matches."""
        # Allow config override for last_n
        try:
            cfg_last_n = int(getattr(self.config.model, "form_last_n", last_n))
            last_n = cfg_last_n if cfg_last_n > 0 else last_n
        except Exception:
            pass

        recent = history[-last_n:] if len(history) >= last_n else history

        points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0

        for match in recent:
            if match["home_team"] == team:
                scored = match["home_score"]
                conceded = match["away_score"]
            else:
                scored = match["away_score"]
                conceded = match["home_score"]

            goals_scored += scored
            goals_conceded += conceded

            if scored > conceded:
                points += 3
                wins += 1
            elif scored == conceded:
                points += 1
                draws += 1
            else:
                losses += 1

        num_matches = len(recent)

        return {
            f"{prefix}_form_points": points,
            f"{prefix}_form_points_per_game": points / num_matches
            if num_matches > 0
            else 0,
            f"{prefix}_form_wins": wins,
            f"{prefix}_form_draws": draws,
            f"{prefix}_form_losses": losses,
            f"{prefix}_form_goals_scored": goals_scored,
            f"{prefix}_form_goals_conceded": goals_conceded,
            f"{prefix}_form_goal_diff": goals_scored - goals_conceded,
            f"{prefix}_form_avg_goals_scored": goals_scored / num_matches
            if num_matches > 0
            else 0,
            f"{prefix}_form_avg_goals_conceded": goals_conceded / num_matches
            if num_matches > 0
            else 0,
        }

    def _calculate_table(self, matches: list[dict]) -> dict:
        """Calculate league table from matches."""
        table = defaultdict(
            lambda: {
                "played": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "goals_for": 0,
                "goals_against": 0,
                "points": 0,
            }
        )

        for match in matches:
            if not match["is_finished"]:
                continue

            home = match["home_team"]
            away = match["away_team"]
            home_score = match["home_score"]
            away_score = match["away_score"]

            table[home]["played"] += 1
            table[away]["played"] += 1
            table[home]["goals_for"] += home_score
            table[home]["goals_against"] += away_score
            table[away]["goals_for"] += away_score
            table[away]["goals_against"] += home_score

            if home_score > away_score:
                table[home]["won"] += 1
                table[home]["points"] += 3
                table[away]["lost"] += 1
            elif home_score < away_score:
                table[away]["won"] += 1
                table[away]["points"] += 3
                table[home]["lost"] += 1
            else:
                table[home]["drawn"] += 1
                table[away]["drawn"] += 1
                table[home]["points"] += 1
                table[away]["points"] += 1

        # Add positions
        sorted_teams = sorted(
            table.items(),
            key=lambda x: (
                x[1]["points"],
                x[1]["goals_for"] - x[1]["goals_against"],
                x[1]["goals_for"],
            ),
            reverse=True,
        )

        for position, (team, data) in enumerate(sorted_teams, 1):
            table[team]["position"] = position

        return dict(table)
