"""Data loading and feature engineering for the Kicktipp predictor.

This module combines data fetching from the OpenLigaDB API with comprehensive
feature engineering for match prediction.
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
except Exception:
    yaml = None


class DataLoader:
    """Fetches match data and creates features for prediction.

    Attributes:
        config: The configuration for the data loader.
        cache_dir: The directory where cached data is stored.
        base_url: The base URL of the API.
        league_code: The league code to fetch data for.
        cache_ttl: The time-to-live for the cache in seconds.
        timeout: The timeout for HTTP requests in seconds.
    """

    def __init__(self):
        """Initializes the DataLoader."""
        self.config = get_config()
        self.cache_dir = str(self.config.paths.cache_dir)
        self.base_url = self.config.api.base_url
        self.league_code = self.config.api.league_code
        self.cache_ttl = self.config.api.cache_ttl
        self.timeout = self.config.api.request_timeout

    def get_current_season(self) -> int:
        """Get the current season.

        Returns:
            The current season.
        """
        now = datetime.now()
        return now.year if now.month >= 7 else now.year - 1

    def fetch_season_matches(self, season: int | None = None) -> list[dict]:
        """Fetch all matches for a given season.

        Args:
            season: The season to fetch matches for.

        Returns:
            A list of matches.
        """
        if season is None:
            season = self.get_current_season()

        cache_file = os.path.join(self.cache_dir, f"matches_{season}.pkl")

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

            with open(cache_file, "wb") as f:
                pickle.dump(matches, f)

            return matches

        except Exception as e:
            print(f"Error fetching matches for season {season}: {e}")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            return []

    def fetch_matchday(self, matchday: int, season: int | None = None) -> list[dict]:
        """Fetch all matches for a given matchday.

        Args:
            matchday: The matchday to fetch matches for.
            season: The season to fetch matches for.

        Returns:
            A list of matches.
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
        """Get the current matchday.

        Args:
            season: The season to get the current matchday for.

        Returns:
            The current matchday.
        """
        matches = self.fetch_season_matches(season)

        now = datetime.now()

        for match in matches:
            if match["date"] > now and not match["is_finished"]:
                return match["matchday"]

        if matches:
            return matches[-1]["matchday"]

        return 1

    def get_upcoming_matches(
        self, days: int = 7, season: int | None = None
    ) -> list[dict]:
        """Get upcoming matches.

        Args:
            days: The number of days to look ahead for upcoming matches.
            season: The season to get upcoming matches for.

        Returns:
            A list of upcoming matches.
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
        """Fetch historical seasons.

        Args:
            start_season: The start season to fetch.
            end_season: The end season to fetch.

        Returns:
            A list of matches from the historical seasons.
        """
        all_matches = []

        for season in range(start_season, end_season + 1):
            if os.getenv("KTP_VERBOSE") == "1":
                print(f"Fetching season {season}/{season + 1}...")
            matches = self.fetch_season_matches(season)
            all_matches.extend(matches)
            time.sleep(0.5)

        return all_matches

    def _parse_matches(self, matches_raw: list[dict]) -> list[dict]:
        """Parse a list of raw matches.

        Args:
            matches_raw: A list of raw matches.

        Returns:
            A list of parsed matches.
        """
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

                if match["matchIsFinished"] and match["matchResults"]:
                    final_result = [
                        r for r in match["matchResults"] if r["resultTypeID"] == 2
                    ]
                    if final_result:
                        result = final_result[0]
                        match_dict["home_score"] = result["pointsTeam1"]
                        match_dict["away_score"] = result["pointsTeam2"]
                    else:
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

    def create_features_from_matches(self, matches: list[dict]) -> pd.DataFrame:
        """Create a feature dataset from a list of matches.

        Args:
            matches: A list of matches.

        Returns:
            A DataFrame with features and target variables.
        """
        if not matches:
            return pd.DataFrame()

        base = pd.DataFrame(matches)
        base = base.loc[
            base["is_finished"]
            & base["home_score"].notna()
            & base["away_score"].notna()
        ].copy()
        if base.empty:
            return pd.DataFrame()
        base["date"] = pd.to_datetime(base["date"])

        long_df = self._build_team_long_df(matches)
        if long_df.empty:
            return pd.DataFrame()
        long_df["date"] = pd.to_datetime(long_df["date"])
        long_df = long_df.sort_values(["team", "date", "match_id"]).reset_index(
            drop=True
        )

        long_df = self._compute_team_history_features(long_df)

        try:
            ewma_long_df = self._compute_ewma_recency_features(matches, span=5)
        except Exception:
            ewma_long_df = None

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
        features_df = features_df.merge(
            home_merge.add_prefix("home_"),
            left_on=["match_id", "home_team"],
            right_on=["home_match_id", "home_home_team"],
            how="left",
        )
        features_df = features_df.merge(
            away_merge.add_prefix("away_"),
            left_on=["match_id", "away_team"],
            right_on=["away_match_id", "away_away_team"],
            how="left",
        )

        drop_cols = [
            "home_match_id",
            "home_home_team",
            "away_match_id",
            "away_away_team",
        ]
        for c in drop_cols:
            if c in features_df.columns:
                features_df.drop(columns=[c], inplace=True)

        if ewma_long_df is not None and not ewma_long_df.empty:
            home_ewm = ewma_long_df.rename(
                columns={"team": "home_team", "match_id": "home_match_id"}
            )
            features_df = features_df.merge(
                home_ewm.add_prefix("home_"),
                left_on=["match_id", "home_team"],
                right_on=["home_home_match_id", "home_home_team"],
                how="left",
            )
            away_ewm = ewma_long_df.rename(
                columns={"team": "away_team", "match_id": "away_match_id"}
            )
            features_df = features_df.merge(
                away_ewm.add_prefix("away_"),
                left_on=["match_id", "away_team"],
                right_on=["away_away_match_id", "away_away_team"],
                how="left",
            )
            for c in [
                "home_home_match_id",
                "home_home_team",
                "away_away_match_id",
                "away_away_team",
            ]:
                if c in features_df.columns:
                    features_df.drop(columns=[c], inplace=True)

            ewm_map = {
                "goals_for_ewm5": "goals_for_ewm5",
                "goals_against_ewm5": "goals_against_ewm5",
                "goal_diff_ewm5": "goal_diff_ewm5",
                "points_ewm5": "points_ewm5",
            }
            for base_col in ewm_map.values():
                hcol = f"home_{base_col}"
                acol = f"away_{base_col}"
                if hcol not in features_df.columns:
                    features_df[hcol] = 0.0
                if acol not in features_df.columns:
                    features_df[acol] = 0.0

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
                features_df["home_goals_pg_at_home"]
                - features_df["away_goals_pg_away"]
            )
        if all(
            c in features_df.columns
            for c in [
                "home_goals_conceded_pg_at_home",
                "away_goals_conceded_pg_away",
            ]
        ):
            features_df["venue_conceded_delta"] = (
                features_df["home_goals_conceded_pg_at_home"]
                - features_df["away_goals_conceded_pg_away"]
            )

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
                features_df["home_avg_goals_for"],
                features_df["away_avg_goals_against"],
            )
        if all(
            c in features_df.columns
            for c in ["away_avg_goals_for", "home_avg_goals_against"]
        ):
            features_df["attack_defense_long_ratio_away"] = _safe_div(
                features_df["away_avg_goals_for"],
                features_df["home_avg_goals_against"],
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
                features_df["home_momentum_score"],
                features_df["away_momentum_score"],
            )
        if all(
            c in features_df.columns
            for c in ["home_points_ewm5", "away_points_ewm5"]
        ):
            features_df["ewm_points_ratio"] = _safe_div(
                features_df["home_points_ewm5"], features_df["away_points_ewm5"]
            )

        features_df["goal_difference"] = (
            features_df["home_score"] - features_df["away_score"]
        )
        features_df["result"] = np.where(
            features_df["goal_difference"] > 0,
            "H",
            np.where(features_df["goal_difference"] < 0, "A", "D"),
        )

        features_df = self._apply_selected_features(features_df)

        return features_df

    def create_prediction_features(
        self, upcoming_matches: list[dict], historical_matches: list[dict]
    ) -> pd.DataFrame:
        """Create a feature dataset for prediction.

        Args:
            upcoming_matches: A list of upcoming matches.
            historical_matches: A list of historical matches.

        Returns:
            A DataFrame with features.
        """
        if not upcoming_matches:
            return pd.DataFrame()

        upcoming = pd.DataFrame(upcoming_matches).copy()
        upcoming["date"] = pd.to_datetime(upcoming["date"])

        long_hist = self._build_team_long_df(historical_matches)
        if long_hist.empty:
            return pd.DataFrame()
        long_hist["date"] = pd.to_datetime(long_hist["date"])
        long_hist = long_hist.sort_values(
            ["team", "date", "match_id"]
        ).reset_index(drop=True)

        long_hist = self._compute_team_history_features(long_hist)

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
            "points_pg_at_home",
            "goals_pg_at_home",
            "goals_conceded_pg_at_home",
            "points_pg_away",
            "goals_pg_away",
            "goals_conceded_pg_away",
        ]
        hist_merge = long_hist[["team", "date"] + hist_cols].copy()

        upcoming = upcoming.sort_values("date")

        home_hist = hist_merge.rename(columns={"team": "home_team"})
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
        for col in hist_cols:
            if col in features_df.columns:
                features_df.rename(
                    columns={col: f"home_{col}"}, inplace=True
                )

        away_hist = hist_merge.rename(columns={"team": "away_team"})
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
        for col in hist_cols:
            away_src = f"{col}_away"
            if away_src in features_df.columns:
                features_df.rename(
                    columns={away_src: f"away_{col}"}, inplace=True
                )
        for col in hist_cols:
            if f"away_{col}" not in features_df.columns and col in features_df.columns:
                features_df.rename(
                    columns={col: f"away_{col}"}, inplace=True
                )

        try:
            ewma_long_df = self._compute_ewma_recency_features(
                historical_matches, span=5
            )
        except Exception:
            ewma_long_df = None

        if ewma_long_df is not None and not ewma_long_df.empty:
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
            for base in [
                "goals_for_ewm5",
                "goals_against_ewm5",
                "goal_diff_ewm5",
                "points_ewm5",
            ]:
                if base in features_df.columns:
                    features_df.rename(
                        columns={base: f"home_{base}"}, inplace=True
                    )
                away_col = f"{base}_away"
                if away_col in features_df.columns:
                    features_df.rename(
                        columns={away_col: f"away_{base}"}, inplace=True
                    )

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
                features_df["home_goals_pg_at_home"]
                - features_df["away_goals_pg_away"]
            )
        if all(
            c in features_df.columns
            for c in [
                "home_goals_conceded_pg_at_home",
                "away_goals_conceded_pg_away",
            ]
        ):
            features_df["venue_conceded_delta"] = (
                features_df["home_goals_conceded_pg_at_home"]
                - features_df["away_goals_conceded_pg_away"]
            )

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
                features_df["home_avg_goals_for"],
                features_df["away_avg_goals_against"],
            )
        if all(
            c in features_df.columns
            for c in ["away_avg_goals_for", "home_avg_goals_against"]
        ):
            features_df["attack_defense_long_ratio_away"] = _safe_div(
                features_df["away_avg_goals_for"],
                features_df["home_avg_goals_against"],
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
                features_df["home_momentum_score"],
                features_df["away_momentum_score"],
            )
        if all(
            c in features_df.columns
            for c in ["home_points_ewm5", "away_points_ewm5"]
        ):
            features_df["ewm_points_ratio"] = _safe_div(
                features_df["home_points_ewm5"], features_df["away_points_ewm5"]
            )

        return features_df

    def _apply_selected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a DataFrame to the selected features.

        Args:
            df: The DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        try:
            sel_path = self.config.paths.config_dir / "kept_features.yaml"
            if not sel_path.exists():
                return df

            selected: list[str] | None = None
            if yaml is not None:
                with open(sel_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, list):
                    selected = [str(c) for c in loaded]
            else:
                txt_path = sel_path.with_suffix(".txt")
                if txt_path.exists():
                    selected = [
                        line.strip()
                        for line in txt_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
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
            keep_cols = [
                c for c in df.columns if (c in essential) or (c in selected)
            ]
            if not keep_cols:
                return df
            return df[keep_cols].copy()
        except Exception:
            return df

    def _build_team_long_df(self, matches: list[dict]) -> pd.DataFrame:
        """Build a long-format DataFrame of teams.

        Args:
            matches: A list of matches.

        Returns:
            A long-format DataFrame of teams.
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
        long_df["points"] = np.select(
            [
                long_df["goals_for"] > long_df["goals_against"],
                long_df["goals_for"] == long_df["goals_against"],
            ],
            [3, 1],
            default=0,
        )
        return long_df

    def _compute_team_history_features(
        self, long_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute historical features for a team.

        Args:
            long_df: A long-format DataFrame of teams.

        Returns:
            A DataFrame with historical features.
        """
        if long_df.empty:
            return long_df

        long_df = long_df.sort_values(
            ["team", "date", "match_id"]
        ).reset_index(drop=True)
        grp = long_df.groupby("team", group_keys=False)

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

        try:
            N = int(getattr(self.config.model, "form_last_n", 5))
        except Exception:
            N = 5

        win_prior = (
            grp["goals_for"].shift(1) > grp["goals_against"].shift(1)
        ).astype(float)
        draw_prior = (
            grp["goals_for"].shift(1) == grp["goals_against"].shift(1)
        ).astype(float)
        loss_prior = (
            grp["goals_for"].shift(1) < grp["goals_against"].shift(1)
        ).astype(float)

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
        long_df["form_avg_goals_scored"] = gf_roll / np.minimum(
            N, grp.cumcount() + 1
        )
        long_df["form_avg_goals_conceded"] = ga_roll / np.minimum(
            N, grp.cumcount() + 1
        )

        try:
            decay = float(getattr(self.config.model, "momentum_decay", 0.9))
        except Exception:
            decay = 0.9
        alpha = max(1e-6, 1.0 - min(max(decay, 0.0), 0.9999))

        long_df["momentum_points"] = pts_prior.groupby(
            long_df["team"]
        ).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        long_df["momentum_goals"] = gf_prior.groupby(
            long_df["team"]
        ).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        long_df["momentum_conceded"] = ga_prior.groupby(
            long_df["team"]
        ).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        long_df["momentum_score"] = (
            (long_df["momentum_points"].fillna(0))
            + (long_df["momentum_goals"].fillna(0)) * 0.5
            - (long_df["momentum_conceded"].fillna(0)) * 0.3
        )

        long_df.fillna(0.0, inplace=True)

        try:
            home_mask = long_df["at_home"] == True
            away_mask = long_df["at_home"] == False

            grp_home = long_df[home_mask].groupby("team", group_keys=False)
            grp_away = long_df[away_mask].groupby("team", group_keys=False)

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
            pass

        return long_df

    def _compute_ewma_recency_features(
        self, matches: list[dict], span: int = 5
    ) -> pd.DataFrame:
        """Compute EWMA recency features.

        Args:
            matches: A list of matches.
            span: The span for the EWMA.

        Returns:
            A DataFrame with EWMA recency features.
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
            long_df[ewm_col] = long_df.groupby("team")[prior_col].transform(
                lambda s: s.ewm(span=span, adjust=False).mean()
            )
            long_df.drop(columns=[prior_col], inplace=True)

        for col in [f"{m}_ewm{span}" for m in metrics]:
            if col in long_df.columns:
                long_df[col] = long_df[col].fillna(long_df[col].mean())

        keep_cols = ["match_id", "date", "team"] + [
            f"{m}_ewm{span}" for m in metrics
        ]
        return long_df[keep_cols]

    def _get_form_features(
        self, team: str, history: list[dict], prefix: str, last_n: int = 5
    ) -> dict:
        """Get form features for a team.

        Args:
            team: The team to get form features for.
            history: The historical matches of the team.
            prefix: The prefix for the feature names.
            last_n: The number of last matches to use for the form features.

        Returns:
            A dictionary of form features.
        """
        try:
            cfg_last_n = int(
                getattr(self.config.model, "form_last_n", last_n)
            )
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
            f"{prefix}_form_points_per_game": (
                points / num_matches if num_matches > 0 else 0
            ),
            f"{prefix}_form_wins": wins,
            f"{prefix}_form_draws": draws,
            f"{prefix}_form_losses": losses,
            f"{prefix}_form_goals_scored": goals_scored,
            f"{prefix}_form_goals_conceded": goals_conceded,
            f"{prefix}_form_goal_diff": goals_scored - goals_conceded,
            f"{prefix}_form_avg_goals_scored": (
                goals_scored / num_matches if num_matches > 0 else 0
            ),
            f"{prefix}_form_avg_goals_conceded": (
                goals_conceded / num_matches if num_matches > 0 else 0
            ),
        }

    def _calculate_table(self, matches: list[dict]) -> dict:
        """Calculate the league table from a list of matches.

        Args:
            matches: A list of matches.

        Returns:
            A dictionary representing the league table.
        """
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
