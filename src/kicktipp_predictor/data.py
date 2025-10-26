"""Data loading and feature engineering for kicktipp predictor.

This module combines data fetching from OpenLigaDB API with
comprehensive feature engineering for match prediction.
"""

# === Imports ===
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
    """Fetches match data and creates features for prediction.

    Exposes a public API used by CLI and predictor modules while hiding
    implementation details behind private helper methods.
    """

    def __init__(self):
        """Initialize with configuration."""
        self.config = get_config()
        self.cache_dir = str(self.config.paths.cache_dir)
        self.base_url = self.config.api.base_url
        self.league_code = self.config.api.league_code
        self.cache_ttl = self.config.api.cache_ttl
        self.timeout = self.config.api.request_timeout

    # =================================================================
    # Internal helpers (caching & requests)
    # =================================================================
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Return True if cache file exists and is within TTL."""
        try:
            age = time.time() - os.path.getmtime(cache_file)
            return age < self.cache_ttl
        except Exception:
            return False

    def _load_cached_matches(self, cache_file: str) -> list[dict] | None:
        """Load cached matches list from pickle file, or None on failure."""
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cached_matches(self, cache_file: str, matches: list[dict]) -> None:
        """Persist matches list to pickle cache, ignoring write errors."""
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(matches, f)
        except Exception:
            pass

    def _request_matches(self, url: str) -> list[dict]:
        """Perform HTTP GET and return parsed JSON list of matches.

        Errors propagate to caller for context-appropriate handling.
        """
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

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
            season: Season year (e.g., 2024 for 2024/2025).
                Defaults to current season.

        Returns:
            List of match dictionaries.
        """
        if season is None:
            season = self.get_current_season()

        cache_file = os.path.join(self.cache_dir, f"matches_{season}.pkl")

        # Use valid cache when possible
        if os.path.exists(cache_file) and self._is_cache_valid(cache_file):
            cached = self._load_cached_matches(cache_file)
            if cached is not None:
                return cached

        url = f"{self.base_url}/getmatchdata/{self.league_code}/{season}"
        try:
            matches_raw = self._request_matches(url)
            matches = self._parse_matches(matches_raw)
            self._save_cached_matches(cache_file, matches)
            return matches
        except Exception as e:
            print(f"Error fetching matches for season {season}: {e}")
            # Fallback to any cached data even if expired
            cached = self._load_cached_matches(cache_file)
            return cached if cached is not None else []

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

        url = (
            f"{self.base_url}/getmatchdata/{self.league_code}/{season}/{matchday}"
        )
        try:
            matches_raw = self._request_matches(url)
            return self._parse_matches(matches_raw)
        except Exception as e:
            print(f"Error fetching matchday {matchday}: {e}")
            return []

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

        return all_matches

    def _parse_matches(self, matches_raw: list[dict]) -> list[dict]:
        """Parse raw API response into standardized match dictionaries.

        Converts OpenLigaDB response objects into a stable schema used by
        downstream feature engineering.
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
            except Exception:
                # Skip malformed entries but continue parsing others
                continue

        return matches

    # =================================================================
    # FEATURE ENGINEERING METHODS
    # =================================================================

    def _compute_tanh_tamed_elo(
        self,
        df: pd.DataFrame,
        elo_diff_col: str = "normalized_elo_diff",
        output_col: str = "tanh_tamed_elo",
    ) -> pd.DataFrame:
        """Compute a bounded, tamed Elo feature via tanh transform.

        Parameters:
        - df: DataFrame that may contain Elo difference columns.
        - elo_diff_col: Name of the normalized Elo difference column to use.
          If missing but raw `elo_diff` exists, this method normalizes it using
          the average prior matches scale factor.
        - output_col: Name of the output feature column to write.

        Returns:
        - df: The same DataFrame with `output_col` added as float in [-1.0, 1.0].

        Notes:
        - Robust to missing inputs: If Elo information is unavailable, sets the
          output feature to 0.0 to avoid downstream KeyErrors.
        - Guards against extreme values via clipping before applying tanh.
        """
        try:
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

            col = elo_diff_col
            if col not in df.columns:
                # Attempt to derive normalized diff from raw Elo difference
                if "elo_diff" in df.columns:
                    home_mp = df.get("home_matches_played", pd.Series(0)).fillna(0)
                    away_mp = df.get("away_matches_played", pd.Series(0)).fillna(0)
                    avg_matches = (home_mp + away_mp) / 2.0
                    scale_factor = 1.0 + avg_matches * 0.01
                    df["normalized_elo_diff"] = (
                        pd.to_numeric(df["elo_diff"], errors="coerce").fillna(0.0)
                        / scale_factor
                    )
                    col = "normalized_elo_diff"
                else:
                    df[output_col] = 0.0
                    return df

            # Numeric conversion and clipping for robustness
            diff = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            max_abs = 800.0  # Cap extreme Elo differentials
            diff = diff.clip(-max_abs, max_abs)

            # Scale keeps typical Elo ranges in tanh's sensitive region
            scale = 250.0
            df[output_col] = np.tanh(diff / scale).astype(float)
            return df
        except Exception:
            # Ensure feature exists even if something unexpected occurs
            df[output_col] = df.get(output_col, 0.0)
            return df

    def _merge_history_features(
        self,
        base: pd.DataFrame,
        long_df: pd.DataFrame,
        hist_cols: list[str],
    ) -> pd.DataFrame:
        """Merge home/away history features into match-level frame.

        Args:
            base: Match-level DataFrame of finished games.
            long_df: Long-format per-team history features.
            hist_cols: Columns from long_df to merge for each side.

        Returns:
            Match-level DataFrame with prefixed home/away history columns.
        """
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
        features_df.drop(columns=["home_match_id"], inplace=True)

        # Merge away team features
        features_df = features_df.merge(
            away_merge.add_prefix("away_"),
            left_on=["match_id", "away_team"],
            right_on=["away_match_id", "away_away_team"],
            how="left",
        )
        features_df.drop(columns=["away_match_id"], inplace=True)

        # Normalize column names for match counts
        if "home_matches_played_prior" in features_df.columns:
            features_df.rename(
                columns={"home_matches_played_prior": "home_matches_played"},
                inplace=True,
            )
        if "away_matches_played_prior" in features_df.columns:
            features_df.rename(
                columns={"away_matches_played_prior": "away_matches_played"},
                inplace=True,
            )

        return features_df

    def create_features_from_matches(self, matches: list[dict]) -> pd.DataFrame:
        """Create feature dataset from match data using vectorized pandas ops.

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
        grp = long_df.groupby("team", group_keys=False)

        # Prior matches played per team (leakage-safe count)
        long_df["matches_played_prior"] = grp.cumcount()

        # Cache current standings for opponent rank weighting
        self._current_table = self._calculate_table(matches)
        long_df = self._compute_team_history_features(long_df)

        # Assemble match-level features by merging home/away history rows
        hist_cols = [
            "avg_goals_for",
            "avg_goals_against",
            "avg_points",
            "form_points_weighted_by_opponent_rank",
            "form_points_per_game",
            # Multi-window points (unweighted and weighted)
            "form_points_L3",
            "form_points_L5",
            "form_points_L10",
            "wform_points_L3",
            "wform_points_L5",
            "wform_points_L10",
            "form_wins",
            "form_draws",
            "form_losses",
            "form_goals_scored",
            "form_goals_conceded",
            "form_goal_diff",
            # Multi-window goal diff
            "form_goal_diff_L3",
            "form_goal_diff_L5",
            "form_goal_diff_L10",
            "form_avg_goals_scored",
            "form_avg_goals_conceded",
            # matches played prior (leakage-safe count)
            "matches_played_prior",
        ]
        features_df = self._merge_history_features(base, long_df, hist_cols)

        # Elo ratings joined per match for normalization
        try:
            match_elos, _ = self._compute_elos_from_matches(matches)
            features_df = features_df.merge(
                match_elos, left_on="match_id", right_on="match_id", how="left"
            )
            # Derived Elo difference (home - away)
            features_df["elo_diff"] = (
                features_df.get("home_elo", 0.0) - features_df.get("away_elo", 0.0)
            )
            # Normalize elo_diff by average matches played prior
            home_mp = features_df.get("home_matches_played", pd.Series(0)).fillna(0)
            away_mp = features_df.get("away_matches_played", pd.Series(0)).fillna(0)
            avg_matches = (home_mp + away_mp) / 2.0
            scale_factor = 1.0 + avg_matches * 0.01
            features_df["normalized_elo_diff"] = (
                features_df["elo_diff"].fillna(0).astype(float) / scale_factor
            )
            # Compute tamed Elo feature before dropping transient columns
            features_df = self._compute_tanh_tamed_elo(features_df)
            # Transient Elo features: created briefly and then removed to prevent leakage/bias
            features_df.drop(
                columns=["home_elo", "away_elo", "elo_diff"],
                errors="ignore",
                inplace=True,
            )
        except Exception:
            # Elo is optional; continue gracefully
            # Ensure feature exists even if Elo computation failed
            features_df = self._compute_tanh_tamed_elo(features_df)
            pass

        # Derived form diffs
        features_df["abs_weighted_form_points_diff"] = (
            features_df.get("home_form_points_weighted_by_opponent_rank", 0)
            - features_df.get("away_form_points_weighted_by_opponent_rank", 0)
        ).abs()
        features_df["weighted_form_points_difference"] = features_df.get(
            "home_form_points_weighted_by_opponent_rank", 0
        ) - features_df.get("away_form_points_weighted_by_opponent_rank", 0)

        # Targets
        features_df["goal_difference"] = (
            features_df["home_score"] - features_df["away_score"]
        )
        features_df["result"] = np.where(
            features_df["goal_difference"] > 0,
            "H",
            np.where(features_df["goal_difference"] < 0, "A", "D"),
        )

        # Optionally reduce to selected features if selection exists
        features_df = self._apply_selected_features(features_df)

        return features_df

    def create_prediction_features(
        self, upcoming_matches: list[dict], historical_matches: list[dict]
    ) -> pd.DataFrame:
        """Create features for upcoming matches using vectorized history merges.

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

        # Cache current standings for opponent rank weighting
        self._current_table = self._calculate_table(historical_matches)
        long_hist = self._compute_team_history_features(long_hist)

        # Columns to use for asof merge
        hist_cols = [
            "avg_goals_for",
            "avg_goals_against",
            "avg_points",
            "form_points_weighted_by_opponent_rank",
            "form_points_per_game",
            # Multi-window points (unweighted and weighted)
            "form_points_L3",
            "form_points_L5",
            "form_points_L10",
            "wform_points_L3",
            "wform_points_L5",
            "wform_points_L10",
            "form_wins",
            "form_draws",
            "form_losses",
            "form_goals_scored",
            "form_goals_conceded",
            "form_goal_diff",
            # Multi-window goal diff
            "form_goal_diff_L3",
            "form_goal_diff_L5",
            "form_goal_diff_L10",
            "form_avg_goals_scored",
            "form_avg_goals_conceded",
            # matches played prior (leakage-safe count)
            "matches_played_prior",
        ]
        # Compute per-team prior match count for upcoming features
        long_hist["matches_played_prior"] = long_hist.groupby("team").cumcount()
        hist_merge = long_hist[["team", "date"] + hist_cols].copy()

        # Prepare per-side frames for asof merge (requires sorting by date)
        upcoming = upcoming.sort_values("date")

        # Home side merge
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
        # Prefix home columns
        for col in hist_cols:
            if col in features_df.columns:
                features_df.rename(columns={col: f"home_{col}"}, inplace=True)

        # Away side merge
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
        )
        # Prefix away columns
        for col in hist_cols:
            colname = col
            if colname in features_df.columns:
                features_df.rename(columns={colname: f"away_{colname}"}, inplace=True)

        # Normalize column names for match counts
        if "home_matches_played_prior" in features_df.columns:
            features_df.rename(
                columns={"home_matches_played_prior": "home_matches_played"},
                inplace=True,
            )
        if "away_matches_played_prior" in features_df.columns:
            features_df.rename(
                columns={"away_matches_played_prior": "away_matches_played"},
                inplace=True,
            )

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
                features_df["elo_diff"] = (
                    features_df.get("home_elo", 0.0)
                    - features_df.get("away_elo", 0.0)
                )
                # Normalize and compute tamed Elo for upcoming matches
                home_mp = features_df.get("home_matches_played", pd.Series(0)).fillna(0)
                away_mp = features_df.get("away_matches_played", pd.Series(0)).fillna(0)
                avg_matches = (home_mp + away_mp) / 2.0
                scale_factor = 1.0 + avg_matches * 0.01
                features_df["normalized_elo_diff"] = (
                    pd.to_numeric(features_df.get("elo_diff", 0.0), errors="coerce").fillna(0.0)
                    / scale_factor
                )
                features_df = self._compute_tanh_tamed_elo(features_df)
            except Exception:
                # If Elo merge fails, still ensure feature exists
                features_df = self._compute_tanh_tamed_elo(features_df)
                pass
        else:
            # No Elo available; still create a safe default feature
            features_df = self._compute_tanh_tamed_elo(features_df)

        # Derived diffs
        features_df["abs_weighted_form_points_diff"] = (
            features_df.get("home_form_points_weighted_by_opponent_rank", 0)
            - features_df.get("away_form_points_weighted_by_opponent_rank", 0)
        ).abs()
        features_df["weighted_form_points_difference"] = features_df.get(
            "home_form_points_weighted_by_opponent_rank", 0
        ) - features_df.get("away_form_points_weighted_by_opponent_rank", 0)

        return features_df

    def _apply_selected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to essential + selected if selection list exists.

        Keeps ID/meta/target columns alongside the selected feature names.
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
                        selected = [
                            s.strip() for s in val.splitlines() if s.strip()
                        ]
                elif isinstance(loaded, str):
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
            if not keep_cols:
                return df
            return df[keep_cols].copy()
        except Exception:
            return df

    def _build_team_long_df(self, matches: list[dict]) -> pd.DataFrame:
        """Build a long-format team-match DataFrame from match dicts.

        Each finished match contributes two rows (home and away) with
        goals_for/goals_against.
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
                        "opponent_team": m.get("away_team"),
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
                        "opponent_team": m.get("home_team"),
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
        # Points per match: 3 win, 1 draw, 0 loss
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
        """Compute Elo ratings chronologically.

        Returns:
            match_elos: DataFrame with columns ['match_id','home_elo','away_elo']
                (pre-match ratings)
            elo_long: DataFrame with columns ['team','date','elo'] giving
                pre-match Elo per team per match
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

        Assumes long_df has columns:
        ['match_id','date','team','goals_for','goals_against','goal_diff','points']
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
        loss_prior = (
            grp["goals_for"].shift(1) < grp["goals_against"].shift(1)
        ).astype(float)

        # Strength-of-Schedule: weight prior points by opponent rank
        try:
            table = getattr(self, "_current_table", {}) or {}
        except Exception:
            table = {}
        opponent_rank = long_df.get(
            "opponent_team", pd.Series(index=long_df.index)
        ).map(lambda t: (table.get(t, {}) or {}).get("position", np.nan))
        opponent_rank = opponent_rank.fillna(10.0).astype(float)
        opponent_weight = ((21.0 - opponent_rank) / 10.0).clip(lower=0.1, upper=2.0)

        weight_prior = opponent_weight.groupby(long_df["team"]).shift(1)
        weighted_pts_prior = pts_prior * weight_prior

        long_df["form_points_weighted_by_opponent_rank"] = (
            weighted_pts_prior.rolling(window=N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        long_df["form_points_per_game"] = long_df[
            "form_points_weighted_by_opponent_rank"
        ] / np.minimum(N, grp.cumcount() + 1)
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
        long_df["form_avg_goals_conceded"] = ga_roll / np.minimum(
            N, grp.cumcount() + 1
        )

        # Multi-window form metrics (L3/L5/L10)
        for w in (3, 5, 10):
            long_df[f"form_points_L{w}"] = (
                pts_prior.rolling(window=w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            long_df[f"wform_points_L{w}"] = (
                weighted_pts_prior.rolling(window=w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            gf_w = (
                gf_prior.rolling(window=w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            ga_w = (
                ga_prior.rolling(window=w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            long_df[f"form_goal_diff_L{w}"] = gf_w - ga_w

        long_df.fillna(0.0, inplace=True)
        return long_df

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
            if home_score is None or away_score is None:
                continue

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
