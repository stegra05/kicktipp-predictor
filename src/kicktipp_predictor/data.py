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

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional: Numba acceleration for Elo updates
try:  # pragma: no cover - optional dependency
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    njit = None

if njit is not None:
    @njit(cache=True)
    def _elo_process(
        home_idx,
        away_idx,
        match_finished,
        home_score,
        away_score,
        start_elos,
        k,
        home_adv,
    ):
        n_matches = len(home_idx)
        elos = start_elos.copy()
        home_pre = np.empty(n_matches, dtype=np.float64)
        away_pre = np.empty(n_matches, dtype=np.float64)
        for i in range(n_matches):
            hi = home_idx[i]
            ai = away_idx[i]
            he = elos[hi]
            ae = elos[ai]
            home_pre[i] = he
            away_pre[i] = ae
            if match_finished[i]:
                hs = home_score[i]
                as_ = away_score[i]
                s_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
                e_home = 1.0 / (1.0 + 10.0 ** (((ae + home_adv) - he) / 400.0))
                delta = k * (s_home - e_home)
                elos[hi] = he + delta
                elos[ai] = ae - delta
        return home_pre, away_pre, elos
else:
    def _elo_process(
        home_idx,
        away_idx,
        match_finished,
        home_score,
        away_score,
        start_elos,
        k,
        home_adv,
    ):
        elos = start_elos.copy()
        home_pre = np.empty(len(home_idx), dtype=float)
        away_pre = np.empty(len(away_idx), dtype=float)
        for i in range(len(home_idx)):
            hi = int(home_idx[i])
            ai = int(away_idx[i])
            he = float(elos[hi])
            ae = float(elos[ai])
            home_pre[i] = he
            away_pre[i] = ae
            if match_finished[i]:
                hs = int(home_score[i])
                as_ = int(away_score[i])
                s_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
                e_home = 1.0 / (1.0 + 10.0 ** (((ae + home_adv) - he) / 400.0))
                delta = k * (s_home - e_home)
                elos[hi] = he + delta
                elos[ai] = ae - delta
        return home_pre, away_pre, elos


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

        # Elo settings (defaults; can be tuned later via config)
        self._elo_base = 1500.0
        self._elo_k = 20.0
        self._elo_home_adv = 50.0
        self._elo_avg_k = 4  # average over top/bottom K teams
        self._elo_history: dict = {}

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
            # Attach season to each parsed match for downstream Elo and features
            for m in matches:
                try:
                    m["season"] = season
                except Exception:
                    pass
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
            parsed = self._parse_matches(matches_raw)
            # Attach season to each parsed match for downstream Elo and features
            for m in parsed:
                try:
                    m["season"] = season
                except Exception:
                    pass
            return parsed
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

    def _series_or_zero(
        self, df: pd.DataFrame, col: str, default: float = 0.0
    ) -> pd.Series:
        """Return a column as a numeric Series or a zero Series aligned to df.

        Ensures arithmetic operations remain vectorized and index-aligned.

        Args:
            df: Source DataFrame.
            col: Column name to fetch.
            default: Fallback fill value when column is missing or non-numeric.

        Returns:
            A numeric Series aligned to df.index.
        """
        try:
            if col in df.columns:
                s = df[col]
            else:
                s = pd.Series(default, index=df.index)
            return pd.to_numeric(s, errors="coerce").fillna(default)
        except Exception:
            return pd.Series(default, index=df.index)

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
                # Attempt to derive elo_diff from home/away elos if available
                if "elo_diff" not in df.columns and {"home_elo", "away_elo"}.issubset(df.columns):
                    df["elo_diff"] = (
                        pd.to_numeric(df["home_elo"], errors="coerce").fillna(0.0)
                        - pd.to_numeric(df["away_elo"], errors="coerce").fillna(0.0)
                    )
                # Attempt to derive normalized diff from raw Elo difference
                if "elo_diff" in df.columns:
                    home_mp = self._series_or_zero(df, "home_matches_played", 0)
                    away_mp = self._series_or_zero(df, "away_matches_played", 0)
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

    def _drop_non_tanh_elo_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove any ELO-related columns except the sanctioned 'tanh_tamed_elo'.

        This acts as a hard isolation guard to ensure that no raw or
        intermediate Elo features influence training/prediction, even if
        feature selection fails to load.
        """
        try:
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            cols = list(df.columns)
            to_drop = [
                c for c in cols
                if ("elo" in str(c).lower()) and (str(c).lower() != "tanh_tamed_elo")
            ]
            # Optionally drop ELO-derived binary to enforce isolation
            if "home_team_is_strong" in df.columns:
                to_drop.append("home_team_is_strong")
            if to_drop:
                df = df.drop(columns=[c for c in set(to_drop) if c in df.columns])
            return df
        except Exception:
            # On any unexpected error, return df unchanged to maintain stability
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

    def _create_features(
        self,
        matches_df: pd.DataFrame,
        context_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build team history features from context with shared logic.

        Args:
            matches_df: Base matches DataFrame for which features are being created.
                        Not used directly for history, but included for API symmetry.
            context_df: Context matches DataFrame used to compute historical team features.
                        Should include finished matches with valid scores and dates.

        Returns:
            A tuple of (long_hist, hist_cols):
            - long_hist: Long-format per-team history features DataFrame with computed stats.
            - hist_cols: List of history feature column names to merge per side.

        Notes:
            - Handles empty or malformed context by returning an empty DataFrame.
            - Normalizes date types and sorts for consistent cumcount and merges.
        """
        # Build long-format frame from context matches (finished only)
        try:
            context_records = context_df.to_dict(orient="records") if not context_df.empty else []
        except Exception:
            context_records = []
        long_hist = self._build_team_long_df(context_records)
        if long_hist.empty:
            return long_hist, []
        # Ensure proper dtypes and ordering
        long_hist["date"] = pd.to_datetime(long_hist["date"])
        long_hist = long_hist.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
        # Normalize key dtypes to avoid merge mismatches (e.g., int vs str)
        if "match_id" in long_hist.columns:
            try:
                long_hist["match_id"] = long_hist["match_id"].astype(str)
            except Exception:
                pass
        if "team" in long_hist.columns:
            try:
                long_hist["team"] = long_hist["team"].astype(str)
            except Exception:
                pass
        # Prior matches played per team (leakage-safe count)
        long_hist["matches_played_prior"] = long_hist.groupby("team", group_keys=False).cumcount()
        # Cache current standings for opponent rank weighting
        try:
            self._current_table = self._calculate_table(context_records)
        except Exception:
            self._current_table = {}
        # Compute team history features
        long_hist = self._compute_team_history_features(long_hist)
        # Shared history columns list
        hist_cols = [
            "avg_goals_for",
            "avg_goals_against",
            "avg_points",
            "form_points_weighted_by_opponent_rank",
            "form_points_per_game",
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
            "form_goal_diff_L3",
            "form_goal_diff_L5",
            "form_goal_diff_L10",
            "form_avg_goals_scored",
            "form_avg_goals_conceded",
            "matches_played_prior",
        ]
        return long_hist, hist_cols

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

        # Compute Elo history and attach pre-match elo_diff to base
        try:
            elo_hist = self._compute_elo_history(matches)
            by_match = elo_hist.get("by_match", {})
            if by_match:
                elo_df = pd.DataFrame(
                    [{"match_id": k, **v} for k, v in by_match.items()]
                )
                # Normalize key type for safe merge
                base["match_id"] = base["match_id"].astype(str)
                elo_df["match_id"] = elo_df["match_id"].astype(str)
                base = base.merge(elo_df, on="match_id", how="left")
            else:
                base["elo_diff"] = 0.0
        except Exception:
            base["elo_diff"] = base.get("elo_diff", 0.0)

        # Create binary feature: home_team_is_strong
        # Definition: 1 if home_elo > 1450, else 0. Exactly 1450 is NOT strong.
        # Robustness: use aligned zero series fallback when column missing/non-numeric.
        home_elo_series = self._series_or_zero(base, "home_elo", 0.0)
        base["home_team_is_strong"] = (home_elo_series > 1450.0).astype(int)

        # Shared history features via helper
        matches_df = pd.DataFrame(matches)
        long_hist, hist_cols = self._create_features(matches_df, matches_df)
        if long_hist.empty:
            return pd.DataFrame()
        # Assemble match-level features by merging home/away history rows
        features_df = self._merge_history_features(base, long_hist, hist_cols)

        # Compute tamed Elo feature (without storing raw Elo data)
        features_df = self._compute_tanh_tamed_elo(features_df)
        # Hard isolation: drop all non-tanh Elo columns
        features_df = self._drop_non_tanh_elo_columns(features_df)

        # Derived form diffs (robust to missing columns)
        h_w = self._series_or_zero(
            features_df, "home_form_points_weighted_by_opponent_rank", 0.0
        )
        a_w = self._series_or_zero(
            features_df, "away_form_points_weighted_by_opponent_rank", 0.0
        )
        features_df["abs_weighted_form_points_diff"] = (h_w - a_w).abs()
        features_df["weighted_form_points_difference"] = h_w - a_w

        # Targets
        features_df["goal_difference"] = (
            features_df["home_score"] - features_df["away_score"]
        )
        features_df["result"] = np.where(
            features_df["goal_difference"] > 0,
            "H",
            np.where(features_df["goal_difference"] < 0, "A", "D"),
        )

        features_df = self._apply_selected_features(features_df)
        # Final guard: ensure no non-tanh Elo columns slipped through
        features_df = self._drop_non_tanh_elo_columns(features_df)
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

        # Compute Elo diff for upcoming matches from historical Elo state
        try:
            elo_hist = self._compute_elo_history(historical_matches)
            season_current = elo_hist.get("season_current_elos", {})
            by_season_hist = self._group_matches_by_season(historical_matches)
            # Ensure season dtype consistency
            if "season" in upcoming.columns:
                try:
                    upcoming["season"] = upcoming["season"].astype(int)
                except Exception:
                    pass
            upcoming_seasons = (
                upcoming["season"].dropna().astype(int).unique().tolist()
                if "season" in upcoming.columns
                else []
            )
            # Initialize elos for any upcoming season not in current state
            for s in upcoming_seasons:
                if s not in season_current:
                    teams_s = set(
                        pd.Series(
                            pd.concat(
                                [
                                    upcoming.loc[upcoming["season"] == s, "home_team"],
                                    upcoming.loc[upcoming["season"] == s, "away_team"],
                                ]
                            )
                        )
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
                    prev_teams = self._teams_in_season(by_season_hist.get(s - 1, []))
                    prior_presence = self._build_prior_presence(
                        [x for x in sorted(by_season_hist.keys()) if x < s],
                        by_season_hist,
                    )
                    prev_table = (
                        self._calculate_table(by_season_hist.get(s - 1, []))
                        if (s - 1) in by_season_hist
                        else {}
                    )
                    prev_final_elos = elo_hist.get("season_final", {}).get(s - 1, {})
                    season_current[s] = self._compute_initial_elos_for_season(
                        s,
                        teams_s,
                        prev_teams,
                        prev_final_elos,
                        prev_table,
                        prior_presence,
                        int(self._elo_avg_k),
                    )

            # Vectorized mapping of home/away Elo via merge
            current_records = [
                {"season": int(season), "team": str(team), "elo": float(elo)}
                for season, team_elos in season_current.items()
                for team, elo in (team_elos or {}).items()
            ]
            current_df = pd.DataFrame(current_records)
            if not current_df.empty:
                try:
                    current_df["season"] = current_df["season"].astype(int)
                    current_df["team"] = current_df["team"].astype(str)
                except Exception:
                    pass
                # Home Elo
                upcoming = upcoming.merge(
                    current_df.rename(columns={"team": "home_team", "elo": "home_elo"}),
                    on=["season", "home_team"],
                    how="left",
                )
                # Away Elo
                upcoming = upcoming.merge(
                    current_df.rename(columns={"team": "away_team", "elo": "away_elo"}),
                    on=["season", "away_team"],
                    how="left",
                )
                # Fill missing with base and compute diff
                upcoming["home_elo"] = pd.to_numeric(upcoming.get("home_elo"), errors="coerce").fillna(self._elo_base)
                upcoming["away_elo"] = pd.to_numeric(upcoming.get("away_elo"), errors="coerce").fillna(self._elo_base)
                upcoming["elo_diff"] = upcoming["home_elo"] - upcoming["away_elo"]
            else:
                upcoming["home_elo"] = float(self._elo_base)
                upcoming["away_elo"] = float(self._elo_base)
                upcoming["elo_diff"] = 0.0
        except Exception:
            upcoming["elo_diff"] = upcoming.get("elo_diff", 0.0)

        # Shared history features via helper
        hist_df = pd.DataFrame(historical_matches)
        long_hist, hist_cols = self._create_features(upcoming, hist_df)
        if long_hist.empty:
            return pd.DataFrame()
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

        # Compute tamed Elo feature (without storing raw Elo data)
        features_df = self._compute_tanh_tamed_elo(features_df)
        # Hard isolation: drop all non-tanh Elo columns
        features_df = self._drop_non_tanh_elo_columns(features_df)

        # Derived diffs (robust to missing columns)
        h_w = self._series_or_zero(
            features_df, "home_form_points_weighted_by_opponent_rank", 0.0
        )
        a_w = self._series_or_zero(
            features_df, "away_form_points_weighted_by_opponent_rank", 0.0
        )
        features_df["abs_weighted_form_points_diff"] = (h_w - a_w).abs()
        features_df["weighted_form_points_difference"] = h_w - a_w

        features_df = self._apply_selected_features(features_df)
        # Final guard: ensure no non-tanh Elo columns slipped through
        features_df = self._drop_non_tanh_elo_columns(features_df)
        return features_df

    def _load_selected_features(self, sel_path: Path) -> list[str]:
        """Load selected features from YAML with robust fallbacks and validation.

        Priority:
        1) Use PyYAML when available.
        2) Fallback to lightweight parser for simple list YAML.
        3) Attempt JSON parse when file resembles JSON.

        Raises:
        - FileNotFoundError if file is missing.
        - ValueError if parsing succeeds but yields no valid features.
        - RuntimeError on unexpected I/O or parse errors (with clear message).
        """
        if not isinstance(sel_path, Path):
            sel_path = Path(sel_path)
        if not sel_path.exists():
            msg = f"Selected features file not found: {sel_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            text = sel_path.read_text(encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to read features file '{sel_path}': {exc}"
            logger.error(msg)
            raise RuntimeError(msg)

        # 1) Primary parse: PyYAML
        try:
            if yaml is not None:
                loaded = yaml.safe_load(text)
                if isinstance(loaded, list):
                    feats = [str(x).strip() for x in loaded if str(x).strip()]
                elif isinstance(loaded, dict) and "features" in loaded:
                    val = loaded.get("features")
                    if isinstance(val, list):
                        feats = [str(x).strip() for x in val if str(x).strip()]
                    elif isinstance(val, str):
                        feats = [s.strip() for s in val.splitlines() if s.strip()]
                    else:
                        feats = []
                elif isinstance(loaded, str):
                    feats = [s.strip() for s in loaded.splitlines() if s.strip()]
                else:
                    feats = []
            else:
                feats = []
        except Exception as exc:
            logger.warning(f"PyYAML parsing failed for '{sel_path}': {exc}")
            feats = []

        # 2) Fallback lightweight parser for simple YAML lists
        if not feats:
            try:
                lines = []
                for raw in text.splitlines():
                    # Strip comments
                    base = raw.split("#", 1)[0]
                    base = base.strip()
                    if not base:
                        continue
                    # Expect dash list items
                    if base.startswith("-"):
                        item = base[1:].strip()
                        if item:
                            lines.append(item)
                feats = lines
            except Exception as exc:
                logger.warning(f"Lightweight parsing failed for '{sel_path}': {exc}")
                feats = []

        # 3) Attempt JSON parse when applicable
        if not feats:
            try:
                stripped = "\n".join([ln.split("#", 1)[0] for ln in text.splitlines()]).strip()
                if stripped.startswith("[") or stripped.startswith("{"):
                    import json
                    obj = json.loads(stripped)
                    if isinstance(obj, list):
                        feats = [str(x).strip() for x in obj if str(x).strip()]
                    elif isinstance(obj, dict) and "features" in obj:
                        val = obj.get("features")
                        if isinstance(val, list):
                            feats = [str(x).strip() for x in val if str(x).strip()]
                        elif isinstance(val, str):
                            feats = [s.strip() for s in val.splitlines() if s.strip()]
            except Exception as exc:
                logger.warning(f"JSON parsing attempt failed for '{sel_path}': {exc}")

        # Validation
        feats = [f for f in feats if isinstance(f, str) and f]
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for f in feats:
            if f not in seen:
                deduped.append(f)
                seen.add(f)
        feats = deduped
        if not feats:
            msg = (
                f"No valid features loaded from '{sel_path}'. Ensure the file is a simple YAML list "
                f"(e.g., '- feature_name' per line) or a mapping with key 'features'."
            )
            logger.error(msg)
            raise ValueError(msg)
        return feats

    def _apply_selected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to essential + selected using robust file loading.

        Enforces exclusive loading from 'kept_features.yaml' with strong fallbacks
        and explicit error reporting when the file is missing or malformed.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        # Always prefer kept_features.yaml per requirement
        sel_filename = "kept_features.yaml"
        sel_path = self.config.paths.config_dir / sel_filename
        # Load selected features with robust fallbacks
        selected = self._load_selected_features(sel_path)
        # Assemble keep set
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
        # Warn if any listed features are missing from df
        missing = [c for c in selected if c not in df.columns]
        if missing:
            logger.warning(
                "Selected features listed in YAML but missing in dataframe: %s",
                ", ".join(missing),
            )
        keep_cols = [c for c in df.columns if (c in essential) or (c in selected)]
        if not keep_cols:
            msg = (
                "After applying selected features, no columns remain. This likely indicates that "
                "feature engineering did not produce the expected columns."
            )
            logger.error(msg)
            raise ValueError(msg)
        return df[keep_cols].copy()

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

    # === Elo computation and initialization ===
    def _group_matches_by_season(self, matches: list[dict]) -> dict[int, list[dict]]:
        seasons: dict[int, list[dict]] = {}
        for m in matches:
            s = m.get("season")
            if s is None:
                continue
            seasons.setdefault(int(s), []).append(m)
        return seasons

    def _teams_in_season(self, season_matches: list[dict]) -> set[str]:
        teams: set[str] = set()
        for m in season_matches:
            ht = m.get("home_team")
            at = m.get("away_team")
            if ht:
                teams.add(str(ht))
            if at:
                teams.add(str(at))
        return teams

    def _identify_new_teams(self, season_teams: set[str], prev_season_teams: set[str]) -> set[str]:
        return set(season_teams) - set(prev_season_teams)

    def _build_prior_presence(self, seasons: list[int], by_season: dict[int, list[dict]]) -> set[str]:
        prior: set[str] = set()
        for s in seasons:
            for m in by_season.get(s, []) or []:
                ht = m.get("home_team")
                at = m.get("away_team")
                if ht:
                    prior.add(str(ht))
                if at:
                    prior.add(str(at))
        return prior

    def _classify_new_teams(
        self,
        new_teams: set[str],
        prior_presence: set[str],
    ) -> dict[str, str]:
        """Classify new entrants heuristically: promoted vs relegated.

        Teams seen in earlier seasons of this league but absent last season
        are treated as 'relegated' (likely stronger). Teams never seen before
        are treated as 'promoted' (likely weaker).
        """
        classes: dict[str, str] = {}
        for t in new_teams:
            classes[t] = "relegated" if t in prior_presence else "promoted"
        return classes

    def _compute_prev_season_bases(
        self,
        prev_final_elos: dict[str, float] | None,
        prev_table: dict | None,
        k: int,
    ) -> tuple[float, float]:
        """Compute base Elo for promoted and relegated using prev season.

        Returns (top_avg_for_relegated, bottom_avg_for_promoted).
        Fallbacks to overall mean or 1500 when insufficient data.
        """
        prev_final_elos = prev_final_elos or {}
        prev_table = prev_table or {}
        # Sort teams by final position
        try:
            sorted_items = sorted(
                prev_table.items(), key=lambda x: x[1].get("position", 9999)
            )
        except Exception:
            sorted_items = []
        top_teams = [t for t, _ in sorted_items[:max(1, k)] if t in prev_final_elos]
        bottom_teams = [t for t, _ in sorted_items[-max(1, k):] if t in prev_final_elos]
        if top_teams:
            top_avg = float(np.mean([prev_final_elos[t] for t in top_teams]))
        else:
            top_avg = (
                float(np.mean(list(prev_final_elos.values())))
                if prev_final_elos
                else self._elo_base
            )
        if bottom_teams:
            bottom_avg = float(np.mean([prev_final_elos[t] for t in bottom_teams]))
        else:
            bottom_avg = (
                float(np.mean(list(prev_final_elos.values())))
                if prev_final_elos
                else self._elo_base
            )
        return top_avg, bottom_avg

    def _compute_initial_elos_for_season(
        self,
        season: int,
        teams_s: set[str],
        prev_teams: set[str],
        prev_final_elos: dict[str, float] | None,
        prev_table: dict | None,
        prior_presence: set[str],
        k: int,
    ) -> dict[str, float]:
        """Determine starting Elo for all teams in a season with enhanced init."""
        start: dict[str, float] = {}
        prev_final_elos = prev_final_elos or {}
        # Continuing teams keep last season's final Elo if available
        for t in teams_s & prev_teams:
            start[t] = float(prev_final_elos.get(t, self._elo_base))
        new_teams = self._identify_new_teams(teams_s, prev_teams)
        classes = self._classify_new_teams(new_teams, prior_presence)
        top_avg, bottom_avg = self._compute_prev_season_bases(prev_final_elos, prev_table, k)
        for t in new_teams:
            start[t] = float(bottom_avg if classes.get(t) == "promoted" else top_avg)
        # Any remaining (e.g., unseen teams) get neutral base
        for t in teams_s:
            start.setdefault(t, self._elo_base)
        return start

    def _compute_elo_history(self, matches: list[dict]) -> dict:
        """Compute Elo by season and pre-match ratings with enhanced initialization.

        Produces:
          - by_match: {match_id: {home_elo, away_elo, elo_diff}}
          - season_initial: {season: {team: elo}}
          - season_final: {season: {team: elo}}
          - season_current_elos: {season: {team: elo}} at the last processed match
        """
        if not matches:
            self._elo_history = {
                "by_match": {},
                "season_initial": {},
                "season_final": {},
                "season_current_elos": {},
            }
            return self._elo_history

        by_season = self._group_matches_by_season(matches)
        seasons_sorted = sorted(by_season.keys())
        season_initial: dict[int, dict[str, float]] = {}
        season_final: dict[int, dict[str, float]] = {}
        by_match: dict[int | str, dict[str, float]] = {}
        season_current: dict[int, dict[str, float]] = {}

        prev_final_elos: dict[str, float] = {}
        for s in seasons_sorted:
            season_matches = sorted(
                by_season.get(s, []),
                key=lambda m: (m.get("date"), int(m.get("matchday", 0)), str(m.get("match_id"))),
            )
            teams_s = self._teams_in_season(season_matches)
            prev_teams = self._teams_in_season(by_season.get(s - 1, []))
            prior_presence = self._build_prior_presence(
                [x for x in seasons_sorted if x < s], by_season
            )
            prev_table = self._calculate_table(by_season.get(s - 1, [])) if (s - 1) in by_season else {}

            # Initial elos for the season
            start_elos = self._compute_initial_elos_for_season(
                s, teams_s, prev_teams, prev_final_elos, prev_table, prior_presence, int(self._elo_avg_k)
            )
            season_initial[s] = dict(start_elos)
            # Accelerated Elo processing across the season
            team_list = sorted(list(teams_s))
            team_to_idx = {t: i for i, t in enumerate(team_list)}
            start_elos_arr = np.array(
                [float(start_elos.get(t, self._elo_base)) for t in team_list],
                dtype=np.float64,
            )
            n_matches = len(season_matches)
            home_idx = np.empty(n_matches, dtype=np.int64)
            away_idx = np.empty(n_matches, dtype=np.int64)
            finished = np.empty(n_matches, dtype=np.bool_)
            home_score_arr = np.empty(n_matches, dtype=np.int64)
            away_score_arr = np.empty(n_matches, dtype=np.int64)
            match_ids: list[int | str] = []

            for i, m in enumerate(season_matches):
                ht = str(m.get("home_team"))
                at = str(m.get("away_team"))
                home_idx[i] = int(team_to_idx.get(ht, 0))
                away_idx[i] = int(team_to_idx.get(at, 0))
                hs = m.get("home_score")
                as_ = m.get("away_score")
                finished[i] = bool(m.get("is_finished") and hs is not None and as_ is not None)
                home_score_arr[i] = int(hs if hs is not None else 0)
                away_score_arr[i] = int(as_ if as_ is not None else 0)
                match_ids.append(m.get("match_id"))

            he_pre, ae_pre, final_elos_arr = _elo_process(
                home_idx,
                away_idx,
                finished,
                home_score_arr,
                away_score_arr,
                start_elos_arr,
                float(self._elo_k),
                float(self._elo_home_adv),
            )

            # Record pre-match elos per game
            for i in range(n_matches):
                he = float(he_pre[i])
                ae = float(ae_pre[i])
                by_match[match_ids[i]] = {
                    "home_elo": he,
                    "away_elo": ae,
                    "elo_diff": he - ae,
                }

            # Final state
            season_final[s] = {team_list[i]: float(final_elos_arr[i]) for i in range(len(team_list))}
            season_current[s] = dict(season_final[s])
            prev_final_elos = dict(season_final[s])

        self._elo_history = {
            "by_match": by_match,
            "season_initial": season_initial,
            "season_final": season_final,
            "season_current_elos": season_current,
        }
        return self._elo_history

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
