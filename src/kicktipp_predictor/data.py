"""Data loading and feature engineering for kicktipp predictor.

This module combines data fetching from OpenLigaDB API with
comprehensive feature engineering for match prediction.
"""

import requests
import pandas as pd
import numpy as np
import pickle
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from .config import get_config


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

    def fetch_season_matches(self, season: Optional[int] = None) -> List[Dict]:
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
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        url = f"{self.base_url}/getmatchdata/{self.league_code}/{season}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            matches_raw = response.json()

            matches = self._parse_matches(matches_raw)

            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump(matches, f)

            return matches

        except Exception as e:
            print(f"Error fetching matches for season {season}: {e}")
            # Try to return cached data even if expired
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            return []

    def fetch_matchday(self, matchday: int, season: Optional[int] = None) -> List[Dict]:
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

    def get_current_matchday(self, season: Optional[int] = None) -> int:
        """Get the current matchday number."""
        matches = self.fetch_season_matches(season)

        now = datetime.now()

        # Find the next unplayed matchday
        for match in matches:
            if match['date'] > now and not match['is_finished']:
                return match['matchday']

        # If all matches are finished, return last matchday
        if matches:
            return matches[-1]['matchday']

        return 1

    def get_upcoming_matches(self, days: int = 7, season: Optional[int] = None) -> List[Dict]:
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
            m for m in matches
            if now <= m['date'] <= future_date and not m['is_finished']
        ]

        return sorted(upcoming, key=lambda x: x['date'])

    def fetch_historical_seasons(self, start_season: int, end_season: int) -> List[Dict]:
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
                print(f"Fetching season {season}/{season+1}...")
            matches = self.fetch_season_matches(season)
            all_matches.extend(matches)
            time.sleep(0.5)  # Be nice to the API

        return all_matches

    def _parse_matches(self, matches_raw: List[Dict]) -> List[Dict]:
        """Parse raw API response into standardized match dictionaries."""
        matches = []

        for match in matches_raw:
            try:
                match_dict = {
                    'match_id': match.get('matchID'),
                    'matchday': match.get('group', {}).get('groupOrderID', 0),
                    'date': datetime.fromisoformat(match['matchDateTime'].replace('Z', '+00:00')),
                    'home_team': match['team1']['teamName'],
                    'away_team': match['team2']['teamName'],
                    'is_finished': match['matchIsFinished'],
                }

                # Add results if match is finished
                if match['matchIsFinished'] and match['matchResults']:
                    final_result = [r for r in match['matchResults'] if r['resultTypeID'] == 2]
                    if final_result:
                        result = final_result[0]
                        match_dict['home_score'] = result['pointsTeam1']
                        match_dict['away_score'] = result['pointsTeam2']
                    else:
                        # Fallback to any result
                        result = match['matchResults'][0]
                        match_dict['home_score'] = result['pointsTeam1']
                        match_dict['away_score'] = result['pointsTeam2']
                else:
                    match_dict['home_score'] = None
                    match_dict['away_score'] = None

                matches.append(match_dict)

            except Exception as e:
                print(f"Error parsing match: {e}")
                continue

        return matches

    # =================================================================
    # FEATURE ENGINEERING METHODS
    # =================================================================

    def create_features_from_matches(self, matches: List[Dict]) -> pd.DataFrame:
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
            base['is_finished'] & base['home_score'].notna() & base['away_score'].notna()
        ].copy()
        if base.empty:
            return pd.DataFrame()
        base['date'] = pd.to_datetime(base['date'])

        # Precompute team-long frame and vectorized history features
        long_df = self._build_team_long_df(matches)
        if long_df.empty:
            return pd.DataFrame()
        long_df['date'] = pd.to_datetime(long_df['date'])
        long_df = long_df.sort_values(['team', 'date', 'match_id']).reset_index(drop=True)

        long_df = self._compute_team_history_features(long_df)

        # Precompute EWMA recency features once across the dataset and merge
        try:
            ewma_long_df = self._compute_ewma_recency_features(matches, span=5)
        except Exception:
            ewma_long_df = None

        # Assemble match-level features by merging home/away history rows for the same match_id
        # Select team history columns to attach
        hist_cols = [
            'avg_goals_for', 'avg_goals_against', 'avg_points',
            'form_points', 'form_points_per_game', 'form_wins', 'form_draws', 'form_losses',
            'form_goals_scored', 'form_goals_conceded', 'form_goal_diff',
            'form_avg_goals_scored', 'form_avg_goals_conceded',
            'momentum_points', 'momentum_goals', 'momentum_conceded', 'momentum_score',
            # venue-specific history metrics
            'points_pg_at_home', 'goals_pg_at_home', 'goals_conceded_pg_at_home',
            'points_pg_away', 'goals_pg_away', 'goals_conceded_pg_away',
        ]
        right_cols = ['match_id', 'team'] + hist_cols
        home_merge = long_df[right_cols].rename(columns={'team': 'home_team'})
        away_merge = long_df[right_cols].rename(columns={'team': 'away_team'})

        features_df = base.copy()
        # Merge home team features
        features_df = features_df.merge(
            home_merge.add_prefix('home_'),
            left_on=['match_id', 'home_team'],
            right_on=['home_match_id', 'home_home_team'],
            how='left'
        )
        # Merge away team features
        features_df = features_df.merge(
            away_merge.add_prefix('away_'),
            left_on=['match_id', 'away_team'],
            right_on=['away_match_id', 'away_away_team'],
            how='left'
        )

        # Drop merge helper columns
        drop_cols = [
            'home_match_id', 'home_home_team', 'away_match_id', 'away_away_team'
        ]
        for c in drop_cols:
            if c in features_df.columns:
                features_df.drop(columns=[c], inplace=True)

        # Attach EWMA recency features via vectorized merges
        if ewma_long_df is not None and not ewma_long_df.empty:
            # Home EWMA
            home_ewm = ewma_long_df.rename(columns={'team': 'home_team', 'match_id': 'home_match_id'})
            features_df = features_df.merge(
                home_ewm.add_prefix('home_'),
                left_on=['match_id', 'home_team'],
                right_on=['home_home_match_id', 'home_home_team'],
                how='left'
            )
            # Away EWMA
            away_ewm = ewma_long_df.rename(columns={'team': 'away_team', 'match_id': 'away_match_id'})
            features_df = features_df.merge(
                away_ewm.add_prefix('away_'),
                left_on=['match_id', 'away_team'],
                right_on=['away_away_match_id', 'away_away_team'],
                how='left'
            )
            # Clean helper columns
            for c in [
                'home_home_match_id','home_home_team','away_away_match_id','away_away_team'
            ]:
                if c in features_df.columns:
                    features_df.drop(columns=[c], inplace=True)

            # Rename ewm columns to expected names (home_/away_ prefix placement)
            ewm_map = {
                'goals_for_ewm5': 'goals_for_ewm5',
                'goals_against_ewm5': 'goals_against_ewm5',
                'goal_diff_ewm5': 'goal_diff_ewm5',
                'points_ewm5': 'points_ewm5',
            }
            for base_col in ewm_map.values():
                hcol = f'home_{base_col}'
                acol = f'away_{base_col}'
                # Already prefixed by add_prefix; columns exist as home_<col>, away_<col>
                # Ensure they exist
                if hcol not in features_df.columns:
                    features_df[hcol] = 0.0
                if acol not in features_df.columns:
                    features_df[acol] = 0.0

        # Compute simple derived differences
        features_df['abs_form_points_diff'] = (
            (features_df.get('home_form_points', 0) - features_df.get('away_form_points', 0)).abs()
        )
        features_df['form_points_difference'] = (
            features_df.get('home_form_points', 0) - features_df.get('away_form_points', 0)
        )
        features_df['abs_momentum_score_diff'] = (
            (features_df.get('home_momentum_score', 0.0) - features_df.get('away_momentum_score', 0.0)).abs()
        )
        features_df['momentum_score_difference'] = (
            features_df.get('home_momentum_score', 0.0) - features_df.get('away_momentum_score', 0.0)
        )

        # Venue deltas
        if all(c in features_df.columns for c in ['home_points_pg_at_home', 'away_points_pg_away']):
            features_df['venue_points_delta'] = features_df['home_points_pg_at_home'] - features_df['away_points_pg_away']
        if all(c in features_df.columns for c in ['home_goals_pg_at_home', 'away_goals_pg_away']):
            features_df['venue_goals_delta'] = features_df['home_goals_pg_at_home'] - features_df['away_goals_pg_away']
        if all(c in features_df.columns for c in ['home_goals_conceded_pg_at_home', 'away_goals_conceded_pg_away']):
            features_df['venue_conceded_delta'] = features_df['home_goals_conceded_pg_at_home'] - features_df['away_goals_conceded_pg_away']

        # Interaction ratios (safe divisions)
        eps = 1e-6
        def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a.astype(float) / (b.astype(float) + eps)

        if all(c in features_df.columns for c in ['home_form_avg_goals_scored', 'away_form_avg_goals_conceded']):
            features_df['attack_defense_form_ratio_home'] = _safe_div(features_df['home_form_avg_goals_scored'], features_df['away_form_avg_goals_conceded'])
        if all(c in features_df.columns for c in ['away_form_avg_goals_scored', 'home_form_avg_goals_conceded']):
            features_df['attack_defense_form_ratio_away'] = _safe_div(features_df['away_form_avg_goals_scored'], features_df['home_form_avg_goals_conceded'])
        if all(c in features_df.columns for c in ['home_avg_goals_for', 'away_avg_goals_against']):
            features_df['attack_defense_long_ratio_home'] = _safe_div(features_df['home_avg_goals_for'], features_df['away_avg_goals_against'])
        if all(c in features_df.columns for c in ['away_avg_goals_for', 'home_avg_goals_against']):
            features_df['attack_defense_long_ratio_away'] = _safe_div(features_df['away_avg_goals_for'], features_df['home_avg_goals_against'])
        if all(c in features_df.columns for c in ['home_form_points_per_game', 'away_form_points_per_game']):
            features_df['form_points_pg_ratio'] = _safe_div(features_df['home_form_points_per_game'], features_df['away_form_points_per_game'])
        if all(c in features_df.columns for c in ['home_momentum_score', 'away_momentum_score']):
            features_df['momentum_score_ratio'] = _safe_div(features_df['home_momentum_score'], features_df['away_momentum_score'])
        if all(c in features_df.columns for c in ['home_points_ewm5', 'away_points_ewm5']):
            features_df['ewm_points_ratio'] = _safe_div(features_df['home_points_ewm5'], features_df['away_points_ewm5'])

        # Targets
        features_df['goal_difference'] = features_df['home_score'] - features_df['away_score']
        features_df['result'] = np.where(
            features_df['goal_difference'] > 0, 'H', np.where(features_df['goal_difference'] < 0, 'A', 'D')
        )

        # Ensure deterministic column order (optional)
        return features_df

    def create_prediction_features(self, upcoming_matches: List[Dict],
                                   historical_matches: List[Dict]) -> pd.DataFrame:
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
        upcoming['date'] = pd.to_datetime(upcoming['date'])

        # Build long_df and history features from historical (finished) matches only
        long_hist = self._build_team_long_df(historical_matches)
        if long_hist.empty:
            return pd.DataFrame()
        long_hist['date'] = pd.to_datetime(long_hist['date'])
        long_hist = long_hist.sort_values(['team', 'date', 'match_id']).reset_index(drop=True)

        long_hist = self._compute_team_history_features(long_hist)

        # Columns to use for asof merge
        hist_cols = [
            'avg_goals_for', 'avg_goals_against', 'avg_points',
            'form_points', 'form_points_per_game', 'form_wins', 'form_draws', 'form_losses',
            'form_goals_scored', 'form_goals_conceded', 'form_goal_diff',
            'form_avg_goals_scored', 'form_avg_goals_conceded',
            'momentum_points', 'momentum_goals', 'momentum_conceded', 'momentum_score',
            # venue-specific history metrics
            'points_pg_at_home', 'goals_pg_at_home', 'goals_conceded_pg_at_home',
            'points_pg_away', 'goals_pg_away', 'goals_conceded_pg_away',
        ]
        hist_merge = long_hist[['team', 'date'] + hist_cols].copy()

        # Prepare per-side frames for asof merge (requires sorting by date)
        upcoming = upcoming.sort_values('date')

        # Home side merge
        home_hist = hist_merge.rename(columns={'team': 'home_team'})
        # merge_asof requires the right keys (on=date) to be globally sorted by 'date'
        home_hist = home_hist.sort_values(['date'])
        features_df = pd.merge_asof(
            upcoming.sort_values('date'),
            home_hist,
            left_on='date', right_on='date',
            left_by='home_team', right_by='home_team',
            direction='backward'
        )
        # Prefix home columns
        for col in hist_cols:
            if col in features_df.columns:
                features_df.rename(columns={col: f'home_{col}'}, inplace=True)

        # Away side merge
        away_hist = hist_merge.rename(columns={'team': 'away_team'})
        # Ensure global sort by 'date' for merge_asof
        away_hist = away_hist.sort_values(['date'])
        features_df = pd.merge_asof(
            features_df.sort_values('date'),
            away_hist,
            left_on='date', right_on='date',
            left_by='away_team', right_by='away_team',
            direction='backward',
            suffixes=('', '_away')
        )
        # Prefix away columns
        for col in hist_cols:
            away_src = f'{col}_away'
            if away_src in features_df.columns:
                features_df.rename(columns={away_src: f'away_{col}'}, inplace=True)
        # If no name collision occurred during merge_asof, pandas will not add the
        # '_away' suffix and the right-hand columns keep their base names. Ensure
        # those get prefixed as away_* so they match the trained feature schema.
        for col in hist_cols:
            if f'away_{col}' not in features_df.columns and col in features_df.columns:
                features_df.rename(columns={col: f'away_{col}'}, inplace=True)

        # Attach EWMA recency features from historical via asof
        try:
            ewma_long_df = self._compute_ewma_recency_features(historical_matches, span=5)
        except Exception:
            ewma_long_df = None

        if ewma_long_df is not None and not ewma_long_df.empty:
            # Prepare EWMA frames
            ewm_home = ewma_long_df.rename(columns={'team': 'home_team'})[['home_team', 'date', 'goals_for_ewm5', 'goals_against_ewm5', 'goal_diff_ewm5', 'points_ewm5']]
            # Ensure global sort by 'date' for merge_asof
            ewm_home = ewm_home.sort_values(['date'])
            ewm_away = ewma_long_df.rename(columns={'team': 'away_team'})[['away_team', 'date', 'goals_for_ewm5', 'goals_against_ewm5', 'goal_diff_ewm5', 'points_ewm5']]
            # Ensure global sort by 'date' for merge_asof
            ewm_away = ewm_away.sort_values(['date'])

            features_df = pd.merge_asof(
                features_df.sort_values('date'), ewm_home,
                left_on='date', right_on='date', left_by='home_team', right_by='home_team', direction='backward'
            )
            features_df = pd.merge_asof(
                features_df.sort_values('date'), ewm_away,
                left_on='date', right_on='date', left_by='away_team', right_by='away_team', direction='backward', suffixes=('', '_away')
            )
            # Rename to home_/away_ prefixed
            for base in ['goals_for_ewm5', 'goals_against_ewm5', 'goal_diff_ewm5', 'points_ewm5']:
                if base in features_df.columns:
                    features_df.rename(columns={base: f'home_{base}'}, inplace=True)
                away_col = f'{base}_away'
                if away_col in features_df.columns:
                    features_df.rename(columns={away_col: f'away_{base}'}, inplace=True)

        # Derived diffs
        features_df['abs_form_points_diff'] = (
            (features_df.get('home_form_points', 0) - features_df.get('away_form_points', 0)).abs()
        )
        features_df['form_points_difference'] = (
            features_df.get('home_form_points', 0) - features_df.get('away_form_points', 0)
        )
        features_df['abs_momentum_score_diff'] = (
            (features_df.get('home_momentum_score', 0.0) - features_df.get('away_momentum_score', 0.0)).abs()
        )
        features_df['momentum_score_difference'] = (
            features_df.get('home_momentum_score', 0.0) - features_df.get('away_momentum_score', 0.0)
        )

        # Venue deltas
        if all(c in features_df.columns for c in ['home_points_pg_at_home', 'away_points_pg_away']):
            features_df['venue_points_delta'] = features_df['home_points_pg_at_home'] - features_df['away_points_pg_away']
        if all(c in features_df.columns for c in ['home_goals_pg_at_home', 'away_goals_pg_away']):
            features_df['venue_goals_delta'] = features_df['home_goals_pg_at_home'] - features_df['away_goals_pg_away']
        if all(c in features_df.columns for c in ['home_goals_conceded_pg_at_home', 'away_goals_conceded_pg_away']):
            features_df['venue_conceded_delta'] = features_df['home_goals_conceded_pg_at_home'] - features_df['away_goals_conceded_pg_away']

        # Interaction ratios (safe divisions)
        eps = 1e-6
        def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a.astype(float) / (b.astype(float) + eps)

        if all(c in features_df.columns for c in ['home_form_avg_goals_scored', 'away_form_avg_goals_conceded']):
            features_df['attack_defense_form_ratio_home'] = _safe_div(features_df['home_form_avg_goals_scored'], features_df['away_form_avg_goals_conceded'])
        if all(c in features_df.columns for c in ['away_form_avg_goals_scored', 'home_form_avg_goals_conceded']):
            features_df['attack_defense_form_ratio_away'] = _safe_div(features_df['away_form_avg_goals_scored'], features_df['home_form_avg_goals_conceded'])
        if all(c in features_df.columns for c in ['home_avg_goals_for', 'away_avg_goals_against']):
            features_df['attack_defense_long_ratio_home'] = _safe_div(features_df['home_avg_goals_for'], features_df['away_avg_goals_against'])
        if all(c in features_df.columns for c in ['away_avg_goals_for', 'home_avg_goals_against']):
            features_df['attack_defense_long_ratio_away'] = _safe_div(features_df['away_avg_goals_for'], features_df['home_avg_goals_against'])
        if all(c in features_df.columns for c in ['home_form_points_per_game', 'away_form_points_per_game']):
            features_df['form_points_pg_ratio'] = _safe_div(features_df['home_form_points_per_game'], features_df['away_form_points_per_game'])
        if all(c in features_df.columns for c in ['home_momentum_score', 'away_momentum_score']):
            features_df['momentum_score_ratio'] = _safe_div(features_df['home_momentum_score'], features_df['away_momentum_score'])
        if all(c in features_df.columns for c in ['home_points_ewm5', 'away_points_ewm5']):
            features_df['ewm_points_ratio'] = _safe_div(features_df['home_points_ewm5'], features_df['away_points_ewm5'])

        return features_df

    def _create_match_features(self, match: Dict, historical_matches: List[Dict],
                              is_prediction: bool = False,
                              ewma_long_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """Create features for a single match."""

        home_team = match['home_team']
        away_team = match['away_team']

        # Restrict to finished matches strictly before current match date
        ref_hist = [m for m in historical_matches
                    if (m.get('is_finished')
                        and m.get('date') is not None
                        and m.get('home_score') is not None
                        and m.get('away_score') is not None
                        and m['date'] < match['date'])]

        # Get team histories from ref_hist only
        home_history = [m for m in ref_hist if
                       (m['home_team'] == home_team or m['away_team'] == home_team)]

        away_history = [m for m in ref_hist if
                       (m['home_team'] == away_team or m['away_team'] == away_team)]

        # Adaptive minimum history by matchday (early season fallback)
        try:
            md = int(match.get('matchday', 0) or 0)
        except Exception:
            md = 0
        if md <= 3:
            min_hist = 1
        elif md <= 5:
            min_hist = 2
        else:
            min_hist = 3

        if len(home_history) < min_hist or len(away_history) < min_hist:
            return None

        features = {
            'match_id': match['match_id'],
            'matchday': match['matchday'],
            'date': match['date'],
            'home_team': home_team,
            'away_team': away_team,
        }

        # Precomputed EWMA recency features (leakage-safe)
        if ewma_long_df is not None:
            try:
                self._attach_ewma_features(features, match, ewma_long_df, span=5, is_prediction=is_prediction)
            except Exception:
                pass

        # Recent form features
        features.update(self._get_form_features(home_team, home_history, prefix='home'))
        features.update(self._get_form_features(away_team, away_history, prefix='away'))

        # Absolute differences (draw indicators)
        features['abs_form_points_diff'] = abs(
            features.get('home_form_points', 0) - features.get('away_form_points', 0)
        )

        # Momentum features
        features.update(self._get_momentum_features(home_team, home_history, prefix='home'))
        features.update(self._get_momentum_features(away_team, away_history, prefix='away'))

        features['abs_momentum_score_diff'] = abs(
            features.get('home_momentum_score', 0.0) - features.get('away_momentum_score', 0.0)
        )

        # Signed separation features (explicit relative strength)
        features['form_points_difference'] = (
            features.get('home_form_points', 0) - features.get('away_form_points', 0)
        )
        features['momentum_score_difference'] = (
            features.get('home_momentum_score', 0.0) - features.get('away_momentum_score', 0.0)
        )

        # Head-to-head features
        features.update(self._get_h2h_features(home_team, away_team, ref_hist))

        # Home/away specific features
        features.update(self._get_home_away_features(home_team, home_history, 'home'))
        features.update(self._get_home_away_features(away_team, away_history, 'away'))

        # Goal statistics
        features.update(self._get_goal_stats(home_team, home_history, prefix='home'))
        features.update(self._get_goal_stats(away_team, away_history, prefix='away'))

        # Strength of schedule
        features.update(self._get_strength_of_schedule(home_team, home_history,
                                                       ref_hist, prefix='home'))
        features.update(self._get_strength_of_schedule(away_team, away_history,
                                                       ref_hist, prefix='away'))

        # Rest days since last match
        features.update(self._get_rest_features(home_team, away_team, match, ref_hist))

        # League position and points
        features.update(self._get_table_features(home_team, away_team, ref_hist))

        # Target variables (only if not prediction)
        if not is_prediction:
            # Guard against matches that are marked finished but have no valid scores (e.g., annulled)
            home_score = match.get('home_score')
            away_score = match.get('away_score')
            if home_score is None or away_score is None:
                return None

            features['home_score'] = home_score
            features['away_score'] = away_score
            features['goal_difference'] = home_score - away_score

            if home_score > away_score:
                features['result'] = 'H'
            elif home_score < away_score:
                features['result'] = 'A'
            else:
                features['result'] = 'D'

        return features

    def _build_team_long_df(self, matches: List[Dict]) -> pd.DataFrame:
        """Build a long-format team-match DataFrame from match dicts.

        Each finished match contributes two rows (home and away) with goals_for/against.
        """
        rows = []
        for m in matches:
            try:
                if not m.get('is_finished'):
                    continue
                date = m.get('date')
                if date is None:
                    continue
                hs = m.get('home_score')
                as_ = m.get('away_score')
                if hs is None or as_ is None:
                    continue
                # Home row
                rows.append({
                    'match_id': m.get('match_id'),
                    'date': date,
                    'team': m.get('home_team'),
                    'goals_for': hs,
                    'goals_against': as_,
                    'at_home': True,
                })
                # Away row
                rows.append({
                    'match_id': m.get('match_id'),
                    'date': date,
                    'team': m.get('away_team'),
                    'goals_for': as_,
                    'goals_against': hs,
                    'at_home': False,
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(columns=['match_id', 'date', 'team'])

        long_df = pd.DataFrame(rows)
        long_df['goal_diff'] = long_df['goals_for'] - long_df['goals_against']
        # Points per match: 3 win, 1 draw, 0 loss (from perspective of team)
        long_df['points'] = np.select(
            [long_df['goals_for'] > long_df['goals_against'], long_df['goals_for'] == long_df['goals_against']],
            [3, 1], default=0
        )
        return long_df

    def _compute_team_history_features(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Compute leakage-safe per-team expanding/rolling and momentum features.

        Assumes long_df has columns: ['match_id','date','team','goals_for','goals_against','goal_diff','points']
        """
        if long_df.empty:
            return long_df

        long_df = long_df.sort_values(['team', 'date', 'match_id']).reset_index(drop=True)
        grp = long_df.groupby('team', group_keys=False)

        # Shifted prior series to prevent leakage
        gf_prior = grp['goals_for'].shift(1)
        ga_prior = grp['goals_against'].shift(1)
        pts_prior = grp['points'].shift(1)

        long_df['avg_goals_for'] = gf_prior.expanding().mean().reset_index(level=0, drop=True)
        long_df['avg_goals_against'] = ga_prior.expanding().mean().reset_index(level=0, drop=True)
        long_df['avg_points'] = pts_prior.expanding().mean().reset_index(level=0, drop=True)

        # Rolling last-N form stats (on prior rows)
        try:
            N = int(getattr(self.config.model, 'form_last_n', 5))
        except Exception:
            N = 5

        win_prior = (grp['goals_for'].shift(1) > grp['goals_against'].shift(1)).astype(float)
        draw_prior = (grp['goals_for'].shift(1) == grp['goals_against'].shift(1)).astype(float)
        loss_prior = (grp['goals_for'].shift(1) < grp['goals_against'].shift(1)).astype(float)

        long_df['form_points'] = pts_prior.rolling(window=N, min_periods=1).sum().reset_index(level=0, drop=True)
        long_df['form_points_per_game'] = long_df['form_points'] / np.minimum(
            N, grp.cumcount() + 1
        )
        long_df['form_wins'] = win_prior.rolling(window=N, min_periods=1).sum().reset_index(level=0, drop=True)
        long_df['form_draws'] = draw_prior.rolling(window=N, min_periods=1).sum().reset_index(level=0, drop=True)
        long_df['form_losses'] = loss_prior.rolling(window=N, min_periods=1).sum().reset_index(level=0, drop=True)

        gf_roll = gf_prior.rolling(window=N, min_periods=1).sum().reset_index(level=0, drop=True)
        ga_roll = ga_prior.rolling(window=N, min_periods=1).sum().reset_index(level=0, drop=True)
        long_df['form_goals_scored'] = gf_roll
        long_df['form_goals_conceded'] = ga_roll
        long_df['form_goal_diff'] = gf_roll - ga_roll
        long_df['form_avg_goals_scored'] = gf_roll / np.minimum(N, grp.cumcount() + 1)
        long_df['form_avg_goals_conceded'] = ga_roll / np.minimum(N, grp.cumcount() + 1)

        # Momentum features via EWMA on prior rows
        try:
            decay = float(getattr(self.config.model, 'momentum_decay', 0.9))
        except Exception:
            decay = 0.9
        alpha = max(1e-6, 1.0 - min(max(decay, 0.0), 0.9999))

        long_df['momentum_points'] = pts_prior.groupby(long_df['team']).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        long_df['momentum_goals'] = gf_prior.groupby(long_df['team']).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        long_df['momentum_conceded'] = ga_prior.groupby(long_df['team']).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        long_df['momentum_score'] = (
            (long_df['momentum_points'].fillna(0)) +
            (long_df['momentum_goals'].fillna(0)) * 0.5 -
            (long_df['momentum_conceded'].fillna(0)) * 0.3
        )

        # Fill initial NaNs
        long_df.fillna(0.0, inplace=True)

        # Venue-specific last-N metrics (home vs away), leakage-safe via shift(1)
        try:
            home_mask = long_df['at_home'] == True
            away_mask = long_df['at_home'] == False

            # Group within venue subsets
            grp_home = long_df[home_mask].groupby('team', group_keys=False)
            grp_away = long_df[away_mask].groupby('team', group_keys=False)

            # Points per game (mean of prior points over last N at that venue)
            long_df.loc[home_mask, 'points_pg_at_home'] = (
                grp_home['points'].shift(1).rolling(window=N, min_periods=1).mean().values
            )
            long_df.loc[away_mask, 'points_pg_away'] = (
                grp_away['points'].shift(1).rolling(window=N, min_periods=1).mean().values
            )

            # Goals for per game at venue
            long_df.loc[home_mask, 'goals_pg_at_home'] = (
                grp_home['goals_for'].shift(1).rolling(window=N, min_periods=1).mean().values
            )
            long_df.loc[away_mask, 'goals_pg_away'] = (
                grp_away['goals_for'].shift(1).rolling(window=N, min_periods=1).mean().values
            )

            # Goals conceded per game at venue
            long_df.loc[home_mask, 'goals_conceded_pg_at_home'] = (
                grp_home['goals_against'].shift(1).rolling(window=N, min_periods=1).mean().values
            )
            long_df.loc[away_mask, 'goals_conceded_pg_away'] = (
                grp_away['goals_against'].shift(1).rolling(window=N, min_periods=1).mean().values
            )

            # Fill remaining NaNs with zeros
            for c in [
                'points_pg_at_home', 'goals_pg_at_home', 'goals_conceded_pg_at_home',
                'points_pg_away', 'goals_pg_away', 'goals_conceded_pg_away'
            ]:
                if c in long_df.columns:
                    long_df[c] = long_df[c].fillna(0.0)
        except Exception:
            # If venue columns are missing for any reason, proceed without them
            pass

        return long_df

    def _compute_ewma_recency_features(self, matches: List[Dict], span: int = 5) -> pd.DataFrame:
        """Compute leakage-safe EWMA recency features on a long-format team-match frame.

        Returns a DataFrame with columns: ['match_id','date','team', '<metric>_ewm{span}', ...]
        where each EWMA is computed on prior values only (via groupby.shift(1)).
        """
        long_df = self._build_team_long_df(matches)
        if long_df.empty:
            return long_df

        long_df = long_df.sort_values(['team', 'date']).reset_index(drop=True)

        metrics = ['goals_for', 'goals_against', 'goal_diff', 'points']
        for col in metrics:
            prior_col = f'{col}_prior'
            ewm_col = f'{col}_ewm{span}'
            long_df[prior_col] = long_df.groupby('team')[col].shift(1)
            # EWMA on prior values to avoid leakage
            long_df[ewm_col] = (
                long_df.groupby('team')[prior_col]
                       .transform(lambda s: s.ewm(span=span, adjust=False).mean())
            )
            # Drop helper
            long_df.drop(columns=[prior_col], inplace=True)

        # Fill early NaNs with global means (season not tracked explicitly)
        for col in [f'{m}_ewm{span}' for m in metrics]:
            if col in long_df.columns:
                long_df[col] = long_df[col].fillna(long_df[col].mean())

        # Keep only lookup-relevant columns
        keep_cols = ['match_id', 'date', 'team'] + [f'{m}_ewm{span}' for m in metrics]
        return long_df[keep_cols]

    def _attach_ewma_features(self, features: Dict, match: Dict, ewma_long_df: pd.DataFrame,
                               span: int = 5, is_prediction: bool = False) -> None:
        """Attach precomputed EWMA features for home/away teams to the features dict.

        For finished matches, uses the EWMA row corresponding to the match_id.
        For upcoming matches, uses the last available EWMA before the match date.
        Falls back to global means if a team has no history.
        """
        if ewma_long_df is None or ewma_long_df.empty:
            return

        match_id = match.get('match_id')
        match_date = match.get('date')
        home_team = match.get('home_team')
        away_team = match.get('away_team')

        ewm_cols = [c for c in ewma_long_df.columns if c.endswith(f'_ewm{span}')]
        global_means = {c: float(ewma_long_df[c].mean()) for c in ewm_cols}

        def lookup(team: str) -> Dict[str, float]:
            if team is None:
                return {c: global_means[c] for c in ewm_cols}
            df_team = ewma_long_df[ewma_long_df['team'] == team]
            if df_team.empty:
                return {c: global_means[c] for c in ewm_cols}
            row = None
            if not is_prediction and match_id is not None:
                df_mid = df_team[df_team['match_id'] == match_id]
                if not df_mid.empty:
                    row = df_mid.iloc[0]
            if row is None and match_date is not None:
                df_before = df_team[df_team['date'] < match_date]
                if not df_before.empty:
                    row = df_before.sort_values('date').iloc[-1]
            if row is None:
                return {c: global_means[c] for c in ewm_cols}
            return {c: float(row[c]) for c in ewm_cols}

        home_vals = lookup(home_team)
        away_vals = lookup(away_team)

        # Map to prefixed feature names
        mapping = {
            f'goals_for_ewm{span}': 'goals_for',
            f'goals_against_ewm{span}': 'goals_against',
            f'goal_diff_ewm{span}': 'goal_diff',
            f'points_ewm{span}': 'points',
        }

        for ewm_col, base in mapping.items():
            features[f'home_{base}_ewm{span}'] = home_vals.get(ewm_col, global_means.get(ewm_col, 0.0))
            features[f'away_{base}_ewm{span}'] = away_vals.get(ewm_col, global_means.get(ewm_col, 0.0))

    def _get_form_features(self, team: str, history: List[Dict], prefix: str,
                          last_n: int = 5) -> Dict:
        """Calculate team form over last N matches."""
        # Allow config override for last_n
        try:
            cfg_last_n = int(getattr(self.config.model, 'form_last_n', last_n))
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
            if match['home_team'] == team:
                scored = match['home_score']
                conceded = match['away_score']
            else:
                scored = match['away_score']
                conceded = match['home_score']

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
            f'{prefix}_form_points': points,
            f'{prefix}_form_points_per_game': points / num_matches if num_matches > 0 else 0,
            f'{prefix}_form_wins': wins,
            f'{prefix}_form_draws': draws,
            f'{prefix}_form_losses': losses,
            f'{prefix}_form_goals_scored': goals_scored,
            f'{prefix}_form_goals_conceded': goals_conceded,
            f'{prefix}_form_goal_diff': goals_scored - goals_conceded,
            f'{prefix}_form_avg_goals_scored': goals_scored / num_matches if num_matches > 0 else 0,
            f'{prefix}_form_avg_goals_conceded': goals_conceded / num_matches if num_matches > 0 else 0,
        }

    def _get_momentum_features(self, team: str, history: List[Dict],
                               prefix: str, decay: float = 0.9) -> Dict:
        """Calculate exponentially weighted momentum features."""
        # Allow config override for decay
        try:
            cfg_decay = float(getattr(self.config.model, 'momentum_decay', decay))
            if 0.0 < cfg_decay < 1.0:
                decay = cfg_decay
        except Exception:
            pass
        if not history:
            return {
                f'{prefix}_momentum_points': 0,
                f'{prefix}_momentum_goals': 0,
                f'{prefix}_momentum_conceded': 0,
                f'{prefix}_momentum_score': 0,
            }

        recent = history[-10:] if len(history) >= 10 else history

        weighted_points = 0
        weighted_goals = 0
        weighted_conceded = 0
        total_weight = 0

        for i, match in enumerate(recent):
            weight = decay ** (len(recent) - 1 - i)
            total_weight += weight

            if match['home_team'] == team:
                scored = match['home_score']
                conceded = match['away_score']
            else:
                scored = match['away_score']
                conceded = match['home_score']

            weighted_goals += scored * weight
            weighted_conceded += conceded * weight

            if scored > conceded:
                weighted_points += 3 * weight
            elif scored == conceded:
                weighted_points += 1 * weight

        if total_weight > 0:
            avg_momentum_points = weighted_points / total_weight
            avg_momentum_goals = weighted_goals / total_weight
            avg_momentum_conceded = weighted_conceded / total_weight
        else:
            avg_momentum_points = 0
            avg_momentum_goals = 0
            avg_momentum_conceded = 0

        momentum_score = (avg_momentum_points +
                         avg_momentum_goals * 0.5 -
                         avg_momentum_conceded * 0.3)

        return {
            f'{prefix}_momentum_points': avg_momentum_points,
            f'{prefix}_momentum_goals': avg_momentum_goals,
            f'{prefix}_momentum_conceded': avg_momentum_conceded,
            f'{prefix}_momentum_score': momentum_score,
        }

    def _get_h2h_features(self, home_team: str, away_team: str,
                         historical_matches: List[Dict]) -> Dict:
        """Calculate head-to-head statistics."""
        h2h_matches = [m for m in historical_matches if
                      ((m['home_team'] == home_team and m['away_team'] == away_team) or
                       (m['home_team'] == away_team and m['away_team'] == home_team)) and
                      m.get('is_finished') and m.get('home_score') is not None and m.get('away_score') is not None]

        if not h2h_matches:
            return {
                'h2h_matches': 0,
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_home_goals': 0,
                'h2h_away_goals': 0,
                'h2h_home_win_rate': 0,
                'h2h_draw_rate': 0.0,
            }

        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals = 0
        away_goals = 0

        recent_h2h = h2h_matches[-10:]
        for match in recent_h2h:
            if match['home_team'] == home_team:
                home_goals += match['home_score']
                away_goals += match['away_score']
                if match['home_score'] > match['away_score']:
                    home_wins += 1
                elif match['home_score'] < match['away_score']:
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals += match['away_score']
                away_goals += match['home_score']
                if match['away_score'] > match['home_score']:
                    home_wins += 1
                elif match['away_score'] < match['home_score']:
                    away_wins += 1
                else:
                    draws += 1

        num_h2h = len(recent_h2h)

        return {
            'h2h_matches': num_h2h,
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_home_goals': home_goals,
            'h2h_away_goals': away_goals,
            'h2h_home_win_rate': home_wins / num_h2h if num_h2h > 0 else 0,
            'h2h_draw_rate': draws / num_h2h if num_h2h > 0 else 0.0,
        }

    def _get_home_away_features(self, team: str, history: List[Dict],
                               venue: str) -> Dict:
        """Calculate home/away specific statistics."""
        if venue == 'home':
            relevant = [m for m in history if m['home_team'] == team]
        else:
            relevant = [m for m in history if m['away_team'] == team]

        if not relevant:
            return {
                f'{venue}_venue_matches': 0,
                f'{venue}_venue_points_per_game': 0,
                f'{venue}_venue_goals_per_game': 0,
                f'{venue}_venue_conceded_per_game': 0,
            }

        points = 0
        goals = 0
        conceded = 0

        for match in relevant[-10:]:
            if venue == 'home':
                goals += match['home_score']
                conceded += match['away_score']
                if match['home_score'] > match['away_score']:
                    points += 3
                elif match['home_score'] == match['away_score']:
                    points += 1
            else:
                goals += match['away_score']
                conceded += match['home_score']
                if match['away_score'] > match['home_score']:
                    points += 3
                elif match['away_score'] == match['home_score']:
                    points += 1

        num_matches = len(relevant[-10:])

        return {
            f'{venue}_venue_matches': num_matches,
            f'{venue}_venue_points_per_game': points / num_matches if num_matches > 0 else 0,
            f'{venue}_venue_goals_per_game': goals / num_matches if num_matches > 0 else 0,
            f'{venue}_venue_conceded_per_game': conceded / num_matches if num_matches > 0 else 0,
        }

    def _get_goal_stats(self, team: str, history: List[Dict], prefix: str) -> Dict:
        """Calculate goal scoring statistics."""
        if not history:
            return {
                f'{prefix}_total_goals': 0,
                f'{prefix}_total_conceded': 0,
                f'{prefix}_avg_goals': 0,
                f'{prefix}_avg_conceded': 0,
            }

        total_goals = 0
        total_conceded = 0

        for match in history:
            if match['home_team'] == team:
                total_goals += match['home_score']
                total_conceded += match['away_score']
            else:
                total_goals += match['away_score']
                total_conceded += match['home_score']

        num_matches = len(history)

        return {
            f'{prefix}_total_goals': total_goals,
            f'{prefix}_total_conceded': total_conceded,
            f'{prefix}_avg_goals': total_goals / num_matches if num_matches > 0 else 0,
            f'{prefix}_avg_conceded': total_conceded / num_matches if num_matches > 0 else 0,
        }

    def _get_strength_of_schedule(self, team: str, history: List[Dict],
                                  all_matches: List[Dict], prefix: str) -> Dict:
        """Calculate strength of schedule based on opponents' positions."""
        if not history:
            return {
                f'{prefix}_avg_opponent_position': 10.0,
                f'{prefix}_avg_opponent_points': 0,
                f'{prefix}_sos_difficulty': 0,
            }

        table = self._calculate_table(all_matches)

        opponent_positions = []
        opponent_points = []

        recent = history[-5:] if len(history) >= 5 else history

        for match in recent:
            if match['home_team'] == team:
                opponent = match['away_team']
            else:
                opponent = match['home_team']

            opp_data = table.get(opponent, {'position': 10, 'points': 0})
            opponent_positions.append(opp_data['position'])
            opponent_points.append(opp_data['points'])

        if opponent_positions:
            avg_opp_pos = np.mean(opponent_positions)
            avg_opp_pts = np.mean(opponent_points)
            sos_difficulty = 1 - (avg_opp_pos / 20)
        else:
            avg_opp_pos = 10.0
            avg_opp_pts = 0
            sos_difficulty = 0.5

        return {
            f'{prefix}_avg_opponent_position': avg_opp_pos,
            f'{prefix}_avg_opponent_points': avg_opp_pts,
            f'{prefix}_sos_difficulty': sos_difficulty,
        }

    def _get_rest_features(self, home_team: str, away_team: str,
                          current_match: Dict, historical_matches: List[Dict]) -> Dict:
        """Calculate rest days since last match for both teams."""
        current_date = current_match['date']

        def get_last_match_date(team: str) -> datetime:
            team_matches = [m for m in historical_matches
                          if (m['home_team'] == team or m['away_team'] == team)
                          and m['is_finished']
                          and m['date'] < current_date]

            if team_matches:
                return max(m['date'] for m in team_matches)
            return current_date - timedelta(days=14)

        home_last_match = get_last_match_date(home_team)
        away_last_match = get_last_match_date(away_team)

        home_rest_days = min((current_date - home_last_match).days, 14)
        away_rest_days = min((current_date - away_last_match).days, 14)

        rest_advantage = home_rest_days - away_rest_days

        return {
            'home_rest_days': home_rest_days,
            'away_rest_days': away_rest_days,
            'rest_advantage': rest_advantage,
            'home_well_rested': 1 if home_rest_days >= 6 else 0,
            'away_well_rested': 1 if away_rest_days >= 6 else 0,
            'home_fatigued': 1 if home_rest_days <= 3 else 0,
            'away_fatigued': 1 if away_rest_days <= 3 else 0,
        }

    def _get_table_features(self, home_team: str, away_team: str,
                           historical_matches: List[Dict]) -> Dict:
        """Calculate current league position and points."""
        table = self._calculate_table(historical_matches)

        home_data = table.get(home_team, {'points': 0, 'position': 20, 'played': 0})
        away_data = table.get(away_team, {'points': 0, 'position': 20, 'played': 0})

        return {
            'home_table_position': home_data['position'],
            'away_table_position': away_data['position'],
            'home_table_points': home_data['points'],
            'away_table_points': away_data['points'],
            'table_position_diff': home_data['position'] - away_data['position'],
            'table_points_diff': home_data['points'] - away_data['points'],
            'abs_table_position_diff': abs(home_data['position'] - away_data['position']),
            'abs_table_points_diff': abs(home_data['points'] - away_data['points']),
        }

    def _calculate_table(self, matches: List[Dict]) -> Dict:
        """Calculate league table from matches."""
        table = defaultdict(lambda: {
            'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0
        })

        for match in matches:
            if not match['is_finished']:
                continue

            home = match['home_team']
            away = match['away_team']
            home_score = match['home_score']
            away_score = match['away_score']

            table[home]['played'] += 1
            table[away]['played'] += 1
            table[home]['goals_for'] += home_score
            table[home]['goals_against'] += away_score
            table[away]['goals_for'] += away_score
            table[away]['goals_against'] += home_score

            if home_score > away_score:
                table[home]['won'] += 1
                table[home]['points'] += 3
                table[away]['lost'] += 1
            elif home_score < away_score:
                table[away]['won'] += 1
                table[away]['points'] += 3
                table[home]['lost'] += 1
            else:
                table[home]['drawn'] += 1
                table[away]['drawn'] += 1
                table[home]['points'] += 1
                table[away]['points'] += 1

        # Add positions
        sorted_teams = sorted(table.items(),
                            key=lambda x: (x[1]['points'],
                                         x[1]['goals_for'] - x[1]['goals_against'],
                                         x[1]['goals_for']),
                            reverse=True)

        for position, (team, data) in enumerate(sorted_teams, 1):
            table[team]['position'] = position

        return dict(table)
