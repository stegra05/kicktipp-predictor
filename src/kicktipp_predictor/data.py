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
        """Create feature dataset from match data.

        Args:
            matches: List of match dictionaries.

        Returns:
            DataFrame with features and target variables.
        """
        features_list = []

        # Sort matches by date
        sorted_matches = sorted(matches, key=lambda x: x['date'])

        for i, match in enumerate(sorted_matches):
            if (not match['is_finished'] or
                match.get('home_score') is None or
                match.get('away_score') is None):
                continue

            # Get historical data up to this match (for training)
            historical_matches = sorted_matches[:i]

            features = self._create_match_features(match, historical_matches)

            if features is not None:
                features_list.append(features)

        return pd.DataFrame(features_list)

    def create_prediction_features(self, upcoming_matches: List[Dict],
                                   historical_matches: List[Dict]) -> pd.DataFrame:
        """Create features for upcoming matches to predict.

        Args:
            upcoming_matches: List of upcoming match dictionaries.
            historical_matches: List of all historical matches for context.

        Returns:
            DataFrame with features (no target variables).
        """
        features_list = []

        for match in upcoming_matches:
            features = self._create_match_features(match, historical_matches, is_prediction=True)

            if features is not None:
                features_list.append(features)

        return pd.DataFrame(features_list)

    def _create_match_features(self, match: Dict, historical_matches: List[Dict],
                              is_prediction: bool = False) -> Optional[Dict]:
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
            'home_team': home_team,
            'away_team': away_team,
        }

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

    def _get_form_features(self, team: str, history: List[Dict], prefix: str,
                          last_n: int = 5) -> Dict:
        """Calculate team form over last N matches."""
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
