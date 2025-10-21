import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class FeatureEngineer:
    """Creates features for football match prediction."""

    def __init__(self):
        self.team_stats = {}

    def create_features_from_matches(self, matches: List[Dict]) -> pd.DataFrame:
        """
        Create feature dataset from match data.

        Args:
            matches: List of match dictionaries

        Returns:
            DataFrame with features and target variables
        """
        features_list = []

        # Sort matches by date
        sorted_matches = sorted(matches, key=lambda x: x['date'])

        for i, match in enumerate(sorted_matches):
            if not match['is_finished']:
                continue

            # Get historical data up to this match (for training)
            historical_matches = sorted_matches[:i]

            features = self._create_match_features(match, historical_matches)

            if features is not None:
                features_list.append(features)

        return pd.DataFrame(features_list)

    def create_prediction_features(self, upcoming_matches: List[Dict],
                                   historical_matches: List[Dict]) -> pd.DataFrame:
        """
        Create features for upcoming matches to predict.

        Args:
            upcoming_matches: List of upcoming match dictionaries
            historical_matches: List of all historical matches for context

        Returns:
            DataFrame with features (no target variables)
        """
        features_list = []

        for match in upcoming_matches:
            features = self._create_match_features(match, historical_matches, is_prediction=True)

            if features is not None:
                features_list.append(features)

        return pd.DataFrame(features_list)

    def _create_match_features(self, match: Dict, historical_matches: List[Dict],
                              is_prediction: bool = False) -> Dict:
        """Create features for a single match."""

        home_team = match['home_team']
        away_team = match['away_team']

        # Restrict to finished matches strictly before current match date
        ref_hist = [m for m in historical_matches
                    if m.get('is_finished') and m.get('date') is not None and m['date'] < match['date']]

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
            # Log reason for debugging early-season feature starvation
            try:
                print(f"[FeatureEngineer] Skipping {home_team} vs {away_team} MD{md}: history home={len(home_history)} away={len(away_history)} (min={min_hist})")
            except Exception:
                pass
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

        # Momentum features (exponentially weighted recent form)
        features.update(self._get_momentum_features(home_team, home_history, prefix='home'))
        features.update(self._get_momentum_features(away_team, away_history, prefix='away'))

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
            features['home_score'] = match['home_score']
            features['away_score'] = match['away_score']
            features['goal_difference'] = match['home_score'] - match['away_score']

            if match['home_score'] > match['away_score']:
                features['result'] = 'H'
            elif match['home_score'] < match['away_score']:
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

    def _get_h2h_features(self, home_team: str, away_team: str,
                         historical_matches: List[Dict]) -> Dict:
        """Calculate head-to-head statistics."""
        # Consider only finished H2H matches with valid scores
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
            }

        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals = 0
        away_goals = 0

        # Use only the last 10 valid H2H matches
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

        for match in relevant[-10:]:  # Last 10 home/away matches
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

    def _get_momentum_features(self, team: str, history: List[Dict],
                               prefix: str, decay: float = 0.9) -> Dict:
        """
        Calculate exponentially weighted momentum features.
        Recent matches have more weight than older ones.

        Args:
            team: Team name
            history: Team match history
            prefix: Feature name prefix
            decay: Decay factor (0-1), higher = more weight on recent

        Returns:
            Dictionary with momentum features
        """
        if not history:
            return {
                f'{prefix}_momentum_points': 0,
                f'{prefix}_momentum_goals': 0,
                f'{prefix}_momentum_conceded': 0,
                f'{prefix}_momentum_score': 0,
            }

        # Use last 10 matches for momentum
        recent = history[-10:] if len(history) >= 10 else history

        weighted_points = 0
        weighted_goals = 0
        weighted_conceded = 0
        total_weight = 0

        # Process matches from oldest to newest, giving more weight to recent
        for i, match in enumerate(recent):
            # Weight increases exponentially for more recent matches
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

        # Momentum score: weighted combination of recent performance
        momentum_score = (avg_momentum_points +
                         avg_momentum_goals * 0.5 -
                         avg_momentum_conceded * 0.3)

        return {
            f'{prefix}_momentum_points': avg_momentum_points,
            f'{prefix}_momentum_goals': avg_momentum_goals,
            f'{prefix}_momentum_conceded': avg_momentum_conceded,
            f'{prefix}_momentum_score': momentum_score,
        }

    def _get_strength_of_schedule(self, team: str, history: List[Dict],
                                  all_matches: List[Dict], prefix: str) -> Dict:
        """
        Calculate strength of schedule based on opponents' league positions.

        Args:
            team: Team name
            history: Team match history
            all_matches: All matches for calculating table
            prefix: Feature name prefix

        Returns:
            Dictionary with strength of schedule features
        """
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
            # Lower position = stronger opponent = harder schedule
            # Normalize: 1 = hardest (position 1), 20 = easiest (position 20)
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
        """
        Calculate rest days since last match for both teams.

        Args:
            home_team: Home team name
            away_team: Away team name
            current_match: Current match dictionary
            historical_matches: All historical matches

        Returns:
            Dictionary with rest features
        """
        current_date = current_match['date']

        def get_last_match_date(team: str) -> datetime:
            team_matches = [m for m in historical_matches
                          if (m['home_team'] == team or m['away_team'] == team)
                          and m['is_finished']
                          and m['date'] < current_date]

            if team_matches:
                return max(m['date'] for m in team_matches)
            return current_date - timedelta(days=14)  # Default: 2 weeks

        home_last_match = get_last_match_date(home_team)
        away_last_match = get_last_match_date(away_team)

        home_rest_days = (current_date - home_last_match).days
        away_rest_days = (current_date - away_last_match).days

        # Cap at 14 days to avoid outliers
        home_rest_days = min(home_rest_days, 14)
        away_rest_days = min(away_rest_days, 14)

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



