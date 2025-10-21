import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, List
import pandas as pd


class PoissonPredictor:
    """
    Statistical predictor using Poisson distribution for goal prediction.
    Based on team attack and defense strengths.
    """

    def __init__(self):
        self.team_attack = {}
        self.team_defense = {}
        self.home_advantage = 0.0
        self.league_avg_goals = 0.0
        # Dixon-Coles low-score interaction parameter
        self.rho = 0.0
        self.max_goals = 8

    def train(self, matches_df: pd.DataFrame):
        """
        Train the Poisson model on historical matches.

        Args:
            matches_df: DataFrame with historical match data and features
        """
        matches = matches_df[matches_df['home_score'].notna()].copy()

        if len(matches) == 0:
            return

        # Calculate league average goals
        total_goals = matches['home_score'].sum() + matches['away_score'].sum()
        total_matches = len(matches)
        self.league_avg_goals = total_goals / (2 * total_matches)

        # Calculate home advantage
        home_goals = matches['home_score'].sum()
        away_goals = matches['away_score'].sum()
        self.home_advantage = (home_goals / total_matches) / (away_goals / total_matches)
        # Smooth home advantage with a prior when data is sparse
        if total_matches < 150:
            prior = 1.10
            self.home_advantage = 0.7 * float(self.home_advantage) + 0.3 * prior

        # Calculate attack and defense strengths
        teams = set(matches['home_team'].unique()) | set(matches['away_team'].unique())

        # Initialize strengths
        for team in teams:
            self.team_attack[team] = 1.0
            self.team_defense[team] = 1.0

        # Iteratively update strengths (opponent-adjusted updates)
        for _ in range(8):
            new_attack = {}
            new_defense = {}

            for team in teams:
                home_matches = matches[matches['home_team'] == team]
                away_matches = matches[matches['away_team'] == team]

                # Attack: goals scored normalized by opponents' current defense
                gs_home = home_matches['home_score'].sum()
                gs_away = away_matches['away_score'].sum()
                denom_attack = 0.0
                for _, m in home_matches.iterrows():
                    denom_attack += self.team_defense.get(m['away_team'], 1.0)
                for _, m in away_matches.iterrows():
                    denom_attack += self.team_defense.get(m['home_team'], 1.0)
                if denom_attack > 0:
                    new_attack[team] = (gs_home + gs_away) / denom_attack / max(self.league_avg_goals, 1e-9)
                else:
                    new_attack[team] = 1.0

                # Defense: goals conceded normalized by opponents' current attack
                gc_home = home_matches['away_score'].sum()
                gc_away = away_matches['home_score'].sum()
                denom_def = 0.0
                for _, m in home_matches.iterrows():
                    denom_def += self.team_attack.get(m['away_team'], 1.0)
                for _, m in away_matches.iterrows():
                    denom_def += self.team_attack.get(m['home_team'], 1.0)
                if denom_def > 0:
                    new_defense[team] = (gc_home + gc_away) / denom_def / max(self.league_avg_goals, 1e-9)
                else:
                    new_defense[team] = 1.0

            self.team_attack = new_attack
            self.team_defense = new_defense

        # Estimate a simple Dixon–Coles rho with mild regularization
        self.rho = self._estimate_rho(matches)

    def _estimate_rho(self, matches: pd.DataFrame) -> float:
        """Estimate DC rho with a coarse line search over [-0.1, 0.1] with L2 penalty."""
        if len(matches) == 0:
            return 0.0

        # Precompute expected goals per match given current strengths
        lambdas = []
        for _, m in matches.iterrows():
            home_attack = self.team_attack.get(m['home_team'], 1.0)
            home_defense = self.team_defense.get(m['home_team'], 1.0)
            away_attack = self.team_attack.get(m['away_team'], 1.0)
            away_defense = self.team_defense.get(m['away_team'], 1.0)
            home_lambda = self.league_avg_goals * home_attack * away_defense * self.home_advantage
            away_lambda = self.league_avg_goals * away_attack * home_defense
            lambdas.append((home_lambda, away_lambda, int(m['home_score']), int(m['away_score'])))

        def dc_factor(hg: int, ag: int, rho_val: float) -> float:
            # Simple symmetric adjustment for low scores
            if hg == 0 and ag == 0:
                return 1.0 + rho_val
            if (hg == 0 and ag == 1) or (hg == 1 and ag == 0):
                return 1.0 - rho_val
            if hg == 1 and ag == 1:
                return 1.0 - rho_val
            return 1.0

        def log_likelihood(rho_val: float) -> float:
            ll = 0.0
            for hl, al, hg, ag in lambdas:
                # Base independent Poisson probability
                p = poisson.pmf(hg, max(hl, 1e-9)) * poisson.pmf(ag, max(al, 1e-9))
                p *= dc_factor(hg, ag, rho_val)
                p = max(p, 1e-15)
                ll += np.log(p)
            return ll

        # If few matches, default to 0 to avoid overfitting
        if len(matches) < 200:
            return 0.0

        best_rho = 0.0
        best_obj = -np.inf
        for rho_val in np.linspace(-0.1, 0.1, 11):
            ll = log_likelihood(rho_val)
            # L2 penalty to shrink toward 0
            obj = ll - 10.0 * (rho_val ** 2)
            if obj > best_obj:
                best_obj = obj
                best_rho = rho_val
        return float(best_rho)

    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """
        Predict match outcome using Poisson distribution.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with predictions
        """
        # Default to league average if team not in training data
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)

        # Calculate expected goals
        home_expected = self.league_avg_goals * home_attack * away_defense * self.home_advantage
        away_expected = self.league_avg_goals * away_attack * home_defense

        # Calculate probabilities for different scorelines
        score_probabilities = np.zeros((self.max_goals, self.max_goals))

        for home_goals in range(self.max_goals):
            for away_goals in range(self.max_goals):
                prob_home = poisson.pmf(home_goals, home_expected)
                prob_away = poisson.pmf(away_goals, away_expected)
                p = prob_home * prob_away
                # Apply simple DC correction on low-score cells
                if home_goals == 0 and away_goals == 0:
                    p *= (1.0 + self.rho)
                elif (home_goals == 0 and away_goals == 1) or (home_goals == 1 and away_goals == 0):
                    p *= (1.0 - self.rho)
                elif home_goals == 1 and away_goals == 1:
                    p *= (1.0 - self.rho)
                score_probabilities[home_goals, away_goals] = p

        # Add tiny diagonal floor to avoid pathological zeros on common low scores
        eps = 1e-12
        score_probabilities[0,0] = max(score_probabilities[0,0], eps)
        score_probabilities[1,1] = max(score_probabilities[1,1], eps)

        # Renormalize after DC correction
        total = np.sum(score_probabilities)
        if total > 0:
            score_probabilities /= total

        # Find most likely scoreline
        most_likely_idx = np.unravel_index(np.argmax(score_probabilities), score_probabilities.shape)
        predicted_home = most_likely_idx[0]
        predicted_away = most_likely_idx[1]

        # Calculate outcome probabilities
        home_win_prob = np.sum(np.tril(score_probabilities, -1))
        draw_prob = np.trace(score_probabilities)
        away_win_prob = np.sum(np.triu(score_probabilities, 1))

        return {
            'predicted_home_score': predicted_home,
            'predicted_away_score': predicted_away,
            'home_expected_goals': home_expected,
            'away_expected_goals': away_expected,
            'home_win_probability': home_win_prob,
            'draw_probability': draw_prob,
            'away_win_probability': away_win_prob,
            'score_probability': score_probabilities[predicted_home, predicted_away],
        }

    def predict_batch(self, matches: List[Tuple[str, str]]) -> List[Dict]:
        """
        Predict multiple matches.

        Args:
            matches: List of (home_team, away_team) tuples

        Returns:
            List of prediction dictionaries
        """
        return [self.predict_match(home, away) for home, away in matches]



