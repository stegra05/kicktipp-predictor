import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os
import pickle


class DataFetcher:
    """Fetches 3. Liga match data from OpenLigaDB API and web scraping as fallback."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.openligadb_base = "https://api.openligadb.de"
        self.league_code = "bl3"  # 3. Liga
        os.makedirs(cache_dir, exist_ok=True)

    def get_current_season(self) -> int:
        """Get current season year (e.g., 2024 for 2024/2025 season)."""
        now = datetime.now()
        # Season typically starts in July/August
        return now.year if now.month >= 7 else now.year - 1

    def fetch_season_matches(self, season: Optional[int] = None) -> List[Dict]:
        """
        Fetch all matches for a given season.

        Args:
            season: Season year (e.g., 2024 for 2024/2025). Defaults to current season.

        Returns:
            List of match dictionaries
        """
        if season is None:
            season = self.get_current_season()

        cache_file = os.path.join(self.cache_dir, f"matches_{season}.pkl")

        # Check cache (valid for 1 hour)
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 3600:  # 1 hour
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        url = f"{self.openligadb_base}/getmatchdata/{self.league_code}/{season}"

        try:
            response = requests.get(url, timeout=10)
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
        """
        Fetch matches for a specific matchday.

        Args:
            matchday: Matchday number (1-38 for 3. Liga)
            season: Season year. Defaults to current season.

        Returns:
            List of match dictionaries
        """
        if season is None:
            season = self.get_current_season()

        url = f"{self.openligadb_base}/getmatchdata/{self.league_code}/{season}/{matchday}"

        try:
            response = requests.get(url, timeout=10)
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
        """
        Get matches in the next N days.

        Args:
            days: Number of days to look ahead
            season: Season year. Defaults to current season.

        Returns:
            List of upcoming match dictionaries
        """
        matches = self.fetch_season_matches(season)

        now = datetime.now()
        future_date = now + timedelta(days=days)

        upcoming = [
            m for m in matches
            if now <= m['date'] <= future_date and not m['is_finished']
        ]

        return sorted(upcoming, key=lambda x: x['date'])

    def get_team_history(self, team_name: str, season: Optional[int] = None,
                         last_n_matches: int = 10) -> List[Dict]:
        """
        Get recent match history for a team.

        Args:
            team_name: Team name
            season: Season year. Defaults to current season.
            last_n_matches: Number of recent matches to return

        Returns:
            List of recent match dictionaries
        """
        matches = self.fetch_season_matches(season)

        team_matches = [
            m for m in matches
            if (m['home_team'] == team_name or m['away_team'] == team_name)
            and m['is_finished']
        ]

        return sorted(team_matches, key=lambda x: x['date'], reverse=True)[:last_n_matches]

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

    def fetch_historical_seasons(self, start_season: int, end_season: int) -> List[Dict]:
        """
        Fetch multiple seasons of historical data for training.

        Args:
            start_season: First season year (e.g., 2019)
            end_season: Last season year (e.g., 2024)

        Returns:
            List of all matches from all seasons
        """
        all_matches = []

        for season in range(start_season, end_season + 1):
            print(f"Fetching season {season}/{season+1}...")
            matches = self.fetch_season_matches(season)
            all_matches.extend(matches)
            time.sleep(0.5)  # Be nice to the API

        return all_matches
