"""Legacy compatibility shim for FeatureEngineer.

Provides a thin wrapper over v2 DataLoader feature creation functions.
"""
from typing import List, Dict
import pandas as pd

from ...data import DataLoader


class FeatureEngineer:
    """Backwards-compatible feature engineering facade.

    Delegates to DataLoader's feature construction.
    """

    def __init__(self):
        self.loader = DataLoader()

    def create_features_from_matches(self, matches: List[Dict]) -> pd.DataFrame:
        return self.loader.create_features_from_matches(matches)

    def create_prediction_features(self, upcoming_matches: List[Dict], historical_matches: List[Dict]) -> pd.DataFrame:
        return self.loader.create_prediction_features(upcoming_matches, historical_matches)


