import pandas as pd


def test_prepare_features_adds_missing_columns():
    from kicktipp_predictor.predictor import MatchPredictor

    mp = MatchPredictor(quiet=True)
    # Simulate trained feature schema
    mp.feature_columns = ["feat_a", "feat_b"]

    df = pd.DataFrame({"feat_a": [0.5], "other": [1.0]})
    prepared = mp._prepare_features(df.copy())

    # Both required features exist and are ordered
    assert list(prepared.columns) == ["feat_a", "feat_b"]
    # Missing feature filled with 0.0
    assert prepared["feat_b"].iloc[0] == 0.0