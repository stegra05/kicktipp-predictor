def test_imports():
    import kicktipp_predictor
    from kicktipp_predictor.core.scraper.data_fetcher import DataFetcher
    from kicktipp_predictor.core.features.feature_engineering import FeatureEngineer
    from kicktipp_predictor.models.hybrid_predictor import HybridPredictor
    from kicktipp_predictor.web.app import app

    assert DataFetcher is not None
    assert FeatureEngineer is not None
    assert HybridPredictor is not None
    assert app is not None


def test_cli_help():
    import subprocess, sys
    res = subprocess.run([sys.executable, "-m", "kicktipp_predictor", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Kicktipp Predictor CLI" in res.stdout


