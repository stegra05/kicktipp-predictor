def test_imports():
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import MatchPredictor
    from kicktipp_predictor.web.app import app

    assert DataLoader is not None
    assert MatchPredictor is not None
    assert app is not None


def test_cli_help():
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-m", "kicktipp_predictor", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Kicktipp Predictor CLI" in res.stdout
