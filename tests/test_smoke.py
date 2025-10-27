def test_imports():
    from kicktipp_predictor.data import DataLoader
    from kicktipp_predictor.predictor import CascadedPredictor

    assert DataLoader is not None
    assert CascadedPredictor is not None


def test_cascaded_predictor_attrs():
    from kicktipp_predictor.predictor import CascadedPredictor

    p = CascadedPredictor()
    # Basic attributes and methods exist
    assert hasattr(p, "draw_model")
    assert hasattr(p, "win_model")
    assert hasattr(p, "train")
    assert hasattr(p, "predict")


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
