import subprocess
import sys


def test_cli_tune_help_has_reset_flag():
    res = subprocess.run(
        [sys.executable, "-m", "kicktipp_predictor.cli", "tune", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--reset-storage" in res.stdout