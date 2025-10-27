import numpy as np
import pandas as pd

from kicktipp_predictor.predictor import CascadedPredictor


def test_prepare_targets_valid_mixed():
    df = pd.DataFrame(
        {
            "result": ["H", "D", "A", "H", "A", "D"],
            "feat1": [1, 2, 3, 4, 5, 6],
        }
    )
    p = CascadedPredictor()
    df2, y_draw, y_win = p._prepare_targets(df)
    # y_draw aligns with all rows
    assert list(y_draw.values) == [0, 1, 0, 0, 0, 1]
    # y_win defined only for non-draw rows and uses 'H'/'A'
    assert list(y_win) == ["H", "A", "H", "A"]
    # is_home_win present for non-draw rows
    assert set(df2.columns) >= {"is_draw", "is_home_win"}


def test_prepare_targets_invalid_type():
    p = CascadedPredictor()
    try:
        p._prepare_targets([])  # type: ignore[arg-type]
        assert False, "Expected ValueError for non-DataFrame input"
    except ValueError:
        pass


def test_prepare_targets_missing_result():
    df = pd.DataFrame({"feat1": [1, 2, 3]})
    p = CascadedPredictor()
    try:
        p._prepare_targets(df)
        assert False, "Expected ValueError for missing 'result' column"
    except ValueError:
        pass


def test_prepare_targets_invalid_result_values():
    df = pd.DataFrame({"result": ["X", "Y", "Z"]})
    p = CascadedPredictor()
    try:
        p._prepare_targets(df)
        assert False, "Expected ValueError for invalid 'result' values"
    except ValueError:
        pass


def test_prepare_targets_all_draws_error_for_win_stage():
    df = pd.DataFrame({"result": ["D", "D", "D"]})
    p = CascadedPredictor()
    try:
        p._prepare_targets(df)
        assert False, "Expected ValueError when no non-draw samples exist"
    except ValueError:
        pass