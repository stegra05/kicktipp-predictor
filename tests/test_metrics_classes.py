import numpy as np
import pandas as pd

from kicktipp_predictor.metrics import (
    LABELS_ORDER,
    ProbabilityMetrics,
    CalibrationMetrics,
    ConfusionMetrics,
    ConfidenceAnalysis,
    KicktippScoring,
)


def test_probability_metrics_perfect_predictions():
    y_true = ["H", "D", "A"]
    proba = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.isclose(ProbabilityMetrics.brier_score_multiclass(y_true, proba), 0.0)
    assert np.isclose(ProbabilityMetrics.log_loss_multiclass(y_true, proba), 0.0)
    assert np.isclose(ProbabilityMetrics.ranked_probability_score_3c(y_true, proba), 0.0)


def test_calibration_metrics_ece_structure_and_range():
    y_true = ["H", "H", "A", "D", "A", "D", "H", "A", "D", "H"]
    # Slightly confident towards H
    proba = np.tile(np.array([0.5, 0.25, 0.25]), (len(y_true), 1))
    ece = CalibrationMetrics.expected_calibration_error(y_true, proba, n_bins=5)
    assert set(ece.keys()) == set(LABELS_ORDER)
    for v in ece.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_confusion_metrics_perfect_accuracy():
    y_true = ["H", "D", "A", "H", "D", "A"]
    proba = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    stats = ConfusionMetrics.confusion_matrix_stats(y_true, proba)
    assert np.isclose(stats["accuracy"], 1.0)
    mat = np.array(stats["matrix"])  # 3x3
    assert mat.shape == (3, 3)
    assert (np.diag(mat) == np.array([2, 2, 2])).all()


def test_confidence_analysis_bin_by_confidence_shape_and_columns():
    conf = np.array([0.1, 0.4, 0.6, 0.9])
    y_true = ["H", "D", "A", "H"]
    proba = np.array([
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
        [0.5, 0.3, 0.2],
    ])
    points = np.array([2, 2, 4, 0])
    df = ConfidenceAnalysis.bin_by_confidence(
        conf, y_true, proba, points, n_bins=2
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert set(["bin", "count", "avg_points", "accuracy", "avg_confidence"]) <= set(df.columns)
    assert df["count"].sum() == len(conf)


def test_kicktipp_scoring_points():
    pred_home = [1, 2, 0]
    pred_away = [0, 1, 0]
    act_home = [1, 2, 1]
    act_away = [0, 1, 1]
    pts = KicktippScoring.compute_points(pred_home, pred_away, act_home, act_away)
    assert pts.tolist() == [4, 4, 3]