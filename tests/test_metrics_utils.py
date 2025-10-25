import numpy as np

from kicktipp_predictor.metrics import LABELS_ORDER, MetricsUtils


def test_label_mapping_order():
    mapping = MetricsUtils.label_mapping()
    assert mapping == {"H": 0, "D": 1, "A": 2}


def test_labels_to_indices_with_unknown():
    labels = ["H", "D", "A", "X"]
    idx = MetricsUtils.labels_to_indices(labels)
    assert idx.tolist() == [0, 1, 2, -1]


def test_normalize_proba_row_sums_and_clipping():
    proba = np.array([
        [0.2, 0.3, 0.5],
        [0.0, 0.0, 0.0],
        [-1.0, 2.0, 0.0],
    ])
    norm = MetricsUtils.normalize_proba(proba)
    # Row sums should be 1
    assert np.allclose(norm.sum(axis=1), 1.0)
    # Values should be within [0, 1]
    assert np.all(norm >= 0.0)
    assert np.all(norm <= 1.0)