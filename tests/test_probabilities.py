import numpy as np
import pandas as pd

from kicktipp_predictor.predictor import CascadedPredictor


class _StubModel:
    def __init__(self, probs):
        # probs: list of [p0, p1] for each sample
        self._probs = np.array(probs, dtype=float)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # Ignore X; return fixed probabilities
        n = len(X)
        assert n == self._probs.shape[0]
        return self._probs


def test_probability_combination_and_normalization():
    # Two samples with simple features
    X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [1.0, 2.0]})

    # Draw probabilities: sample1 p(D)=0.3, sample2 p(D)=0.1
    draw_probs = [[0.7, 0.3], [0.9, 0.1]]

    # Win probabilities (conditional): sample1 P(H|~D)=0.6, sample2 P(H|~D)=0.25
    win_probs = [[0.4, 0.6], [0.75, 0.25]]

    pred = CascadedPredictor()
    pred.feature_columns = ["f1", "f2"]
    pred.draw_label_encoder.fit([0, 1])
    pred.win_label_encoder.fit(["A", "H"])
    pred.draw_model = _StubModel(draw_probs)
    pred.win_model = _StubModel(win_probs)

    out = pred.predict(X)
    assert isinstance(out, list) and len(out) == 2

    # Expected combined probabilities
    # sample1: pD=0.3, pH|~D=0.6 -> pH=0.7*0.6=0.42, pA=0.7*0.4=0.28
    # sample2: pD=0.1, pH|~D=0.25 -> pH=0.9*0.25=0.225, pA=0.9*0.75=0.675
    expected = [
        (0.42, 0.3, 0.28),
        (0.225, 0.1, 0.675),
    ]

    for i, row in enumerate(out):
        ph = row["home_win_probability"]
        pd_ = row["draw_probability"]
        pa = row["away_win_probability"]
        s = ph + pd_ + pa
        # Check normalization and closeness to expected values
        assert abs(s - 1.0) < 1e-9
        assert abs(ph - expected[i][0]) < 1e-9
        assert abs(pd_ - expected[i][1]) < 1e-9
        assert abs(pa - expected[i][2]) < 1e-9

        # Outcome matches argmax
        probs = np.array([ph, pd_, pa])
        labels = ["H", "D", "A"]
        assert row["predicted_outcome"] == labels[int(np.argmax(probs))]

        # Scoreline heuristic exists and is integer
        assert isinstance(row["predicted_home_score"], int)
        assert isinstance(row["predicted_away_score"], int)