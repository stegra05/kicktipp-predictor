import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from kicktipp_predictor.config import Config, PathConfig, ModelConfig
from kicktipp_predictor.predictor import CascadedPredictor
from sklearn.linear_model import LogisticRegression


def _make_test_config(models_dir: Path) -> Config:
    paths = PathConfig(models_dir=models_dir)
    model = ModelConfig(
        # Keep training light for tests
        draw_n_estimators=30,
        draw_max_depth=3,
        draw_learning_rate=0.1,
        win_n_estimators=30,
        win_max_depth=3,
        win_learning_rate=0.1,
        random_state=0,
        min_training_matches=10,
        n_jobs=1,
        val_fraction=0.2,
    )
    return Config(paths=paths, model=model)


def _make_synthetic_matches(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    results = rng.choice(["H", "D", "A"], size=n, p=[0.45, 0.25, 0.30])
    df = pd.DataFrame(
        {
            "match_id": np.arange(n),
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "feat1": rng.normal(0, 1, size=n),
            "feat2": rng.normal(1, 0.5, size=n),
            "result": results,
        }
    )
    return df


def test_save_and_load_v4_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        cfg = _make_test_config(models_dir)

        # Prepare minimal models without relying on XGBoost training
        cp = CascadedPredictor(config=cfg)
        df = _make_synthetic_matches(40)
        cp.feature_columns = ["feat1", "feat2"]
        X_all = df[["feat1", "feat2"]]
        # Targets
        y_draw = (df["result"] == "D").astype(int)
        non_draw_mask = y_draw == 0
        y_win = np.where(df.loc[non_draw_mask, "result"] == "H", "H", "A")

        # Encoders
        cp.draw_label_encoder.fit([0, 1])
        cp.win_label_encoder.fit(["A", "H"])

        # Fit simple logistic models
        lr_draw = LogisticRegression(max_iter=100)
        cp.draw_model = lr_draw.fit(X_all, cp.draw_label_encoder.transform(y_draw.tolist()))

        lr_win = LogisticRegression(max_iter=100)
        X_nd = X_all.loc[non_draw_mask]
        y_win_enc = cp.win_label_encoder.transform(list(y_win))
        cp.win_model = lr_win.fit(X_nd, y_win_enc)

        # Save to v4 layout
        cp.save_models()

        # Load into a fresh instance
        cp2 = CascadedPredictor(config=cfg)
        cp2.load_models()

        # Metadata integrity
        assert isinstance(cp2.feature_columns, list) and len(cp2.feature_columns) > 0
        assert hasattr(cp2.draw_label_encoder, "classes_")
        assert hasattr(cp2.win_label_encoder, "classes_")

        # Basic functionality: predict on a small batch
        preds = cp2.predict(df.sample(5, random_state=1))
        assert len(preds) == 5
        for p in preds:
            assert set(["home_win_probability", "draw_probability", "away_win_probability"]).issubset(p.keys())


def test_backward_compatibility_load_old_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        cfg = _make_test_config(models_dir)
        cp = CascadedPredictor(config=cfg)
        df = _make_synthetic_matches(30)
        cp.feature_columns = ["feat1", "feat2"]
        X_all = df[["feat1", "feat2"]]
        y_draw = (df["result"] == "D").astype(int)
        non_draw_mask = y_draw == 0
        y_win = np.where(df.loc[non_draw_mask, "result"] == "H", "H", "A")

        cp.draw_label_encoder.fit([0, 1])
        cp.win_label_encoder.fit(["A", "H"])

        lr_draw = LogisticRegression(max_iter=100)
        cp.draw_model = lr_draw.fit(X_all, cp.draw_label_encoder.transform(y_draw.tolist()))

        lr_win = LogisticRegression(max_iter=100)
        X_nd = X_all.loc[non_draw_mask]
        y_win_enc = cp.win_label_encoder.transform(list(y_win))
        cp.win_model = lr_win.fit(X_nd, y_win_enc)

        # Save v4 artifacts first
        cp.save_models()

        # Remove v4 metadata and create legacy artifacts
        v4_meta = models_dir / "metadata_v4.joblib"
        if v4_meta.exists():
            os.remove(v4_meta)

        joblib.dump({"draw": cp.draw_label_encoder, "win": cp.win_label_encoder}, models_dir / "encoders.joblib")
        joblib.dump({"feature_columns": list(cp.feature_columns)}, models_dir / "cascaded_metadata.joblib")

        # Load with new loader which should fallback to legacy files
        cp2 = CascadedPredictor(config=cfg)
        cp2.load_models()

        assert isinstance(cp2.feature_columns, list) and len(cp2.feature_columns) == len(cp.feature_columns)
        assert hasattr(cp2.draw_label_encoder, "classes_")
        assert hasattr(cp2.win_label_encoder, "classes_")


def test_load_models_missing_files_raises():
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        cfg = _make_test_config(models_dir)
        cp = CascadedPredictor(config=cfg)

        # No artifacts present
        try:
            cp.load_models()
            raise AssertionError("Expected FileNotFoundError for missing artifacts")
        except FileNotFoundError:
            pass