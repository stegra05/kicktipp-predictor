import tempfile
from pathlib import Path
import warnings

from kicktipp_predictor.config import Config


def test_config_load_ignores_legacy_gd_keys(tmp_path: Path):
    yml = tmp_path / "legacy_params.yaml"
    yml.write_text(
        """
        gd_n_estimators: 1000
        gd_max_depth: 8
        draw_n_estimators: 123
        draw_learning_rate: 0.05
        win_n_estimators: 456
        win_learning_rate: 0.1
        """
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = Config.load(yml)

    # Legacy keys must be ignored; ModelConfig has no gd_* attrs
    assert not hasattr(cfg.model, "gd_n_estimators")
    assert cfg.model.draw_n_estimators == 123
    assert abs(cfg.model.draw_learning_rate - 0.05) < 1e-9
    assert cfg.model.win_n_estimators == 456
    assert abs(cfg.model.win_learning_rate - 0.1) < 1e-9
    # At least one warning for legacy keys should be emitted
    assert any("legacy" in str(x.message).lower() for x in w)