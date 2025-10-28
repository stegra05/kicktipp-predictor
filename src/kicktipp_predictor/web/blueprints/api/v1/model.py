from __future__ import annotations

import joblib
from flask import jsonify

from ... import v1_bp
from kicktipp_predictor.config import get_config


@v1_bp.get("/model/features")
def get_model_features():
    cfg = get_config()
    metadata_path = cfg.paths.gd_model_path.with_name("metadata.joblib")
    try:
        metadata = joblib.load(metadata_path)
        features = metadata.get("feature_columns", []) if isinstance(metadata, dict) else []
        return jsonify({
            "model_version": str(cfg.paths.gd_model_path.name),
            "feature_count": len(features),
            "features": features,
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Model metadata not found"}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to read model metadata: {exc}"}), 500


