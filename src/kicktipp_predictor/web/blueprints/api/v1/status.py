from __future__ import annotations

from flask import current_app, jsonify
from kicktipp_predictor.predictor import GoalDifferencePredictor

from . import v1_bp


@v1_bp.get("/status")
def api_status() -> tuple[dict[str, object], int]:
    """Returns the status of the API.

    This endpoint provides information about the API status, including whether
    the prediction model is loaded, the API version, and the current application
    configuration.

    Returns:
        A JSON response containing the API status information.
    """
    cfg = {
        "debug": bool(current_app.config.get("debug", False)),
        "testing": bool(current_app.config.get("testing", False)),
    }
    # Model loaded?
    model_loaded = False
    try:
        p = GoalDifferencePredictor()
        p.load_model()
        model_loaded = p.model is not None
    except Exception:
        model_loaded = False
    return jsonify({"api": "v1", "status": "ok", "model_loaded": model_loaded, "version": "4.0.0a2", "config": cfg}), 200


