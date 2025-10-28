from __future__ import annotations

from flask import current_app, jsonify

from ... import v1_bp


@v1_bp.get("/status")
def api_status() -> tuple[dict[str, object], int]:
    cfg = {
        "debug": bool(current_app.config.get("debug", False)),
        "testing": bool(current_app.config.get("testing", False)),
    }
    return jsonify({"api": "v1", "status": "ok", "config": cfg}), 200


