from __future__ import annotations

from flask import Blueprint, jsonify


health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health() -> tuple[dict[str, str], int]:
    """Health check endpoint.

    This endpoint can be used to verify that the application is running.

    Returns:
        A JSON response with a status of "ok".
    """
    return jsonify({"status": "ok"}), 200


