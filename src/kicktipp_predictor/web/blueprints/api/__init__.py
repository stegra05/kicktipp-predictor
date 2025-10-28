"""Main API blueprint.

This blueprint registers all the versioned API blueprints.
"""
from __future__ import annotations

from flask import Blueprint


api_bp = Blueprint("api", __name__)

# Sub-blueprints and versioning can be registered here, e.g. v1
from .v1 import v1_bp  # noqa: E402

api_bp.register_blueprint(v1_bp, url_prefix="/v1")


