from __future__ import annotations

from flask import Blueprint


v1_bp = Blueprint("api_v1", __name__)

# Example: group endpoints by resource in separate modules
from . import status  # noqa: E402,F401


