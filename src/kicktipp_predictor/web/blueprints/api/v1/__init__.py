from __future__ import annotations

from flask import Blueprint


v1_bp = Blueprint("api_v1", __name__)

# Example: group endpoints by resource in separate modules
from . import status  # noqa: E402,F401
from . import predictions  # noqa: E402,F401
from . import league  # noqa: E402,F401
from . import teams  # noqa: E402,F401
from . import matches  # noqa: E402,F401
from . import evaluation  # noqa: E402,F401
from . import model  # noqa: E402,F401
from . import admin  # noqa: E402,F401


