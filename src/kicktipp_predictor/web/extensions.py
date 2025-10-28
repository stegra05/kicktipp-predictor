from __future__ import annotations

from flask import Flask

try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional dependency
    CORS = None  # type: ignore


def configure_logging(app: Flask) -> None:
    # Use Flask/Werkzeug default logging; allow gunicorn/uvicorn to override
    if not app.debug:
        # Avoid duplicate handlers in reloader
        pass


def configure_cors(app: Flask) -> None:
    origins = app.config.get("cors_origins")
    if origins and CORS is not None:
        allowed = [o.strip() for o in str(origins).split(",") if o.strip()]
        CORS(app, resources={r"/api/*": {"origins": allowed}})


