from __future__ import annotations

from flask import Flask, request

try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional dependency
    CORS = None  # type: ignore


def configure_logging(app: Flask) -> None:
    """Configures logging for the Flask application.

    This function currently relies on the default logging configuration provided
    by Flask and Werkzeug. It can be extended to support more advanced logging
    setups, such as logging to a file or a remote service.

    Args:
        app: The Flask application instance.
    """
    # Use Flask/Werkzeug default logging; allow gunicorn/uvicorn to override
    if not app.debug:
        # Avoid duplicate handlers in reloader
        pass


def configure_cors(app: Flask) -> None:
    """Configures Cross-Origin Resource Sharing (CORS) for the application.

    This function sets up CORS to allow requests from specified origins to the
    API endpoints. The allowed origins are read from the application
    configuration.

    Args:
        app: The Flask application instance.
    """
    origins = app.config.get("cors_origins")
    # Development fallback: allow local frontends if not explicitly configured
    if not origins:
        if bool(app.config.get("debug", False)) or bool(app.config.get("testing", False)):
            allowed = ["http://localhost:3000", "http://127.0.0.1:3000"]
        else:
            return
    else:
        allowed = [o.strip() for o in str(origins).split(",") if o.strip()]
    if CORS is not None:
        CORS(
            app,
            resources={
                r"/api/*": {
                    "origins": allowed,
                    "methods": ["GET", "POST", "OPTIONS"],
                    "allow_headers": ["Content-Type"],
                }
            },
        )
    else:
        @app.after_request
        def _add_cors_headers(resp):  # type: ignore
            try:
                origin = request.headers.get("Origin")
                if origin and origin in allowed and request.path.startswith("/api/"):
                    resp.headers["Access-Control-Allow-Origin"] = origin
                    resp.headers["Vary"] = "Origin"
                    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            except Exception:
                pass
            return resp


