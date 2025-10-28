from __future__ import annotations

from flask import Flask

from .config import AppConfig, load_config_from_env
from .extensions import configure_cors, configure_logging
from .routes.health import health_bp
from .blueprints.api import api_bp


def create_app(config: AppConfig | None = None) -> Flask:
    """Creates and configures the Flask application.

    This function initializes the Flask app, loads the configuration, sets up
    extensions like CORS and logging, and registers all necessary blueprints.

    Args:
        config: An optional application configuration object. If not provided,
            the configuration is loaded from environment variables.

    Returns:
        The configured Flask application instance.
    """
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Configuration
    app_config = config or load_config_from_env()
    # Ensure lowercase keys like 'cors_origins' are preserved
    app.config.update(app_config.as_dict())

    # Extensions
    configure_logging(app)
    configure_cors(app)

    # Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


