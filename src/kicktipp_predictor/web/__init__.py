from __future__ import annotations

from flask import Flask

from .config import AppConfig, load_config_from_env
from .extensions import configure_cors, configure_logging
from .routes.health import health_bp
from .blueprints.api import api_bp


def create_app(config: AppConfig | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Configuration
    app_config = config or load_config_from_env()
    app.config.from_mapping(app_config.as_dict())

    # Extensions
    configure_logging(app)
    configure_cors(app)

    # Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


