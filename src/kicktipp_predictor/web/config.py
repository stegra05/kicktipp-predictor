from __future__ import annotations

from dataclasses import dataclass, asdict
import os


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the Flask web application.

    This class defines the configuration parameters for the Flask app,
    such as debug mode, testing mode, secret key, and CORS origins.

    Attributes:
        debug: Whether to run the app in debug mode.
        testing: Whether to run the app in testing mode.
        secret_key: The secret key for the app.
        cors_origins: A comma-separated list of allowed CORS origins.
    """
    debug: bool = False
    testing: bool = False
    secret_key: str = "change-this-secret"
    cors_origins: str | None = None  # comma-separated list

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config_from_env() -> AppConfig:
    """Loads the application configuration from environment variables.

    This function creates an `AppConfig` instance with values sourced from
    environment variables. It provides default values for missing variables.

    Returns:
        An `AppConfig` instance with the loaded configuration.
    """
    return AppConfig(
        debug=_str_to_bool(os.getenv("FLASK_DEBUG"), False),
        testing=_str_to_bool(os.getenv("FLASK_TESTING"), False),
        secret_key=os.getenv("FLASK_SECRET_KEY", "change-this-secret"),
        cors_origins=os.getenv("CORS_ORIGINS"),
    )


