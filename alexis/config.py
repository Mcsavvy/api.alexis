"""Dynaconf configuration file."""

import os

from dotenv import load_dotenv
from dynaconf import Dynaconf, Validator  # type: ignore[import-untyped]

load_dotenv()
os.environ.setdefault("ALEXIS_ENV", "production")

APP_ESSENTIALS = Validator(
    "SQLALCHEMY_DATABASE_URI",
    "REDIS_URL",
    "OPENAI_API_KEY",
    "SECRET_KEY",
    must_exist=True,
)


def export_openai_key(settings: Dynaconf):
    """Export the OpenAI key."""
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key


settings = Dynaconf(
    envvar_prefix="ALEXIS",
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,
    env_switcher="ALEXIS_ENV",
    post_hooks=[export_openai_key],
    validators=[APP_ESSENTIALS],
)
