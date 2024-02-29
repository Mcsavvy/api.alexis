"""Dynaconf configuration file."""

import os

from dotenv import load_dotenv
from dynaconf import Dynaconf  # type: ignore[import-untyped]

load_dotenv()


def export_openai_key(settings: Dynaconf):
    """Export the OpenAI key."""
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key


settings = Dynaconf(
    envvar_prefix="ALEXIS",
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,
    load_dotenv=True,
    env_switcher="ALEXIS_ENV",
    post_hooks=[export_openai_key],
)
