"""Dynaconf configuration file."""

from dynaconf import Dynaconf  # type: ignore[import-untyped]

settings = Dynaconf(
    envvar_prefix="ALEXIS",
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,
    load_dotenv=True,
    env_switcher="ALEXIS_ENV",
)
