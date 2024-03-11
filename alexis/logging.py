"""Logging module."""

import logging
from logging import Logger, _nameToLevel
from logging.config import dictConfig

from alexis.config import settings as config
from alexis.utils import cast_fn

# debug settings
debug_mode = config.get("DEBUG", False)


dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(message)s",
            },
            "verbose": {
                "format": "[%(asctime)s] %(levelname)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": config.LOG_LEVEL,
                "class": config.LOG_HANDLER_CLASS,
                "formatter": "default",
                "show_time": config.LOG_SHOW_TIME,
                "rich_tracebacks": config.LOG_RICH_TRACEBACKS,
                "tracebacks_show_locals": config.LOG_TRACEBACKS_SHOW_LOCALS,
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "formatter": "verbose",
                "filename": "alexis.log",
                "mode": "w",
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
        "loggers": {
            "alexis": {
                "level": "DEBUG",
                "handlers": config.get("LOG_HANDLERS", ["console"]),
                "propagate": False,
            },
        },
    }
)

_logger: Logger | None = None


def get_logger() -> "Logger":
    """Get logger."""
    global _logger

    if _logger:
        return _logger
    logger = logging.getLogger("alexis")
    _logger = logger
    return logger


@cast_fn(logging.log)
def log(level: str, msg: str, *args, **kwargs):
    """Log."""
    lvl = _nameToLevel[level.upper()]
    kwargs.setdefault("stacklevel", 2)
    get_logger().log(lvl, msg, *args, **kwargs)


@cast_fn(logging.debug)
def debug(msg: str, *args, **kwargs):
    """Debug."""
    kwargs.setdefault("stacklevel", 2)
    get_logger().debug(msg, *args, **kwargs)


@cast_fn(logging.info)
def info(msg, *args, **kwargs):
    """Info."""
    kwargs.setdefault("stacklevel", 2)
    get_logger().info(msg, *args, **kwargs)


@cast_fn(logging.warning)
def warning(msg, *args, **kwargs):
    """Warning"""
    kwargs.setdefault("stacklevel", 2)
    get_logger().warning(msg, *args, **kwargs)


@cast_fn(logging.error)
def error(msg, *args, **kwargs):
    """Error"""
    kwargs.setdefault("stacklevel", 2)
    get_logger().error(msg, *args, **kwargs)


@cast_fn(logging.critical)
def critical(msg, *args, **kwargs):
    """Critical."""
    kwargs.setdefault("stacklevel", 2)
    get_logger().critical(msg, *args, **kwargs)


@cast_fn(logging.exception)
def exception(msg, *args, exc_info=True, **kwargs):
    """Exception."""
    kwargs.setdefault("stacklevel", 2)
    get_logger().exception(msg, *args, exc_info=exc_info, **kwargs)
