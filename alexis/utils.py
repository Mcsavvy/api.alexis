"""Utility functions for the Alexis API."""

from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from typing import ParamSpec, TypeVar

import tiktoken

from alexis.config import settings

P = ParamSpec("P")
R = TypeVar("R")
_Type = TypeVar("_Type")


encoding = tiktoken.get_encoding(settings.MODEL_ENCODING_NAME)


def cast_fn(type: Callable[P, R]):
    """Copy the signature of a function."""

    def cast(func: Callable) -> Callable[P, R]:
        return func  # type: ignore

    return cast


def load_entry_point(import_string: str, cast: type[_Type] = Callable) -> _Type:  # type: ignore
    """Load an entrypoint by name."""
    entry_point = EntryPoint(name=None, group=None, value=import_string)  # type: ignore
    return entry_point.load()


def pascal_to_snake(name: str) -> str:
    """Convert a pascal case string to snake case."""
    return "".join(
        ["_" + i.lower() if i.isupper() else i for i in name]
    ).lstrip("_")  # noqa: E501


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(tz=timezone.utc)


def count_tokens(text):
    """Count the number of tokens in a text."""
    return len(encoding.encode(text))
