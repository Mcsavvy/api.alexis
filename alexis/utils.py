"""Utility functions for the Alexis API."""

from collections.abc import Callable
from importlib.metadata import EntryPoint
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
_Type = TypeVar("_Type")


def cast_fn(type: Callable[P, R]):
    """Copy the signature of a function."""

    def cast(func: Callable) -> Callable[P, R]:
        return func  # type: ignore

    return cast


def load_entry_point(import_string: str, cast: type[_Type] = Callable) -> _Type:  # type: ignore
    """Load an entrypoint by name."""
    entry_point = EntryPoint(name=None, group=None, value=import_string)  # type: ignore
    return entry_point.load()
