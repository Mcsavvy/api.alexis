"""Utility functions for the Alexis API."""

from collections.abc import Callable
from contextvars import ContextVar
from importlib.metadata import EntryPoint
from typing import Any, Generic, ParamSpec, TypeVar

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


class LocalProxy(Generic[_Type]):
    """Proxy to a contextvar."""

    def __init__(self, var: ContextVar[_Type], error: str | None = None):
        """Initialize the local proxy."""
        self._var = var
        self._error = error or f"Working outside of {var.name} context."

    def _get_object(self) -> _Type:
        try:
            return self._var.get()
        except LookupError:
            raise RuntimeError(self._error) from None

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the local context."""
        return getattr(self._get_object(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute in the local context."""
        if name in ("_var", "_error"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._get_object(), name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute from the local context."""
        delattr(self._get_object(), name)

    def __repr__(self) -> str:
        """Return a string representation of the local proxy."""
        return repr(self._get_object())
