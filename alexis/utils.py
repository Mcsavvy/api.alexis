"""Utility functions for the Alexis API."""

from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from typing import Any, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
_Type = TypeVar("_Type")


def print_context():
    """Print execution context."""
    from asyncio import current_task
    from threading import current_thread

    thread = current_thread()
    print(f"[*] Thread('{thread.name}'): {thread.ident}")
    try:
        task = current_task()
        if not task:
            raise RuntimeError
        print(f"[*] Task('{task.get_name()}'): {id(task)}")
    except RuntimeError:
        print("[*] Task: None")


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


class LocalProxy(Generic[_Type]):
    """Proxy to a contextvar."""

    def __init__(
        self,
        var: ContextVar[_Type],
        error: str | None = None,
        factory: Callable[[], _Type] | None = None,
    ):
        """Initialize the local proxy."""
        self._var = var
        self._error = error or f"Working outside of {var.name} context."
        self._factory = factory

    def _get_object(self) -> _Type:
        try:
            return self._var.get()
        except LookupError:
            if self._factory is not None:
                value = self._factory()
                # print_context()
                self._var.set(value)
                try:
                    return self._var.get()
                except LookupError:
                    raise RuntimeError(self._error) from None
            raise RuntimeError(self._error)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the local context."""
        if name in ("_var", "_error", "_factory"):
            return object.__getattribute__(self, name)
        return getattr(self._get_object(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute in the local context."""
        if name in ("_var", "_error", "_factory"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._get_object(), name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute from the local context."""
        delattr(self._get_object(), name)

    def __repr__(self) -> str:
        """Return a string representation of the local proxy."""
        return repr(self._get_object())
