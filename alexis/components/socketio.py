"""socketio component."""
from collections.abc import Callable
from typing import Any, Protocol, TextIO, TypedDict

import sentry_sdk
from fastapi import APIRouter, FastAPI
from socketio import (  # type: ignore[import]
    ASGIApp,
    AsyncNamespace,
)
from socketio import (
    AsyncServer as _AsyncServer,
)

from alexis import logging
from alexis.config import settings


class AsyncServer(_AsyncServer):
    """Socketio server with session management."""

    async def _trigger_event(self, event, namespace, *args):
        """Trigger an event."""
        with sentry_sdk.configure_scope() as scope:
            scope.set_transaction_name(f"socketio.{event}")
            scope.set_tag("socketio.namespace", namespace)
            scope.set_context(
                "socketio",
                {
                    "event": event,
                    "namespace": namespace,
                    "args": args,
                },
            )
            try:
                result = await super()._trigger_event(event, namespace, *args)
            except Exception as e:
                logging.error(f"Error triggering event: {event}", exc_info=e)
                scope.capture_exception(e)
                raise
            return result


sio = AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
)
sio.instrument(
    auth={
        "username": settings.get("SOCKETIO_USERNAME", "admin"),
        "password": settings.get("SOCKETIO_PASSWORD", "admin"),
    },
    mode=settings.get("SOCKETIO_MODE", settings.env),
)
app = ASGIApp(sio)


class Namespace(AsyncNamespace):
    """Base namespace."""

    def __init__(self, namespace: str, server: AsyncServer):
        """Initialize the namespace."""
        self.namespace = namespace
        super().__init__(namespace)
        self.server = server

    def on(self, event: str):
        """Register an event handler."""

        def decorator(f: Callable):
            setattr(self, f"on_{event}", f)
            return f

        return decorator

    async def on_connect(self, sid, environ):
        """On connect."""
        print(type(environ))
        print(f"Connected: {sid}")

    async def on_disconnect(self, sid):
        """On disconnect."""
        print(f"Disconnected: {sid}")


class SupportsRead(Protocol):
    """Supports read protocol."""

    def read(self, n: int) -> bytes:
        """Read n bytes."""
        ...


class ASGIScope(TypedDict):
    """ASGI Scope."""

    app: FastAPI
    app_root_path: str
    asgi: dict[str, str]
    client: tuple[str, int]
    endpoint: ASGIApp
    extensions: dict
    headers: list[tuple[bytes, bytes]]
    http_version: str
    path: str
    path_params: dict
    query_string: bytes
    raw_path: bytes
    root_path: str
    router: APIRouter
    scheme: str
    server: tuple[str, int]  # (host, port)
    starlette_exception_handlers: dict
    state: dict
    subprotocols: list[Any]
    type: str


class SocketIOEnviron(TypedDict):
    """WebSocket connection info."""

    HTTP_CONNECTION: str
    HTTP_HOST: str
    HTTP_SEC_WEBSOCKET_EXTENSIONS: str
    HTTP_SEC_WEBSOCKET_KEY: str
    HTTP_SEC_WEBSOCKET_VERSION: str
    HTTP_UPGRADE: str
    PATH_INFO: str
    QUERY_STRING: str
    RAW_URI: str
    REMOTE_ADDR: str
    REMOTE_PORT: str
    REQUEST_METHOD: str
    SCRIPT_NAME: str
    SERVER_NAME: str
    SERVER_PORT: str
    SERVER_PROTOCOL: str
    SERVER_SOFTWARE: str
    asgi_receive: Callable
    asgi_scope: ASGIScope
    asgi_send: Callable
    wsgi_async: bool
    wsgi_errors: TextIO
    wsgi_input: SupportsRead
    wsgi_multiprocess: bool
    wsgi_multithread: bool
    wsgi_run_once: bool
    wsgi_url_scheme: str
    wsgi_version: tuple[int, int]
