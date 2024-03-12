"""Alexis chat socket namespace."""
from dataclasses import dataclass
from typing import Any, TypedDict
from uuid import UUID, uuid4

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, model_validator
from socketio import AsyncServer  # type: ignore[import]

from alexis import logging
from alexis.chat.chains import AlexisChain
from alexis.chat.models import Thread
from alexis.components import redis
from alexis.components.socketio import SocketIOConnectionInfo, sio


def _make_config(
    user_id: str,
    thread_id: str,
    query_id: str,
    response_id: str,
    data: redis.SocketData,
    max_token_limit: int = 2000,
) -> RunnableConfig:
    """Make a config."""
    from langserve import __version__  # type: ignore[import]

    return {
        "run_name": "AlexisChain",
        "metadata": {
            "__useragent": data["user_agent"],
            "__langserve_version": __version__,
            "__langserve_endpoint": "ws:query",
        },
        "max_concurrency": None,
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
            "max_token_limit": max_token_limit,
            "query_id": query_id,
            "response_id": response_id,
        },
    }


class AuthInfo(BaseModel):
    """Auth info."""

    accessToken: str  # noqa: N815
    projectID: int  # noqa: N815


class QueryPayload(BaseModel):
    """Query payload."""

    query: str
    project_id: int
    user_id: UUID
    thread_id: UUID = None  # type: ignore

    @model_validator(mode="before")
    @classmethod
    def validate_thread_id(cls, data: dict):
        """Validate the thread ID."""
        if not data.get("thread_id"):
            if not data.get("project_id"):
                raise ValueError(
                    "Either thread_id or project_id must be provided"
                )
            thread = Thread.create(
                project=int(data["project_id"]), user_id=UUID(data["user_id"])
            )
            data["thread_id"] = thread.id
        return data


@dataclass
class SocketIOCallbackHandler(AsyncCallbackHandler):
    """Redis backed socket.io callback handler."""

    sid: str
    sio: AsyncServer

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """On new token."""
        self.sio.emit("response_token", {"token": token}, to=self.sid)
        return await super().on_llm_new_token(
            token,
            **kwargs,
        )


@sio.on("connect")
async def connect(sid, environ: SocketIOConnectionInfo, auth: dict):
    """On connect."""
    from alexis.components.auth import is_authenticated
    auth_info = AuthInfo(**auth)
    try:
        user = await is_authenticated(auth_info.accessToken)
    except Exception:
        return False
    user_agent = ""
    for k, v in environ["asgi.scope"]["headers"]:  # type: ignore
        if k.lower() == b"user-agent":
            user_agent = v.decode()
            break
    data: redis.SocketData = {
        "user": user.uid,
        "project": str(auth_info.projectID),
        "user_agent": user_agent,
    }
    await redis.SocketConnection.open(sid, data)
    print(f"Connected: {sid}")


@sio.on("disconnect")
async def disconnect(sid):
    """On disconnect."""
    await redis.SocketConnection.close(sid)
    print(f"Disconnected: {sid}")


@sio.on("query")
async def query(sid: str, data: dict):
    """On query."""

    class Input(TypedDict):
        query: str

    connection = await redis.SocketConnection.get(sid)
    project = connection["project"]
    user = connection["user"]

    logging.info(
        "Query from user '%s' in connection '%s' using project '%s'",
        user[:8],
        sid,
        project,
    )
    data.update(user_id=user, project_id=project)
    payload = QueryPayload(**data)
    query_id, response_id = uuid4().hex, uuid4().hex
    logging.debug("Thread ID: %s", payload.thread_id.hex[:8])
    logging.debug("Query ID: %s", query_id[:8])
    logging.debug("Response ID: %s", response_id[:8])
    config = _make_config(
        user_id=payload.user_id.hex,
        thread_id=payload.thread_id.hex,
        query_id=query_id,
        response_id=response_id,
        data=connection,
    )
    chain = AlexisChain.with_types(
        input_type=Input, output_type=str
    ).with_config(config)
    thread = Thread.get(payload.thread_id)
    await sio.emit(
        "chat_info",
        {
            "thread_id": thread.uid,
            "thread_title": thread.title,
            "query_id": query_id,
            "response_id": response_id,
        },
        to=sid,
    )
    async for chunk in chain.astream({"query": payload.query}):
        await sio.emit("response", {"chunk": chunk}, to=sid)
    print(f"Response sent to {sid}")
