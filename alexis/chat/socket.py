"""Alexis chat socket namespace."""
from typing import Any, TypedDict
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from alexis import logging
from alexis.chat.chains import AlexisChain
from alexis.chat.models import Thread
from alexis.components.socketio import SocketIOEnviron, sio
from alexis.models import User


async def _make_config(
    sid: str,
    thread_id: str,
    query_id: str,
    response_id: str,
    max_token_limit: int = 2000,
) -> RunnableConfig:
    """Make a config."""
    from langserve import __version__  # type: ignore[import]

    session: SessionData = await sio.get_session(sid)
    environ: SocketIOEnviron = sio.get_environ(sid)
    user_agent = ""
    for key, val in environ["asgi.scope"]["headers"]:  # type: ignore
        if key == b"user-agent":
            user_agent = val.decode()
            break

    return {
        "run_name": "AlexisChain",
        "metadata": {
            "__useragent": user_agent,
            "__langserve_version": __version__,
            "__langserve_endpoint": "ws:query",
        },
        "max_concurrency": None,
        "configurable": {
            "user": session["user"],
            "user_id": session["user_id"],
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


class SessionData(TypedDict):
    """Session data."""

    user: str
    user_id: str
    project: str


class QueryPayload(BaseModel):
    """Query payload."""

    query: str
    project_id: int = 0
    user_id: UUID
    thread_id: UUID = Field(default_factory=uuid4)


@sio.on("connect")
async def connect(sid, environ: Any, auth: dict):
    """On connect."""
    from alexis.components.auth import is_authenticated

    auth_info = AuthInfo(**auth)
    try:
        user = await is_authenticated(auth_info.accessToken)
    except Exception:
        return False
    data: SessionData = {
        "user_id": user.uid,
        "project": str(auth_info.projectID),
        "user": user.name,
    }
    await sio.save_session(sid, data)
    logging.debug(f"Connected: {sid}")


@sio.on("disconnect")
async def disconnect(sid):
    """On disconnect."""
    logging.debug(f"Disconnected: {sid}")


@sio.on("query")
async def query(sid: str, data: dict):
    """On query."""

    class Input(TypedDict):
        query: str

    session: SessionData = await sio.get_session(sid)
    data.setdefault("user_id", session["user_id"])
    data.setdefault("project_id", session["project"])

    logging.info(
        "Query from user '%s' in connection '%s' using project '%s'",
        data["user_id"][:8],
        sid,
        data["project_id"],
    )
    payload = QueryPayload(**data)  # type: ignore[arg-type]
    query_id, response_id = str(uuid4()), str(uuid4())
    # create thread if it doesn't exist
    if not Thread.objects.filter(id=payload.thread_id).count():
        title = payload.query.split("\n")[0][:100]
        user = User.objects.get(payload.user_id)
        Thread.create(
            id=payload.thread_id,
            title=title,
            user=user,
            project=payload.project_id,
        )
    logging.debug("Thread ID: %s", payload.thread_id.hex[:8])
    logging.debug("Query ID: %s", query_id[:8])
    logging.debug("Response ID: %s", response_id[:8])
    config = await _make_config(
        sid,
        thread_id=str(payload.thread_id),
        query_id=query_id,
        response_id=response_id,
    )
    chain = AlexisChain.with_types(
        input_type=Input, output_type=str
    ).with_config(config)
    thread = Thread.objects.get(payload.thread_id)
    await sio.emit(
        "chat_info",
        {
            "thread_id": thread.uid,
            "thread_title": thread.title,
            "thread_created_at": thread.created_at.isoformat(),
            "thread_description": thread.description,
            "query_id": query_id,
            "response_id": response_id,
        },
        to=sid,
    )
    async for chunk in chain.astream({"query": payload.query}):
        await sio.emit("response", {"chunk": chunk}, to=sid)
    logging.debug(f"Response sent to {sid}")
