"""Chat views."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from alexis.components import is_authenticated, redis, session
from alexis.models import Thread, User

router = APIRouter(prefix="/chat", tags=["chat"])


class ThreadSchema(BaseModel):
    """Thread schema."""

    id: UUID
    title: str
    project: int

    class Config:
        """Pydantic config."""

        from_attributes = True


class ThreadCreateSchema(BaseModel):
    """Thread create schema."""

    project: int
    title: str | None = None


class ChatMessageSchema(BaseModel):
    """Chat message schema."""

    id: UUID
    thread_id: UUID
    content: str
    chat_type: str
    order: int
    previous_chat_id: UUID | None
    next_chat_id: UUID | None

    class Config:
        """Pydantic config."""

        from_attributes = True


@router.get("/threads/{project_id}", response_model=list[ThreadSchema])
async def get_threads(
    project_id: int, user: User = Depends(is_authenticated)
) -> list[Thread]:
    """Get all threads for a project."""
    threads = (
        session.query(Thread)
        .filter(Thread.user_id == user.id, Thread.project == project_id)
        .all()
    )
    return threads


@router.get("/threads/{thread_id}", response_model=ThreadSchema)
async def get_thread(
    thread_id: UUID, user: User = Depends(is_authenticated)
) -> Thread:
    """Get a thread."""
    try:
        thread = Thread.get(thread_id)
    except Thread.DoesNotExistError:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return thread


@router.post("/threads", response_model=ThreadSchema)
async def create_thread(
    data: ThreadCreateSchema, user: User = Depends(is_authenticated)
) -> Thread:
    """Create a thread."""
    project_id = str(data.project)
    project = redis.get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project {project_id} not found"
        )
    title = data.title or project.title

    thread = Thread.create(
        user_id=user.id, project=data.project, title=title, commit=True
    )
    return thread


@router.get(
    "/threads/{thread_id}/messages", response_model=list[ChatMessageSchema]
)
async def get_messages(thread_id: UUID, user: User = Depends(is_authenticated)):
    """Get all messages for a thread."""
    try:
        thread = Thread.get(thread_id)
    except Thread.DoesNotExistError:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    messages = thread.chats
    return messages
