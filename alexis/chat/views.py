"""Chat views."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from alexis.chat.models import MThread
from alexis.components import redis
from alexis.components.auth import is_authenticated
from alexis.models import MUser, Project

router = APIRouter(prefix="/chat", tags=["chat"])
project = APIRouter(prefix="/project", tags=["project"])


class ProjectExistsQuery(BaseModel):
    """Project exists query."""

    project: str
    """Project id."""
    tasks: list[str]
    """List of task ids."""


class ProjectExistsResponse(BaseModel):
    """Project exists response."""

    project: bool
    """Project exists."""
    tasks: list[bool]
    """List of task exists."""


class ProjectStoreQuery(Project):
    """Project store query."""


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
    project_id: int, user: MUser = Depends(is_authenticated)
) -> list[MThread]:
    """Get all threads for a project."""
    threads = MThread.objects.filter(user=user, project=project_id)
    return list(threads)


@router.get("/threads/{thread_id}", response_model=ThreadSchema)
async def get_thread(
    thread_id: UUID, user: MUser = Depends(is_authenticated)
) -> MThread:
    """Get a thread."""
    try:
        thread = MThread.objects.get(thread_id)
    except MThread.DoesNotExist:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user.id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return thread


@router.post("/threads", response_model=ThreadSchema)
async def create_thread(
    data: ThreadCreateSchema, user: MUser = Depends(is_authenticated)
) -> MThread:
    """Create a thread."""
    project_id = str(data.project)
    project = redis.get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project {project_id} not found"
        )
    title = data.title or project.title
    thread = MThread.create(
        user=user, project=data.project, title=title, commit=True
    )
    return thread


@router.get(
    "/threads/{thread_id}/messages", response_model=list[ChatMessageSchema]
)
async def get_messages(
    thread_id: UUID, user: MUser = Depends(is_authenticated)
):
    """Get all messages for a thread."""
    try:
        thread = MThread.objects.get(thread_id)
    except MThread.DoesNotExist:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user.id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    messages = list(thread.iter_chats())
    order = 0
    for message in messages:
        message.thread_id = thread_id
        message.previous_chat_id = (
            message.previous_chat.id if message.previous_chat else None
        )
        message.next_chat_id = (
            message.next_chat.id if message.next_chat else None
        )
        message.order = order
        order += 1
    return messages


# a route to query if a project is stored in the database
@project.post("/exists", response_model=ProjectExistsResponse)
async def project_exists(project: ProjectExistsQuery):
    """Check if project exists."""
    result = redis.project_exists(project=project.project, tasks=project.tasks)
    return {"project": result[0], "tasks": result[1]}


@project.post("/save")
async def save_project(project: ProjectStoreQuery):
    """Store project."""
    redis.store_project(project)
    return Response(status_code=201)
