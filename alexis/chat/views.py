"""Chat views."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from pydantic import BaseModel

from alexis.chat.models import Thread
from alexis.components.auth import is_authenticated
from alexis.components.contexts import (
    ProjectContext,
    TaskContext,
    preprocess_project,
)
from alexis.models import User

router = APIRouter(prefix="/chat", tags=["chat"])
project = APIRouter(prefix="/project", tags=["project"])


class ProjectExistsQuery(BaseModel):
    """Project exists query."""

    project: int
    """Project id."""
    tasks: list[int]
    """List of task ids."""


class ProjectExistsResponse(BaseModel):
    """Project exists response."""

    project: bool
    """Project exists."""
    tasks: list[bool]
    """List of task exists."""


class ProjectTask(BaseModel):
    """Project task."""

    id: int
    """Task id."""
    title: str
    """Task title."""
    number: int
    """Task number."""
    description: str
    """Task description."""


class ProjectStoreQuery(BaseModel):
    """Project store query."""

    id: int
    """Project id."""
    title: str
    """Project title."""
    description: str
    """Project description."""
    tasks: list[ProjectTask]
    """List of tasks."""

    class Config:
        """Pydantic config."""

        from_attributes = True


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
    threads = Thread.objects.filter(user=user, project=project_id)
    return list(threads)


@router.get("/threads/{thread_id}", response_model=ThreadSchema)
async def get_thread(
    thread_id: UUID, user: User = Depends(is_authenticated)
) -> Thread:
    """Get a thread."""
    try:
        thread = Thread.objects.get(thread_id)
    except Thread.DoesNotExist:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user.id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return thread


@router.post("/threads", response_model=ThreadSchema)
async def create_thread(
    data: ThreadCreateSchema, user: User = Depends(is_authenticated)
) -> Thread:
    """Create a thread."""
    project_id = int(data.project)
    if not ProjectContext.exists(project_id, [])[0]:
        raise HTTPException(
            status_code=404, detail=f"Project {project_id} not found"
        )
    title = data.title
    thread = Thread.create(
        user=user, project=data.project, title=title, commit=True
    )
    return thread


@router.get(
    "/threads/{thread_id}/messages", response_model=list[ChatMessageSchema]
)
async def get_messages(thread_id: UUID, user: User = Depends(is_authenticated)):
    """Get all messages for a thread."""
    try:
        thread = Thread.objects.get(thread_id)
    except Thread.DoesNotExist:
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
    result = ProjectContext.exists(project.project, project.tasks)
    return dict(zip(("project", "tasks"), result))


@project.post("/save")
async def save_project(project: ProjectStoreQuery, bg_tasks: BackgroundTasks):
    """Store project."""
    tasks = project.tasks
    project_is_saved = ProjectContext.exists(project.id, tasks=[])[0]
    if project_is_saved:
        proj_ctx = ProjectContext.load(project.id, include_tasks=True)
    else:
        project_dump = project.model_dump()
        project_dump["tasks"] = []
        proj_ctx = ProjectContext(**project_dump)
        bg_tasks.add_task(preprocess_project, project=proj_ctx.dump())
    for task in tasks:
        task_exists = TaskContext.exists(task.id, project.id)
        if task_exists:
            task_ctx = TaskContext.load(task.id)
        else:
            task_ctx = TaskContext(**task.model_dump())
        if task_ctx not in proj_ctx.tasks:
            proj_ctx.tasks.append(task_ctx)
    proj_ctx.tasks.sort(key=lambda x: x.number)
    proj_ctx.save()
    return Response(status_code=201)
