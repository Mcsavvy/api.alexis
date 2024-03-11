"""Alexis data models."""
from pydantic import BaseModel, Field

from alexis.auth.models import User  # noqa: F401
from alexis.chat.models import Chat, ChatType, Thread  # noqa: F401


class Task(BaseModel):
    """Task model."""

    id: str
    title: str
    number: int
    description: str

    class Config:
        """Pydantic config."""

        from_attributes = True


class Project(BaseModel):
    """Project model."""

    id: str
    title: str
    description: str
    tasks: list[Task] = Field(default_factory=list)

    class Config:
        """Pydantic config."""

        from_attributes = True


__all__ = ["Chat", "ChatType", "Thread", "User", "Task", "Project"]
