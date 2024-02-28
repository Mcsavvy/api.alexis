"""Alexis data models."""
from pydantic import BaseModel, Field


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
