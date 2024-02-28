"""chain views."""
from fastapi import APIRouter, Response
from pydantic import BaseModel

from alexis.components import redis
from alexis.models import Project

project = APIRouter(prefix="/chains", tags=["chains"])


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
