"""Alexis Redis component."""
from typing import Any, cast

from redis import Redis

from alexis.config import settings
from alexis.models import Project

client = Redis.from_url(settings.REDIS_URL)


def d(value: Any) -> Any:
    """Try to decode bytes."""
    if isinstance(value, bytes):
        return value.decode()
    return value


def project_exists(project: str, tasks: list[str]):
    """Check if project exists."""
    task_exists = [False] * len(tasks)
    project_key = f"project:{project}"
    project_exists = bool(client.exists(project_key))
    for i, task in enumerate(tasks):
        task_key = f"{project_key}:task:{task}"
        task_exists[i] = bool(client.exists(f"{task_key}"))
    return project_exists, task_exists


def get_project(project_id: str, include_tasks: bool = True) -> Project | None:
    """Get project."""
    project_key = f"project:{project_id}"
    project = cast(dict, client.hgetall(project_key))
    if not project:
        return None
    tasks = project.pop(b"tasks").split(b",")
    project["tasks"] = []
    project["id"] = project_id
    if include_tasks:
        for task_id in tasks:
            task_key = f"{project_key}:task:{task_id.decode()}"
            task = cast(dict, client.hgetall(task_key))
            task["id"] = d(task_id)
            del task[b"project"]
            project["tasks"].append(
                {d(key): d(value) for key, value in task.items()}
            )
    project = {d(key): d(value) for key, value in project.items()}
    return Project(**project)


def store_project(project: Project):
    """Store project."""
    project_key = f"project:{project.id}"
    if not client.exists(project_key):
        client.hset(
            project_key,
            mapping={
                "title": project.title,
                "description": project.description,
                "tasks": ",".join(t.id for t in project.tasks),
            },
        )
    for task in project.tasks:
        task_key = f"{project_key}:task:{task.id}"
        if not client.exists(task_key):
            client.hset(
                task_key,
                mapping={
                    "project": project.id,
                    "title": task.title,
                    "number": task.number,
                    "description": task.description,
                },
            )


def delete_project(project_id: str):
    """Delete project."""
    project_key = f"project:{project_id}"
    if client.exists(project_key):
        client.delete(project_key)
    task_ids: list[str] = client.keys(f"{project_key}:task:*")  # type: ignore
    for task in task_ids:
        client.delete(task)


def get_all_projects(limit: int = 100) -> list[str]:
    """Get all projects ids."""
    import re

    key: bytes
    projects: list[str] = []
    for key in client.keys("project:*[0-9]"):  # type: ignore
        if len(projects) >= limit:
            break
        if re.match(r"^project:[0-9]+$", key.decode()):
            id = key.decode().split(":")[-1]
            projects.append(id)
    return projects


def get_all_tasks(project_id: str, limit: int = 100) -> list[str]:
    """Get all tasks ids."""
    tasks: list[bytes] = client.keys(f"project:{project_id}:task:*")  # type: ignore
    return [t.decode().split(":")[-1] for t in tasks[:limit]]
