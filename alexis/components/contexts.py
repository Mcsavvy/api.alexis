"""Context for the LLM."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeAlias, TypedDict

from typing_extensions import Self

from alexis import logging
from alexis.components.storage import Projection
from alexis.components.storage import default_storage as storage
from alexis.utils import count_tokens

FormattedContext: TypeAlias = str


def get_excluded_properties(
    all_properties: list[str], projection: Projection
) -> list[str]:
    """Get all excluded properties from the projection.

    The projection is a list of properties to include
    """
    if projection is None:
        return []
    return [prop for prop in all_properties if prop not in projection]


class ContextError(Exception):
    """Context error."""


class ContextNotFound(ContextError):  # noqa: N818
    """Context not found."""


class BaseContext(ABC):
    """Base context class."""

    collection: ClassVar[str]
    """Collection name in the storage."""

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> Self:
        """Load the context from database."""

    @abstractmethod
    def format(self, *args, **kwargs) -> FormattedContext:
        """Build formatted context to be passed to AI."""

    @abstractmethod
    def save(self):
        """Save this context to database."""

    @abstractmethod
    def delete(self):
        """Delete this context from the database."""

    @abstractmethod
    def dump(self) -> Any:
        """Deserialize the context."""

    @classmethod
    @abstractmethod
    def all(cls, limit: int | None = None, skip: int = 0) -> list[Self]:
        """List all contexts in the database."""


class ProjectDump(TypedDict, total=False):
    """Project dump."""

    id: int
    title: str
    description: str
    tasks: list[int]


class TaskDump(TypedDict, total=False):
    """Task dump."""

    id: int
    project: int
    title: str
    number: int
    description: str


@dataclass(eq=False, repr=False)
class Project(BaseContext):
    """A project details."""

    collection: ClassVar[str] = "projects"

    id: int = 0
    """Project ID."""
    tasks: list["Task"] = field(default_factory=list)
    """List of tasks in this project."""
    title: str = ""
    """Project title."""
    description: str = ""
    """Project description."""
    projection: Projection = None
    """Fields included when loading the project.
    `None` implies all fields are included."""

    def __eq__(self, other: object) -> bool:
        """Check if two projects are equal."""
        if not isinstance(other, Project):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        """Return the string representation of this project."""
        return f"Project[{self.id}] '{self.title}'"

    @classmethod
    def exists(cls, id: int, tasks: list[int]) -> tuple[bool, list[bool]]:
        """Check if a project exists."""
        project_exists = (
            storage.get(cls.collection, id=id, only=["id"]) is not None
        )
        existing_tasks: set[int] = set()
        for res in storage.all(
            Task.collection, only=["id"], project=id, id={"$in": tasks}
        ):
            existing_tasks.add(res["id"])
        tasks_exist = [task_id in existing_tasks for task_id in tasks]
        return project_exists, tasks_exist

    @classmethod
    def load(
        cls,
        id: int,
        only: Projection = None,
        **query,
    ) -> Self:
        """Load project from database."""
        project = storage.get(cls.collection, id=id, only=only, **query)
        fields = ["id", "title", "description", "tasks"]
        excluded_fields = get_excluded_properties(fields, only)
        if project is None:
            raise ContextNotFound(f"Project('{id}') not found.")
        tasks = []
        if "tasks" not in excluded_fields:
            for task_id in project["tasks"]:
                task = Task.load(task_id, project=id)
                if task is not None:
                    tasks.append(task)
        project["tasks"] = tasks
        return cls(**project, projection=only)

    def format(
        self,
        *args,
        included_tasks: list[int] | None = None,
        show_tasks: bool = False,
    ) -> FormattedContext:
        """Format project."""
        if not self.description:
            return ""
        included_tasks = included_tasks or []
        fmt = f"# {self.title}\n{self.description}"
        tasks_fmt = ""
        for task in self.tasks:
            if task.id in included_tasks:
                tasks_fmt += "\n" + task.format()
                tasks_fmt += "\n" + task.description + "\n"
            elif show_tasks:
                tasks_fmt += "\n" + task.format() + "\n"
        if tasks_fmt:
            fmt += "\n" + "# Tasks" + tasks_fmt
        return fmt

    def save(self):
        """Save this project to storage."""
        storage.set(
            self.dump(),
            self.collection,
            self.id,
        )
        for task in self.tasks:
            task.save()

    def dump(self) -> ProjectDump:
        """Deserialize the project."""
        fields = ["id", "title", "description", "tasks"]
        excluded_fields = get_excluded_properties(fields, self.projection)
        dump: ProjectDump = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tasks": [task.id for task in self.tasks],
        }
        for _field in excluded_fields:
            dump.pop(_field, None)  # type: ignore[misc]
        return dump

    @classmethod
    def all(
        cls,
        limit: int | None = None,
        skip: int = 0,
        only: Projection = None,
        **query,
    ) -> list[Self]:
        """List all projects in the database."""
        projects: list[Self] = []
        for project in storage.all(
            cls.collection, limit=limit, skip=skip, only=only, **query
        ):
            fields = ["id", "title", "description", "tasks"]
            excluded_fields = get_excluded_properties(fields, only)
            tasks = []
            if "tasks" not in excluded_fields:
                assert "tasks" in project  # type: ignore
                assert (
                    "id" in project
                ), "`id` must be included in the projection."
                id = project["id"]
                for task_id in project["tasks"]:
                    task = Task.load(task_id, project=id)
                    if task is not None:
                        tasks.append(task)
            project["tasks"] = tasks
            projects.append(cls(**project, projection=only))
        return projects

    def delete(self):
        """Delete this project from the database."""
        storage.delete(self.collection, self.id)


@dataclass(eq=False, repr=False)
class Task(BaseContext):
    """A task details."""

    collection: ClassVar[str] = "tasks"

    id: int
    """ID of the task."""
    project: int = 0
    """ID of the project this task belongs to."""
    title: str = ""
    """Title of the task."""
    number: int = 0
    """Task number."""
    description: str = ""
    """Task description."""
    projection: Projection = None
    """Fields included when loading the task.
    `None` implies all fields are included."""

    def __eq__(self, other: object) -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, Task):
            return NotImplemented
        return self.id == other.id and self.project == other.project

    def __repr__(self) -> str:
        """Return the string representation of this task."""
        return f"Task[{self.id}] '{self.number}. {self.title}'"

    @classmethod
    def exists(cls, id: int, project: int | None = None) -> bool:
        """Check if a task exists."""
        query = {"id": id}
        if project is not None:
            query["project"] = project
        return storage.get(cls.collection, only=["id"], **query) is not None

    @classmethod
    def load(
        cls,
        id: int,
        project: int | None = None,
        only: Projection = None,
        **query,
    ) -> Self:
        """Load task from database."""
        params = query | {"id": id}
        if project is not None:
            params["project"] = project
        fields = ["id", "project", "title", "number", "description"]
        excluded_fields = get_excluded_properties(fields, only)
        task = storage.get(cls.collection, **params)
        if task is None:
            raise ContextNotFound(
                f"Task('{id}') not found"
                + ("for project '{project}'." if project else ".")
            )
        for _field in excluded_fields:
            task.pop(_field, None)  # type: ignore[misc]
        return cls(**task, projection=only)

    def format(self, *args, **kwargs) -> FormattedContext:
        """Format task."""
        return f"## {self.number}. {self.title}"

    def save(self):
        """Save this task to the storage."""
        storage.set(
            self.dump(),
            self.collection,
            self.id,
        )

    def dump(self) -> TaskDump:
        """Deserialize the task."""
        fields = ["id", "project", "title", "number", "description"]
        excluded_fields = get_excluded_properties(fields, self.projection)
        dump: TaskDump = {
            "id": self.id,
            "project": self.project,
            "title": self.title,
            "number": self.number,
            "description": self.description,
        }
        for _field in excluded_fields:
            dump.pop(_field, None)  # type: ignore[misc]
        return dump

    @classmethod
    def all(
        cls,
        limit: int | None = None,
        skip: int = 0,
        only: Projection = None,
        **query,
    ) -> list[Self]:
        """List all tasks in the database."""
        tasks: list[Self] = []
        for task in storage.all(
            cls.collection, limit=limit, skip=skip, only=only, **query
        ):
            tasks.append(cls(**task, projection=only))
        return tasks

    def delete(self):
        """Delete this task from the database."""
        storage.delete(self.collection, self.id)


async def summarize_project_description(description: str) -> str:
    """Summarize project description."""
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import (
        chain,
    )
    from langchain_openai.chat_models import ChatOpenAI

    prompt: ChatPromptTemplate = hub.pull("alexis/project-summary")
    llm = ChatOpenAI()
    parser = StrOutputParser()

    @chain
    def start(description: str) -> dict[str, str]:
        """Start the project summary chain."""
        return {"project": description}

    ProjectSummaryChain = start | prompt | llm | parser  # noqa: N806
    return await ProjectSummaryChain.ainvoke(description)


async def preprocess_project(project: ProjectDump) -> ProjectDump:
    """Preprocess project description to minimize cost."""
    from alexis.tools.preprocessors import ProjectDescriptionPreprocessor

    description: str = project["description"]
    preprocessor = ProjectDescriptionPreprocessor(description)
    logging.info(f"Preprocessing project '{project['id']}' description...")
    logging.info(f"Token count (before): {count_tokens(description)}")
    preprocessed = preprocessor.preprocess()
    token_count = count_tokens(preprocessed)
    logging.info(f"Token count (after): {token_count}")
    if token_count > 700:
        logging.info("Project description too long. Summarizing...")
        preprocessed = await summarize_project_description(description)
        logging.info(f"Token count (after): {count_tokens(preprocessed)}")
    project["description"] = preprocessed
    return project


__all__ = [
    "BaseContext",
    "Project",
    "Task",
    "ContextNotFound",
    "summarize_project_description",
    "preprocess_project",
]


def migrate():
    """Migrate from old storage to new storage."""
    from alexis.components.redis import get_all_projects, get_project

    all_projects = get_all_projects()
    p_migrated = 0
    p_failed = 0
    p_count = 0
    p_total = len(all_projects)
    logging.info(f"Migrating {p_total} projects...")
    for id in get_all_projects():
        p_count += 1
        proj_id = int(id)
        proj = get_project(id)
        logging.info(f"Migrating project '{proj_id}' ({p_count}/{p_total})")
        if proj is None:
            logging.error(f"Project '{proj_id}' could not be loaded.")
            p_failed += 1
            continue
        if Project.exists(proj_id, [])[0]:
            logging.warning(f"Project '{proj_id}' already exists.")
            try:
                proj_ctx = Project.load(proj_id, only=["id", "tasks"])
            except ContextNotFound:
                logging.error(f"Project '{proj_id}' could not be loaded.")
                p_failed += 1
                continue
        else:
            proj_ctx = Project(
                id=proj_id,
                title=proj.title,
                description=proj.description,
                tasks=[],
            )
            p_migrated += 1
        t_total = len(proj.tasks)
        t_count = 0
        logging.info(f"Migrating {t_total} tasks...")
        for task in proj.tasks:
            t_count += 1
            task_id = int(task.id)
            logging.info(f"Migrating task '{task_id}' ({t_count}/{t_total})")
            if Task.exists(task_id, project=proj_id):
                logging.warning(f"Task '{task_id}' already exists.")
                task_ctx = Task.load(task_id, only=["id", "project"])
            else:
                task_ctx = Task(
                    id=task_id,
                    project=proj_id,
                    title=task.title,
                    number=task.number,
                    description=task.description,
                )
                task_ctx.save()
            if task_id not in proj_ctx.tasks:
                proj_ctx.tasks.append(task_ctx)
        proj_ctx.save()
        logging.info(f"Migrated project '{proj_id}'")
    logging.info(
        "Migration completed. %d/%d projects migrated, %d failed.",
        p_migrated,
        p_total,
        p_failed,
    )


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        sys.exit(0)
    if args[0] == "migrate":
        migrate()
