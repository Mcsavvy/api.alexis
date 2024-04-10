"""Context for the LLM."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeAlias, TypedDict

from typing_extensions import Self

from alexis import logging
from alexis.components.storage import default_storage as storage
from alexis.utils import count_tokens

FormattedContext: TypeAlias = str


class ContextError(Exception):
    """Context error."""


class ContextNotFound(ContextError):  # noqa: N818
    """Context not found."""


class Project(TypedDict):
    """Project details."""

    id: int
    title: str
    description: str
    tasks: list[int]


class Task(TypedDict):
    """Task details."""

    id: int
    project: int
    title: str
    number: int
    description: str


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


@dataclass(eq=False, repr=False)
class ProjectContext(BaseContext):
    """A project details."""

    collection: ClassVar[str] = "projects"
    id: int
    tasks: list["TaskContext"] = field(default_factory=list)
    title: str = ""
    description: str = ""

    def __eq__(self, other: object) -> bool:
        """Check if two projects are equal."""
        if not isinstance(other, ProjectContext):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        """Return the string representation of this project."""
        return f"Project[{self.id}] '{self.title}'"

    @classmethod
    def exists(cls, id: int, tasks: list[int]) -> tuple[bool, list[bool]]:
        """Check if a project exists."""
        project_exists = storage.get(cls.collection, id=id) is not None
        tasks_exist = [
            TaskContext.exists(task, project=id) is not None for task in tasks
        ]
        return project_exists, tasks_exist

    @classmethod
    def load(cls, id: int, include_tasks: bool = False, **query) -> Self:
        """Load project from database."""
        project = storage.get(cls.collection, id=id, **query)
        if project is None:
            raise ContextNotFound(f"Project('{id}') not found.")
        tasks = []
        if include_tasks:
            for task_id in project["tasks"]:
                task = TaskContext.load(task_id, project=id)
                if task is not None:
                    tasks.append(task)
        project["tasks"] = tasks
        return cls(**project)

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
        fmt = f"# {self.title} {self.description}"
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

    def dump(self) -> Project:
        """Deserialize the project."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tasks": [task.dump()["id"] for task in self.tasks],
        }

    @classmethod
    def all(
        cls,
        limit: int | None = None,
        skip: int = 0,
        include_tasks: bool = False,
        **query,
    ) -> list[Self]:
        """List all projects in the database."""
        projects = []
        for id in storage.object_ids(
            cls.collection, limit=limit, skip=skip, **query
        ):
            projects.append(cls.load(id, include_tasks=include_tasks))
        return projects

    def delete(self):
        """Delete this project from the database."""
        storage.delete(self.collection, self.id)


@dataclass(eq=False, repr=False)
class TaskContext(BaseContext):
    """A task details."""

    id: int
    project: int = 0
    collection = "tasks"
    title: str = ""
    number: int = 0
    description: str = ""

    def __eq__(self, other: object) -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, TaskContext):
            return NotImplemented
        return self.id == other.id and self.project == other.project

    def __repr__(self) -> str:
        """Return the string representation of this task."""
        return f"Task[{self.id}] '{self.number}. {self.title}'"

    @classmethod
    def exists(cls, id: int, project: int | None = None) -> bool:
        """Check if a task exists."""
        if project is not None:
            return (
                storage.get(cls.collection, id=id, project=project) is not None
            )
        return storage.get(cls.collection, id=id) is not None

    @classmethod
    def load(cls, id: int, project: int | None = None, **query) -> Self:
        """Load task from database."""
        params = query | {"id": id}
        if project is not None:
            params["project"] = project
        task = storage.get(cls.collection, **params)
        if task is None:
            raise ContextNotFound(
                f"Task('{id}') not found"
                + ("for project '{project}'." if project else ".")
            )
        return cls(**task)

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

    def dump(self) -> Task:
        """Deserialize the task."""
        return {
            "id": self.id,
            "project": self.project,
            "title": self.title,
            "number": self.number,
            "description": self.description,
        }

    @classmethod
    def all(
        cls, limit: int | None = None, skip: int = 0, **query
    ) -> list[Self]:
        """List all tasks in the database."""
        tasks = []
        for id in storage.object_ids(
            cls.collection, limit=limit, skip=skip, **query
        ):
            tasks.append(cls.load(id))
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


async def preprocess_project(project: Project) -> Project:
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
    "ProjectContext",
    "TaskContext",
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
        if ProjectContext.exists(proj_id, [])[0]:
            logging.warning(f"Project '{proj_id}' already exists.")
            try:
                proj_ctx = ProjectContext.load(proj_id, include_tasks=True)
            except ContextNotFound:
                logging.error(f"Project '{proj_id}' could not be loaded.")
                p_failed += 1
                continue
        else:
            proj_ctx = ProjectContext(
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
            if TaskContext.exists(task_id, project=proj_id):
                logging.warning(f"Task '{task_id}' already exists.")
                task_ctx = TaskContext.load(task_id)
            else:
                task_ctx = TaskContext(
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
