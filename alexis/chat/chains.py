"""Alexis Chains."""

from typing import TypedDict, cast

from fastapi import HTTPException, Request
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (  # noqa: F401, F403
    ConfigurableFieldSpec,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI

from ..components import redis
from ..models import Project, Task, Thread, User
from .callbacks import StreamCallbackHandler  # noqa: F401, F403
from .memory import (  # noqa: F401, F403
    fetch_messages_from_thread,
    get_history_from_thread,
)
from .prompts import ProjectPrompt, TaskPrompt

PROJECT_FMT = """
## {title}
{description}

### tasks:
{tasks}
""".strip()


TASK_SHORT_FMT = "{number}. {title}"


TASK_FMT = """
## {number}. {title}
{description}
""".strip()


def format_project(project: Project):
    """Convert a project to a markdown string."""
    tasks = "\n".join(
        [TASK_SHORT_FMT.format(**task.model_dump()) for task in project.tasks]
    )
    data = project.model_dump()
    data["tasks"] = tasks
    return PROJECT_FMT.format(**data)


def format_task(task: Task):
    """Convert a task to a markdown string."""
    return TASK_FMT.format(**task.model_dump())


def fetch_project(project_id: str) -> Project:
    """Fetch a project."""
    project = redis.get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project '{project_id}' not found"
        )
    return project


def fetch_task(project: str | Project, task_id: str) -> Task:
    """Fetch a task."""
    if isinstance(project, str):
        project = fetch_project(project)
    project = cast(Project, project)
    for task in project.tasks:
        if task.id == task_id:
            break
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found in project '{project.id}'",
        )
    return task


def load_project_context(project_id: str, task_id: str | None = None) -> str:
    """Format a project and task."""
    project = fetch_project(project_id)
    context = format_project(project)
    if task_id:
        task = fetch_task(project, task_id)
        context += "\n\n" + format_task(task)
    return context


model = ChatOpenAI()
parser = StrOutputParser()


class ContextInput(TypedDict):
    """Context input."""

    query: str


class ContextOutput(TypedDict):
    """Context output."""

    query: str
    history: list[BaseMessage]
    context: str
    has_task: bool


@chain  # type: ignore[arg-type]
async def GetChainContext(  # noqa: N802
    data: ContextInput, config: RunnableConfig
) -> ContextOutput:
    """Load chain context."""
    metadata = config.get("metadata", {})
    assert "thread_id" in metadata, "thread_id not found in metadata"
    assert "user_id" in metadata, "user_id not found in metadata"
    user_id: str = metadata["user_id"]
    thread_id: str = metadata["thread_id"]
    max_token_limit: int = metadata.get("max_token_limit", 3000)
    try:
        user = User.get(user_id)
    except User.DoesNotExistError:
        raise HTTPException(
            status_code=404, detail=f"User '{user_id}' not found"
        )
    try:
        thread = Thread.get(thread_id)
    except Thread.DoesNotExistError:
        raise HTTPException(
            status_code=404, detail=f"Thread '{thread_id}' not found"
        )
    if thread.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    project = str(thread.project)
    task = metadata.get("task", None)
    if task is not None:
        task = str(task)
    history = fetch_messages_from_thread(thread_id, max_token_limit)
    context = load_project_context(project, task)
    return {
        "query": data["query"],
        "history": history,
        "context": context,
        "has_task": bool(task),
    }


@chain
def GetPromptTemplate(data: ContextOutput) -> ChatPromptTemplate:  # noqa: N802
    """Select prompt."""
    has_task = data["has_task"]
    if has_task:
        return TaskPrompt
    return ProjectPrompt


AlexisChain = RunnableWithMessageHistory(
    (
        GetChainContext  # type: ignore[arg-type]
        | GetPromptTemplate
        | model
        | parser
    ),
    get_session_history=get_history_from_thread,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="thread_id",
            annotation=str,
            name="Thread ID",
            description="The thread ID",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="task",
            annotation=int,
            name="Task ID",
            description="ID of the task",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="max_token_limit",
            annotation=int,
            name="Max Token Limit",
            description="The maximum token limit",
            is_shared=True,
            default=3000,
        ),
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="ID of the current user",
            is_shared=True,
        ),
    ],
)



async def get_current_user_from_request(request: Request) -> User:
    """Get the current user for the chain."""
    from alexis.components.auth import get_token, is_authenticated, security

    creds = await security(request)
    token = await get_token(creds)
    user = await is_authenticated(token)
    return user


async def get_current_user_from_config(config: dict) -> User:
    """Get the current user for the chain."""
    user_id: str = config["configurable"].get("user_id")
    try:
        user = User.get(user_id)
    except User.DoesNotExistError:
        raise HTTPException(
            status_code=404, detail=f"User '{user_id}' not found"
        )
    return user


async def user_injection(config: dict, request: Request) -> dict:
    """Modify the config for each request."""
    config.setdefault("configurable", {})
    user_id = config["configurable"].get("user_id")
    if user_id:
        user = await get_current_user_from_config(config)
    else:
        user = await get_current_user_from_request(request)
    config["configurable"]["user_id"] = user.uid
    return config
