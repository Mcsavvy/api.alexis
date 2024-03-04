"""Alexis Chains."""

from functools import partial
from typing import TypedDict, cast

from fastapi import HTTPException, Request
from langchain import hub
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (  # noqa: F401, F403
    ConfigurableFieldSpec,
    Runnable,
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

PROJECT_SHORT_FMT = "{title} (`id={id}`)"

PROJECT_FMT = """
## {title}  (`id={id}`)
{description}

### Tasks
{tasks}
""".strip()

TASK_SHORT_FMT = """
#### {number}. {title} (`id={id}`)
[details not included]
""".strip()

TASK_FMT = """
#### {number}. {title}  (`id={id}`)
{description}
""".strip()


def format_project(project: Project, tasks: list[str]):
    """Convert a project to a markdown string."""
    data = project.model_dump()
    tasks_formatted = "\n\n".join(
        [format_task(t, t.id in tasks) for t in project.tasks]
    )
    data["tasks"] = tasks_formatted
    return PROJECT_FMT.format(**data)


def format_task(task: Task, include_details: bool = False):
    """Convert a task to a markdown string."""
    if include_details:
        return TASK_FMT.format(**task.model_dump())
    else:
        return TASK_SHORT_FMT.format(**task.model_dump())


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


def load_project_context(
    project: Project, task_ids: list[str] | None = None
) -> str:
    """Format a project and task."""
    context = format_project(project, task_ids or [])
    return context


model = ChatOpenAI()
parser = StrOutputParser()
prompt = hub.pull("alexis/project-prompt")


class ContextInput(TypedDict):
    """Context input."""

    query: str
    history: list[BaseMessage]


class ContextOutput(TypedDict):
    """Context output."""

    query: str
    history: list[BaseMessage]
    context: str


def extract_tasks(query: str, project: Project) -> list[str]:
    """Extract task numbers from a query."""
    import re

    task_inclusion = re.compile(r"#(\d+)", re.IGNORECASE | re.MULTILINE)
    task_nums: list[str] = task_inclusion.findall(query)
    task_ids: list[str] = []
    if not task_nums:
        return task_ids
    for task in project.tasks:
        if str(task.number) in task_nums:
            task_ids.append(task.id)
    return task_ids


@chain
@partial(cast, Runnable[ContextInput, ContextOutput])
async def GetChainContext(  # noqa: N802
    data: ContextInput, config: RunnableConfig
) -> ContextOutput:
    """Load chain context."""
    metadata = config["metadata"]
    for key in ["thread_id", "user_id"]:
        assert key in metadata, f"{key} not found in metadata"

    user_id: str = metadata["user_id"]
    thread_id: str = metadata["thread_id"]

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

    project_id = str(thread.project)
    project = fetch_project(project_id)
    included_tasks = extract_tasks(data["query"], project)
    context = load_project_context(project, included_tasks)
    return {
        "query": data["query"],
        "history": data["history"],
        "context": context,
    }


@chain
def ParseOutput(data: dict) -> str:  # noqa: N802
    """Parse the output."""
    return data["output"]


AlexisChain = RunnableWithMessageHistory(
    (
        GetChainContext  # type: ignore[arg-type]
        | prompt
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
