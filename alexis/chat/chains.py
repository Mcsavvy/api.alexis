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

from alexis.components.contexts import (
    ContextNotFound,
    Project,
)

from ..models import Thread, User
from .callbacks import StreamCallbackHandler  # noqa: F401, F403
from .memory import (  # noqa: F401, F403
    fetch_messages_from_thread,
    get_history_from_thread,
)

model = ChatOpenAI()
parser = StrOutputParser()
prompt = hub.pull("alexis/project-prompt")


class ContextInput(TypedDict):
    """Context input."""

    query: str
    # history: list[BaseMessage]


class ContextOutput(TypedDict):
    """Context output."""

    query: str
    history: list[BaseMessage]
    context: str


def load_project_context(
    project: Project, task_ids: list[int] | None = None, query: str = ""
) -> str:
    """Format a project and task."""
    show_tasks = any(kw in query.lower() for kw in ["task"])
    context = project.format(show_tasks=show_tasks, included_tasks=task_ids)
    return context


def extract_tasks(query: str, project: Project) -> list[int]:
    """Extract task numbers from a query."""
    import re

    task_inclusion = re.compile(r"task #?(?P=<task_no>\d+)", re.I | re.M)
    task_nums: list[str] = []
    task_ids: list[int] = []
    for match in task_inclusion.finditer(query):
        task_nums.append(match.group("task_no"))
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
        user = User.objects.get(user_id)
    except User.DoesNotExist:
        raise HTTPException(
            status_code=404, detail=f"User '{user_id}' not found"
        )
    try:
        thread = Thread.objects.get(thread_id)
    except Thread.DoesNotExist:
        raise HTTPException(
            status_code=404, detail=f"Thread '{thread_id}' not found"
        )
    if thread.user.id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        project = Project.load(thread.project)
    except ContextNotFound:
        raise HTTPException(
            status_code=404, detail=f"Project '{thread.project}' not found"
        )
    included_tasks = extract_tasks(data["query"], project)
    context = load_project_context(project, included_tasks)
    return {
        "query": data["query"],
        "history": data["history"],  # type: ignore
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
        ConfigurableFieldSpec(
            id="query_id",
            annotation=str,
            name="Query ID",
            description="ID of the query",
            is_shared=True,
            default=None,
        ),
        ConfigurableFieldSpec(
            id="response_id",
            annotation=str,
            name="Response ID",
            description="ID of the response",
            is_shared=True,
            default=None,
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
        user = User.objects.get(user_id)
    except User.DoesNotExist:
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
    config["configurable"]["user"] = user.name
    return config
