"""Alexis Chains."""

from typing import TypedDict, cast

from fastapi import HTTPException, Request
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
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
from .prompts import ProjectPrompt  # noqa: F401, F403

PROJECT_SHORT_FMT = "{title} (`id={id}`)"

PROJECT_FMT = """
## {title}  (`id={id}`)
{description}

### Tasks
{tasks}
""".strip()

TASK_SHORT_FMT = "- {number}. {title} (`id={id}`)"


TASK_FMT = """
## {number}. {title}  (`id={id}`)
{description}
""".strip()

TASK_DETAILS_DESCRIPTION = """
Extract all information about a task in a project. \
Useful for when a user asks a task-specific question. \
The task information is bulky so avoid using this tool \
if you don't need the full task information.
""".strip()


class TaskDetailsInput(BaseModel):
    """Task details input."""

    project_id: str = Field(description="ID of the project to search in.")
    id: str = Field(
        description=(
            "ID of the task to extract information about."
            " The id is numeric and you can get the correct id "
            "by first listing all tasks in the project."
        )
    )


class ProjectDetailsInput(BaseModel):
    """Project details input."""

    project_id: str


class ProjectTaskListInput(BaseModel):
    """Project task list input."""

    project_id: str


class ProjectTaskList(BaseTool):
    """Tool that extracts the list of tasks in a project."""

    name: str = "project_task_list"
    description: str = "Lists all tasks in a project."
    args_schema: type[BaseModel] = ProjectTaskListInput

    def _run(self, project_id: str) -> str:
        """Use the tool."""
        project = fetch_project(project_id)
        return "\n".join(
            [
                TASK_SHORT_FMT.format(**task.model_dump())
                for task in project.tasks
            ]
        )


class ProjectDetails(BaseTool):
    """Tool that extracts information about a project."""

    name: str = "project_details"
    description: str = "Extracts information about a project."
    args_schema: type[BaseModel] = ProjectDetailsInput

    def _run(self, project_id: str) -> str:
        """Use the tool."""
        project = fetch_project(project_id)
        return format_project(project)


class TaskDetails(BaseTool):
    """Tool that extracts information about a task in a project."""

    name: str = "project_task_details"
    description: str = TASK_DETAILS_DESCRIPTION
    args_schema: type[BaseModel] = TaskDetailsInput

    def _run(self, project_id: str, id: str) -> str:
        """Use the tool."""
        project = fetch_project(project_id)
        task = fetch_task(project, id)
        return format_task(task)


def format_project(project: Project):
    """Convert a project to a markdown string."""
    data = project.model_dump()
    tasks = "\n".join([TASK_SHORT_FMT.format(**task) for task in data["tasks"]])
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
    return context


model = ChatOpenAI()
parser = StrOutputParser()
tools = [ProjectTaskList(), ProjectDetails(), TaskDetails()]


class ContextInput(TypedDict):
    """Context input."""

    query: str


class ContextOutput(TypedDict):
    """Context output."""

    query: str
    history: list[BaseMessage]
    context: str


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
    history = fetch_messages_from_thread(thread_id, max_token_limit)
    context = load_project_context(project)
    return {
        "query": data["query"],
        "history": history,
        "context": context,
    }


@chain
def GetExecutor(data: ContextOutput) -> AgentExecutor:  # noqa: N802
    """Select prompt."""
    from langchain import hub

    prompt = hub.pull("alexis/project-prompt")
    agent = create_openai_tools_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)  # type: ignore
    return executor


@chain
def ParseOutput(data: dict) -> str:  # noqa: N802
    """Parse the output."""
    return data["output"]


AlexisChain = RunnableWithMessageHistory(
    (
        GetChainContext  # type: ignore[arg-type]
        | GetExecutor
        | ParseOutput
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
