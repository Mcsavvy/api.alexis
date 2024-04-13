# ruff: noqa
"""A chain for creating a context by cherrypicking parts of a project."""
from functools import partial
import json
from typing import Any, Iterable, TypedDict, cast

import jmespath
from alexis.components.contexts import Project
from langchain_core.runnables import chain, RunnableConfig, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate


class Input(TypedDict):
    """Context input."""

    project: Project | int
    query: str


class Output(TypedDict):
    """Context output."""

    json_schema: str
    query: str


@chain
@partial(cast, Runnable[str, Output])
def start_cherrypick_chain(query: str, config: RunnableConfig):
    """Cherrypick a context."""

    project_id: int = config["configurable"]["project"]
    project = Project.load(project_id, only=["id", "tasks"])
    json_schema = project.json_schema()
    return {"json_schema": json.dumps(json_schema), "query": query}


@chain
@partial(cast, Runnable[dict[str, str], str])
def context_from_jmespath_queries(
    output: dict[str, str], config: RunnableConfig
) -> str:
    """Build a context from jmespath queries."""

    project_id: int = config["configurable"]["project"]
    project = Project.load(project_id)

    def is_task(data: dict) -> bool:
        return data.get("type") == "task"

    def is_project(data: dict) -> bool:
        return data.get("type") == "project"

    def format_task(data: dict) -> str:
        if not is_task(data):
            return ""
        return "{number}. {name}\n{description}".format(**data)

    def format_project(data: dict) -> str:
        if not is_project(data):
            return ""
        fmt = "{name}\n{description}\n"
        tasks_fmt = ""
        for task in data.get("tasks", []):
            tasks_fmt += format_task(task) + "\n"
        if tasks_fmt:
            fmt += "\nTasks:\n" + tasks_fmt
        return fmt

    def format_generic(data: dict) -> str:
        fmt = ""
        for key, value in data.items():
            fmt += f"{key}: {value}\n"
        return fmt

    def format_data(data: Any) -> Iterable[str]:
        if isinstance(data, list):
            for item in data:
                yield from format_data(item)
        elif isinstance(data, dict):
            if is_task(data):
                yield format_task(data)
            if is_project(data):
                yield format_project(data)
            yield format_generic(data)
        else:
            yield str(data)

    context = ""

    for name, jmespath_query in output.items():
        jmespath_query = jmespath_query.removeprefix("project.")
        data = jmespath.search(
            jmespath_query,
            project.get_json(),
        )
        if not data:
            continue
        context += f"{name}:\n"
        context += "\n".join(format_data(data))
        context += "\n"
    return context.strip()


llm = ChatOpenAI()
prompt: ChatPromptTemplate = hub.pull("alexis/jmespath-content-cherrypicking")
parser = JsonOutputParser()
CherryPickingChain = (
    start_cherrypick_chain.with_config({"run_name": "StartCherryPick"})
    | prompt
    | llm
    | parser
    | context_from_jmespath_queries.with_config({"run_name": "ProcessQueries"})
)
CherryPickingChain.name = "CherryPickingChain"
