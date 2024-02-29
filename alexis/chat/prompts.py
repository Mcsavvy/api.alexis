"""Chatbot Prompt Templates."""

from pathlib import Path

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

PROJECT_PROMPT = Path(__file__).parent / "prompts" / "project.txt"
TASK_PROMPT = Path(__file__).parent / "prompts" / "task.txt"

assert PROJECT_PROMPT.exists(), "project prompt does not exist"
assert TASK_PROMPT.exists(), "task prompt does not exist"


ProjectPrompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(PROJECT_PROMPT.read_text()),
        MessagesPlaceholder("history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

ProjectPrompt.name = "ProjectPrompt"

TaskPrompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(TASK_PROMPT.read_text()),
        MessagesPlaceholder("history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

TaskPrompt.name = "TaskPrompt"
