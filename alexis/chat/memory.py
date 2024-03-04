"""chatbot memory."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (  # noqa: F401,UP035
    Iterable,
    List,
    overload,
)
from uuid import UUID

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from alexis.components import session
from alexis.models import Thread


@dataclass
class ThreadChatMessageHistory(BaseChatMessageHistory):
    """Thread chat message history."""

    thread_id: str | UUID
    max_token_limit: int = 3000
    messages: list[BaseMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the thread chat message history."""
        self.load_messages()

    def load_messages(self) -> None:
        """Load messages."""
        try:
            thread = Thread.get(self.thread_id)
        except Thread.DoesNotExistError:
            self.messages = []
            return
        chats = thread.chats
        buffer: list[BaseMessage] = []
        token_count = 0
        for chat in chats[::-1]:
            cost = chat.cost
            if not cost and chat.content.strip():
                chat.compute_token_cost()
                chat.save()
            if token_count + cost > self.max_token_limit:
                break
            token_count += cost
            buffer.append(chat.to_message())
        self.messages = buffer[::-1]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the history."""
        prev = None
        try:
            thread = Thread.get(self.thread_id)
        except Thread.DoesNotExistError:
            return
        for message in messages:
            if isinstance(message, HumanMessage) or (
                isinstance(message, BaseMessage) and message.type == "human"
            ):
                prev = thread.add_query(
                    content=str(message.content),
                    commit=True,
                    previous_chat=prev,
                )
            elif isinstance(message, AIMessage) or (
                isinstance(message, BaseMessage) and message.type == "ai"
            ):
                prev = thread.add_response(
                    content=str(message.content),
                    commit=True,
                    previous_chat=prev,
                )
        self.load_messages()

    def clear(self) -> None:
        """Clear the messages."""
        try:
            thread = Thread.get(self.thread_id)
            thread.clear()
            session.refresh(thread)
        except Thread.DoesNotExistError:
            pass
        self.load_messages()


def get_history_from_thread(
    thread_id: str | UUID,
    max_token_limit: int,
    user_id: str,
) -> ThreadChatMessageHistory:
    """Get chat history from thread."""
    history = ThreadChatMessageHistory(
        thread_id=thread_id, max_token_limit=max_token_limit
    )
    history.load_messages()
    return history


def fetch_messages_from_thread(
    thread_id: str | UUID, max_token_limit: int
) -> list[BaseMessage]:
    """Fetch messages from thread."""
    history = ThreadChatMessageHistory(
        thread_id=thread_id,
        max_token_limit=max_token_limit,
    )
    history.load_messages()
    return list(history.messages)
