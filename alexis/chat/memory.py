"""chatbot memory."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable, List, overload  # noqa: F401,UP035
from uuid import UUID

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from alexis import logging
from alexis.chat.models import MChat, MThread
from alexis.components import session


@dataclass
class ThreadChatMessageHistory(BaseChatMessageHistory):
    """Thread chat message history."""

    thread_id: str | UUID
    query_id: str | None = None
    response_id: str | None = None
    max_token_limit: int = 3000
    messages: list[BaseMessage] = field(default_factory=list)
    prev: MChat | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Initialize the thread chat message history."""
        self.load_messages()

    def load_messages(self) -> None:
        """Load messages."""
        try:
            thread = MThread.objects.get(self.thread_id)
        except MThread.DoesNotExist:
            self.messages = []
            return
        chats = thread.chats
        buffer: list[BaseMessage] = []
        token_count = 0
        for chat in reversed(chats):
            cost = chat.cost
            if not cost and chat.content.strip():
                chat.compute_token_cost()
                chat.save()
            if token_count + cost > self.max_token_limit:
                break
            token_count += cost
            buffer.append(chat.to_message())
        self.messages = buffer[::-1]

    def add_user_message(self, message: HumanMessage | str) -> None:
        """Add a user message to the history."""
        logging.debug(
            "Adding user message with ID %s to history",
            self.query_id[:8] if self.query_id else "None",
        )
        if isinstance(message, str):
            msg = HumanMessage(content=message)
        else:
            msg = message
        chat_data: dict[str, Any] = {
            "content": str(msg.content),
            "previous_chat": self.prev,
            "commit": True,
        }
        if self.query_id:
            chat_data["id"] = UUID(self.query_id)
            # prevent reuse of same query ID
            self.query_id = None
        try:
            thread = MThread.objects.get(self.thread_id)
        except MThread.DoesNotExist:
            return
        self.prev = thread.add_query(**chat_data)

    def add_ai_message(self, message: AIMessage | str) -> None:
        """Add an AI message to the history."""
        logging.debug(
            "Adding AI message with ID %s to history",
            self.response_id[:8] if self.response_id else "None",
        )
        if isinstance(message, str):
            msg = AIMessage(content=message)
        else:
            msg = message
        chat_data: dict[str, Any] = {
            "content": str(msg.content),
            "previous_chat": self.prev,
            "commit": True,
        }
        if self.response_id:
            chat_data["id"] = UUID(self.response_id)
            # prevent reuse of same response ID
            self.response_id = None
        try:
            thread = MThread.objects.get(self.thread_id)
        except MThread.DoesNotExist:
            return
        self.prev = thread.add_response(**chat_data)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the history."""
        for message in messages:
            if isinstance(message, HumanMessage) or (
                isinstance(message, BaseMessage) and message.type == "human"
            ):
                self.add_user_message(message)  # type: ignore
            elif isinstance(message, AIMessage) or (
                isinstance(message, BaseMessage) and message.type == "ai"
            ):
                self.add_ai_message(message)  # type: ignore
        self.load_messages()

    def clear(self) -> None:
        """Clear the messages."""
        try:
            thread = MThread.objects.get(self.thread_id)
            thread.clear()
            session.refresh(thread)
        except MThread.DoesNotExist:
            pass
        self.load_messages()


def get_history_from_thread(
    user_id: str,
    thread_id: str | UUID,
    max_token_limit: int,
    query_id: str | None = None,
    response_id: str | None = None,
) -> ThreadChatMessageHistory:
    """Get chat history from thread."""
    history = ThreadChatMessageHistory(
        thread_id=thread_id,
        max_token_limit=max_token_limit,
        query_id=query_id,
        response_id=response_id,
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
