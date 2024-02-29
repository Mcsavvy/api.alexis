"""chatbot memory."""

from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import (  # noqa: F401,UP035
    Iterable,
    List,
    overload,
)
from uuid import UUID

from attr import dataclass
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from alexis.models import Chat, ChatType, Thread


class ChatSlidingWindow(list[BaseMessage]):
    """Token sliding window."""

    def __init__(
        self,
        chats: list[Chat],
        llm: BaseLanguageModel,
        max_token_limit: int = 3000,
    ):
        """Initialize the token sliding window."""
        self.chats = chats
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.messages: list[BaseMessage] = []
        token_count = 0

        for chat in chats[::-1]:
            msg: BaseMessage
            # TODO: improve the speed of this by caching cost
            if chat.chat_type == ChatType.QUERY:
                msg = HumanMessage(content=chat.content)
            else:
                msg = AIMessage(content=chat.content)
            count = llm.get_num_tokens_from_messages([msg])
            if token_count + count > max_token_limit:
                break
            token_count += count
            self.messages.append(msg)

    @overload  # type: ignore[override]
    def __getitem__(self, index: int) -> BaseMessage:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[BaseMessage]:
        ...

    def __getitem__(
        self, index: int | slice
    ) -> BaseMessage | Sequence[BaseMessage]:
        """Get item."""
        return self.messages[index]

    def __len__(self) -> int:
        """Return length."""
        return len(self.messages)

    def __iter__(self) -> Iterator[BaseMessage]:
        """Return iterator."""
        return iter(self.messages)

    def extend(self, __iterable: Iterable[BaseMessage]) -> None:
        """Extend the list by appending elements from the iterable."""
        messages = list(__iterable)
        count = self.llm.get_num_tokens_from_messages(messages)
        while 1:
            all_messages_count = self.llm.get_num_tokens_from_messages(
                self.messages
            )
            if count + all_messages_count > self.max_token_limit:
                self.messages.pop(0)
            else:
                break
        self.messages.extend(messages)

    def append(self, __object: BaseMessage) -> None:
        """Append object to the end."""
        self.extend([__object])


@dataclass
class ThreadChatMessageHistory(BaseChatMessageHistory):
    """Thread chat message history."""

    thread_id: str | UUID
    llm: BaseLanguageModel
    max_token_limit: int = 3000

    @cached_property
    def thread(self) -> Thread:
        """Return the thread."""
        return Thread.get(self.thread_id)

    @cached_property
    def messages(self) -> ChatSlidingWindow:  # type: ignore[override]
        """Return the messages."""
        return ChatSlidingWindow(
            chats=self.thread.chats,
            llm=self.llm,
            max_token_limit=self.max_token_limit,
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the history."""
        prev = None
        for message in messages:
            if isinstance(message, HumanMessage) or (
                isinstance(message, BaseMessage) and message.type == "human"
            ):
                prev = self.thread.add_query(
                    content=str(message.content),
                    commit=True,
                    previous_chat=prev,
                )
            elif isinstance(message, AIMessage) or (
                isinstance(message, BaseMessage) and message.type == "ai"
            ):
                prev = self.thread.add_response(
                    content=str(message.content),
                    commit=True,
                    previous_chat=prev,
                )
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear the messages."""
        self.thread.clear()


def get_history_from_thread(
    thread_id: str | UUID,
    max_token_limit: int,
    user_id: str,
    task: str | None = None,
) -> ThreadChatMessageHistory:
    """Get chat history from thread."""
    from langchain_openai.llms import OpenAI

    return ThreadChatMessageHistory(
        thread_id=thread_id, max_token_limit=max_token_limit, llm=OpenAI()
    )


def fetch_messages_from_thread(
    thread_id: str | UUID, max_token_limit: int
) -> list[BaseMessage]:
    """Fetch messages from thread."""
    from langchain_openai.llms import OpenAI

    history = ThreadChatMessageHistory(
        thread_id=thread_id,
        max_token_limit=max_token_limit,
        llm=OpenAI(),
    )
    return list(history.messages)
