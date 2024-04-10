"""Alexis chat database models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Union
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from mongoengine import (  # type: ignore[import]
    CASCADE,
    BooleanField,
    DateTimeField,
    EnumField,
    IntField,
    ReferenceField,
    StringField,
)

from alexis import logging
from alexis.auth.models.user import User
from alexis.components.contexts import ContextNotFound, ProjectContext
from alexis.components.database import BaseDocument, BaseDocumentMeta

if TYPE_CHECKING:
    pass


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(tz=timezone.utc)


class ChatType(str, _Enum):
    """Chat type enum."""

    QUERY = "query"
    RESPONSE = "response"


ChatT = Union["Chat", UUID, str]


class Thread(BaseDocument):
    """Thread model based on mongoengine."""

    meta = BaseDocumentMeta | {
        "collection": "threads",
    }
    title: str = StringField(max_length=80, required=True)
    project: int = IntField(required=True)
    user: User = ReferenceField(
        User, required=True, reverse_delete_rule=CASCADE
    )
    closed: bool = BooleanField(default=False)

    def __repr__(self):
        """Get the string representation of the thread."""
        num_chats = len(self.chats)
        return "Thread[{}](user={}, chats={}, closed={})".format(
            self.uid[:6],
            self.user.name,
            num_chats,
            "✔" if self.closed else "✗",
        )

    @property
    def chats(self):
        """Get all chats in the thread."""
        return Chat.objects.filter(thread=self)

    @property
    def cost(self) -> int:
        """Total cost of all messages."""
        return sum(chat.cost for chat in self.chats)

    @property
    def chat_count(self) -> int:
        """Number of messages in the thread."""
        return len(self.chats)

    @property
    def first_chat(self) -> Chat | None:
        """Get the first chat in the thread."""
        return self.chats.filter(previous_chat=None).one()

    @property
    def last_chat(self) -> Chat | None:
        """Get the last chat in the thread."""
        return self.chats.filter(next_chat=None).one()

    @classmethod
    def create(cls, commit=True, **kwargs):
        """Create a thread."""
        if "project" not in kwargs:
            raise cls.CreateError("Project is required")
        project_id = int(kwargs["project"])
        try:
            project = ProjectContext.load(project_id, include_tasks=False)
        except ContextNotFound:
            raise cls.CreateError(f"Project '{project_id}' does not exist")
        if not kwargs.get("title"):
            kwargs["title"] = f"{project.title}"
        return super().create(commit, **kwargs)

    def add_chat(
        self,
        content: str,
        chat_type: ChatType,
        previous_chat: ChatT | None = None,
        cost: int = 0,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a chat to the thread."""
        previous_chat = previous_chat or self.last_chat
        if isinstance(previous_chat, UUID | str):
            previous_chat = Chat.objects.get(previous_chat)
        logging.debug(
            "adding %s after chat %s in thread %s: %s",
            chat_type.value.lower(),
            previous_chat.uid[:6] if previous_chat else "none",
            self.uid[:6],
            content[:20],
        )
        new_chat = Chat.create(
            content=content,
            chat_type=chat_type,
            previous_chat=previous_chat,
            next_chat=None,
            cost=cost,
            thread=self,
            **attrs,
        )
        if commit:
            self.save()
        return new_chat

    def add_query(
        self,
        content: str,
        cost: int = 0,
        previous_chat: ChatT | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a query to the thread."""
        return self.add_chat(
            content=content,
            chat_type=ChatType.QUERY,
            cost=cost,
            previous_chat=previous_chat,
            commit=commit,
            **attrs,
        )

    def add_response(
        self,
        content: str,
        cost: int = 0,
        previous_chat: ChatT | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a response to the thread."""
        return self.add_chat(
            content=content,
            chat_type=ChatType.RESPONSE,
            cost=cost,
            previous_chat=previous_chat,
            commit=commit,
            **attrs,
        )

    def iter_chats(self):
        """Iterate over all chats in the thread."""
        chat = self.first_chat
        while chat:
            yield chat
            chat = chat.next_chat


class Chat(BaseDocument):
    """Chat model based on mongoengine."""

    meta = BaseDocumentMeta | {
        "ordering": ["order"],
        "collection": "chats",
    }

    content: str = StringField(required=True)
    cost: int = IntField(default=0)
    chat_type: ChatType = EnumField(ChatType, required=True)
    sent_time = DateTimeField(default=utcnow, required=True)
    thread: ReferenceField = ReferenceField(
        Thread, required=True, reverse_delete_rule=CASCADE
    )
    next_chat: Chat | None = ReferenceField(
        "self",
        reverse_delete_rule=CASCADE,
        required=False,
        null=True,
        default=None,
    )
    previous_chat: Chat | None = ReferenceField(
        "self",
        reverse_delete_rule=CASCADE,
        required=False,
        null=True,
        default=None,
    )

    def __repr__(self):
        """Get the string representation of the chat."""
        return "{}[{}](user={}, cost={}, thread={}, prev={}, next={})".format(
            self.chat_type.title(),
            self.uid[:8],
            self.thread.user.name,
            self.cost,
            self.thread.uid[:8],
            self.previous_chat.uid[:8] if self.previous_chat else "none",
            self.next_chat.uid[:8] if self.next_chat else "none",
        )

    def to_message(self) -> BaseMessage:
        """Convert the chat to a message."""
        if self.chat_type == ChatType.QUERY:
            return HumanMessage(
                content=self.content,
                id=self.uid,
            )
        return AIMessage(
            id=self.uid,
            content=self.content,
        )

    def compute_token_cost(self, commit=True):
        """Get the token count of the chat."""
        logging.debug("computing token cost for chat %s", self.uid[:8])
        llm = ChatOpenAI()
        message = self.to_message()
        token_count = llm.get_num_tokens_from_messages([message])
        self.cost = token_count
        if commit:
            self.save()

    @classmethod
    def create(cls, commit: bool = True, **kwargs):
        """Create a chat."""
        previous_chat: Chat | None = kwargs.pop("previous_chat", None)
        if previous_chat and previous_chat.next_chat:
            # this is not the last chat in the thread
            # perform doubly-linked list insertion
            next_chat = previous_chat.next_chat
            instance = super().create(False, **kwargs)

            instance.next_chat = next_chat
            next_chat.previous_chat = instance
            next_chat.save()
        else:
            instance = super().create(False, **kwargs)
        instance.previous_chat = previous_chat
        if previous_chat:
            previous_chat.next_chat = instance
            previous_chat.save()
        if instance.content and not instance.cost:
            instance.compute_token_cost(commit)
        elif commit:
            instance.save()
        return instance
