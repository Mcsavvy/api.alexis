"""Alexis chat database models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Union, cast
from uuid import UUID, uuid4

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
from sqlalchemy import Enum, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship

from alexis import logging
from alexis.auth.models.user import MUser
from alexis.components import BaseModel, session
from alexis.components.database import BaseDocument, BaseDocumentMeta

if TYPE_CHECKING:
    from alexis.auth.models import User


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(tz=timezone.utc)


class ChatType(str, _Enum):
    """Chat type enum."""

    QUERY = "query"
    RESPONSE = "response"


MChatT = Union["MChat", UUID, str]


class Chat(BaseModel):
    """Chat model."""

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    content: Mapped[str] = mapped_column(Text)
    cost: Mapped[int] = mapped_column(default=0)
    chat_type: Mapped[ChatType] = mapped_column(Enum(ChatType))
    thread_id: Mapped[UUID] = mapped_column(
        ForeignKey("thread.id"), nullable=False
    )
    previous_chat_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat.id"))
    previous_chat: Mapped[Chat | None] = relationship(  # type: ignore[assignment]
        remote_side=[id],
        backref=backref("next_chat", uselist=False, cascade="all, delete"),
        uselist=False,
        single_parent=True,
        foreign_keys=[previous_chat_id],
    )
    sent_time: Mapped[datetime] = mapped_column(default=utcnow)
    order: Mapped[int] = mapped_column(default=0)
    thread: Mapped[Thread] = relationship(  # type: ignore[assignment]
        back_populates="chats",
        lazy=True,
        single_parent=True,
        foreign_keys=[thread_id],
    )

    __table_args__ = (  # type: ignore[assignment]
        UniqueConstraint("thread_id", "order", name="unique_order_per_thread"),
    )

    def __repr__(self):
        """Get the string representation of the chat."""
        return "{}[{}](user={}, cost={}, thread={}, prev={}, next={})".format(
            self.chat_type.value.title(),
            self.id.hex[:6],
            self.thread.user.name,
            self.cost,
            self.thread_id.hex[:6],
            self.previous_chat_id.hex[:6] if self.previous_chat_id else "none",
            self.next_chat_id.hex[:6] if self.next_chat_id else "none",
        )

    @property
    def next_chat_id(self) -> UUID | None:
        """Get the next chat's id."""
        if self.next_chat:
            return self.next_chat.id
        return None

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
    def create(cls, commit=True, **kwargs):
        """Create a chat."""
        kwargs.setdefault("id", uuid4())
        instance = super().create(False, **kwargs)
        if instance.content and not instance.cost:
            instance.compute_token_cost(commit)
        return instance


class MThread(BaseDocument):
    """Thread model based on mongoengine."""

    meta = BaseDocumentMeta | {
        "collection": "threads",
    }
    title: str = StringField(max_length=80, required=True)
    project: int = IntField(required=True)
    user: MUser = ReferenceField(
        MUser, required=True, reverse_delete_rule=CASCADE
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
        return MChat.objects.filter(thread=self)

    @property
    def cost(self) -> int:
        """Total cost of all messages."""
        return sum(chat.cost for chat in self.chats)

    @property
    def chat_count(self) -> int:
        """Number of messages in the thread."""
        return len(self.chats)

    @property
    def first_chat(self) -> MChat | None:
        """Get the first chat in the thread."""
        return self.chats.filter(previous_chat=None).one()

    @property
    def last_chat(self) -> MChat | None:
        """Get the last chat in the thread."""
        return self.chats.filter(next_chat=None).one()

    @classmethod
    def create(cls, commit=True, **kwargs):
        """Create a thread."""
        from alexis.components import redis

        if "project" not in kwargs:
            raise cls.CreateError("Project is required")
        project_id = kwargs["project"]
        project = redis.get_project(str(project_id))
        if project is None:
            raise cls.CreateError(f"Project '{project_id}' does not exist")
        if not kwargs.get("title"):
            kwargs["title"] = f"{project.title}"
        return super().create(commit, **kwargs)

    def add_chat(
        self,
        content: str,
        chat_type: ChatType,
        previous_chat: MChatT | None = None,
        cost: int = 0,
        commit=True,
        **attrs,
    ) -> MChat:
        """Add a chat to the thread."""
        previous_chat = previous_chat or self.last_chat
        if isinstance(previous_chat, UUID | str):
            previous_chat = MChat.objects.get(previous_chat)
        logging.debug(
            "adding %s after chat %s in thread %s: %s",
            chat_type.value.lower(),
            previous_chat.uid[:6] if previous_chat else "none",
            self.uid[:6],
            content[:20],
        )
        new_chat = MChat.create(
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
        previous_chat: MChatT | None = None,
        commit=True,
        **attrs,
    ) -> MChat:
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
        previous_chat: MChatT | None = None,
        commit=True,
        **attrs,
    ) -> MChat:
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


class MChat(BaseDocument):
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
        MThread, required=True, reverse_delete_rule=CASCADE
    )
    next_chat: MChat | None = ReferenceField(
        "self",
        reverse_delete_rule=CASCADE,
        required=False,
        null=True,
        default=None,
    )
    previous_chat: MChat | None = ReferenceField(
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
        previous_chat: MChat | None = kwargs.pop("previous_chat", None)
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


class Thread(BaseModel):  # type: ignore
    """A conversation thread."""

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, nullable=False
    )
    title: Mapped[str] = mapped_column(String(80))
    project: Mapped[int] = mapped_column(nullable=False)
    chats: Mapped[list[Chat]] = relationship(  # type: ignore[assignment]
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="Chat.order",
    )
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"))
    user: Mapped[User] = relationship(  # type: ignore[assignment]
        back_populates="threads",
        lazy=True,
        single_parent=True,
        foreign_keys=[user_id],
    )
    closed: Mapped[bool] = mapped_column(default=False)

    def __repr__(self):
        """Get the string representation of the thread."""
        num_chats = len(self.chats)  # type: ignore
        return "Thread[{}](user={}, chats={}, closed={})".format(
            self.id.hex[:6],
            self.user.name,
            num_chats,
            "✔" if self.closed else "✗",
        )

    @property
    def cost(self) -> int:
        """Total cost of all messages."""
        return sum(chat.cost for chat in self.chats)  # type: ignore

    @property
    def chat_count(self) -> int:
        """Number of messages in the thread."""
        return len(self.chats)  # type: ignore

    @property
    def last_chat(self) -> Chat | None:
        """Get the last chat in the thread."""
        return (
            session.query(Chat)
            .filter(Chat.thread_id == self.id, ~Chat.next_chat.has())
            .order_by(Chat.order.desc())
            .first()
        )  # type: ignore

    def add_chat(
        self,
        content: str,
        chat_type: ChatType,
        cost: int = 0,
        previous_chat: Chat | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a chat to the thread."""
        logging.debug(
            "adding %s to thread %s: %s",
            chat_type.value.lower(),
            self.id.hex[:6],
            content[:20],
        )
        previous_chat = previous_chat or self.last_chat
        # TODO: if `previous_chat` already has a `next_chat` then
        #       perform doubly-linked list insertion

        if previous_chat and previous_chat.thread_id != self.id:
            raise ValueError("previous_chat not in same thread")

        order = previous_chat.order + 1 if previous_chat else 0
        new_chat = Chat.create(
            commit,
            content=content,
            chat_type=chat_type,
            cost=cost,
            thread=self,
            previous_chat=previous_chat,
            order=order,
            **attrs,
        )
        session.flush([new_chat, self])
        session.refresh(self)
        return new_chat

    def add_query(
        self,
        content: str,
        cost: int = 0,
        previous_chat: Chat | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a query to the thread."""
        return self.add_chat(
            content, ChatType.QUERY, cost, previous_chat, commit, **attrs
        )

    def add_response(
        self,
        content: str,
        cost: int = 0,
        previous_chat: Chat | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a response to the thread."""
        return self.add_chat(
            content, ChatType.RESPONSE, cost, previous_chat, commit, **attrs
        )

    def close(self):
        """Close the thread."""
        logging.debug("closing thread %s", self.id.hex[:6])
        self.update(closed=True)

    def clear(self):
        """Clear all messages in the thread."""
        logging.debug(
            "clearing %d chats from thread %s",
            len(self.chats),
            self.id.hex[:6],
        )
        for chat in (
            session.query(Chat)
            .filter(
                Chat.thread_id == self.id,
                Chat.previous_chat_id == None,  # noqa: E711
            )
            .all()
        ):
            cast(Chat, chat).delete()

    @classmethod
    def create(cls, commit=True, **kwargs):
        """Create a thread."""
        from alexis.components import redis

        if "project" not in kwargs:
            raise cls.CreateError("Project is required")
        project_id = kwargs["project"]
        project = redis.get_project(str(project_id))
        if project is None:
            raise cls.CreateError(f"Project '{project_id}' does not exist")
        if not kwargs.get("title"):
            kwargs["title"] = f"{project.title}"
        return super().create(commit, **kwargs)

    def update(self, commit=True, **kwargs):
        """Update the thread."""
        from alexis.components import redis

        project = kwargs.get("project", None)
        if project is not None:
            if not redis.project_exists(str(project), []):
                raise self.UpdateError(f"Project '{project}' does not exist")
        return super().update(commit, **kwargs)
