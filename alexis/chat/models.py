"""Alexis chat database models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as _Enum
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from sqlalchemy import Enum, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship

from alexis import logging
from alexis.components import BaseModel, session

if TYPE_CHECKING:
    from alexis.auth.models import User


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(tz=timezone.utc)


class ChatType(str, _Enum):
    """Chat type enum."""

    QUERY = "query"
    RESPONSE = "response"


class Chat(BaseModel):
    """Chat model."""

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    content: Mapped[str] = mapped_column(Text)
    cost: Mapped[int] = mapped_column(default=0)
    chat_type: Mapped[ChatType] = mapped_column(Enum(ChatType))
    thread_id: Mapped[UUID] = mapped_column(ForeignKey("thread.id"))
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


class Thread(BaseModel):  # type: ignore
    """A conversation thread."""

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

        project = kwargs.get("project", None)
        if project is not None:
            if not redis.project_exists(str(project), []):
                raise cls.CreateError(f"Project '{project}' does not exist")
        return super().create(commit, **kwargs)

    def update(self, commit=True, **kwargs):
        """Update the thread."""
        from alexis.components import redis

        project = kwargs.get("project", None)
        if project is not None:
            if not redis.project_exists(str(project), []):
                raise self.UpdateError(f"Project '{project}' does not exist")
        return super().update(commit, **kwargs)


class ThreadMixin:
    """Mixin class for handling threads."""

    @property
    def total_chat_cost(self):
        """Total cost of all messages."""
        return sum(trd.cost for trd in self.threads)  # type: ignore

    def create_thread(
        self, project: int, title: str, closed=False, commit=True
    ):
        """Create a new thread."""
        logging.debug(
            "creating thread for project %d: %r for %s %s",
            project,
            title,
            self.type.value,  # type: ignore
            self.name,  # type: ignore
        )
        return Thread.create(
            project=project,
            title=title,
            user=self,
            closed=closed,
            commit=commit,
        )

    def add_message(
        self,
        content: str,
        chat_type: ChatType,
        cost: int = 0,
        previous_chat: Chat | None = None,
        thread_id: UUID | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a message to the thread."""
        if thread_id is None:
            if previous_chat:
                thread = previous_chat.thread
            else:
                raise RuntimeError("thread_id or previous_chat must be given")
        else:
            try:
                thread = Thread.get(thread_id)
            except Thread.DoesNotExistError:
                raise ValueError("thread_id is invalid")
            if thread.user != self:
                raise ValueError("thread not owned by user")
        return thread.add_chat(
            content=content,
            chat_type=chat_type,
            cost=cost,
            previous_chat=previous_chat,
            commit=commit,
            **attrs,
        )

    def add_query(
        self,
        content: str,
        cost: int = 0,
        previous_chat: Chat | None = None,
        thread_id: UUID | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a query to the thread."""
        return self.add_message(
            content,
            ChatType.QUERY,
            cost,
            previous_chat,
            thread_id,
            commit,
            **attrs,
        )

    def add_response(
        self,
        content: str,
        cost: int = 0,
        previous_chat: Chat | None = None,
        thread_id: UUID | None = None,
        commit=True,
        **attrs,
    ) -> Chat:
        """Add a response to the thread."""
        return self.add_message(
            content,
            ChatType.RESPONSE,
            cost,
            previous_chat,
            thread_id,
            commit,
            **attrs,
        )

    def clear_chats(self, threads: list[Thread]):
        """Clear all messages in specified threads."""
        logging.debug(
            "clearing %d threads for %s %r",
            len(threads),
            self.type.value,  # type: ignore
            self.name,  # type: ignore
        )
        for thread in threads:
            thread.clear()

    def get_active_threads(self) -> list[Thread]:
        """Get all active threads."""
        return Thread.query.filter(
            Thread.user_id == self.id,  # type: ignore
            Thread.closed == False,  # noqa: E712
        ).all()
