"""User model."""
from __future__ import annotations

from uuid import UUID

import jwt
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import Self

from alexis.chat.models import Thread, ThreadMixin
from alexis.components.database import BaseModel, session


class User(BaseModel, ThreadMixin):
    """User model."""

    kinde_user: Mapped[str] = mapped_column(String(50), nullable=False)
    first_name: Mapped[str] = mapped_column(String(255), nullable=True)
    last_name: Mapped[str] = mapped_column(String(255), nullable=True)
    email: Mapped[str] = mapped_column(String(255), nullable=True, unique=True)
    picture: Mapped[str] = mapped_column(String(255), nullable=True)
    threads: Mapped[list[Thread]] = relationship(  # type: ignore[assignment]
        back_populates="user",
        lazy=True,
        cascade="all, delete-orphan",
    )

    @property
    def name(self) -> str:
        """Return the user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        return "Anonymous"

    def __repr__(self):
        """Return a string representation of the user."""
        return f"<User[{self.uid[:8]}]: {self.name}>"

    @classmethod
    def create(cls, commit=True, **kwargs):
        """Create a user."""
        if session.query(cls).filter(cls.email == kwargs["email"]).first():
            raise cls.CreateError("Email is taken.")
        return super().create(commit, **kwargs)

    def create_token(self) -> str:
        """Create a token for the user."""
        if self.kinde_user:
            sub = self.kinde_user
        else:
            sub = self.uid
        return jwt.encode({"sub": sub}, "secret", algorithm="HS256")

    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode a token."""
        return jwt.decode(token, "secret", algorithms=["HS256"])

    @classmethod
    def get_by_token(cls, token: str) -> Self | None:
        """Get a user by a token."""
        data = cls.decode_token(token)
        sub = data["sub"]
        user = session.query(cls).filter(cls.kinde_user == sub).first()
        if user:
            return user
        try:
            # Try to parse the sub as a UUID
            UUID(sub)
        except ValueError:
            return None
        try:
            return cls.get(sub)
        except cls.DoesNotExistError:
            return None
