"""User model."""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import jwt
from mongoengine import StringField  # type: ignore[import]
from typing_extensions import Self

from alexis import logging
from alexis.components.database import (
    BaseDocument,
    BaseDocumentMeta,
)
from alexis.config import settings

if TYPE_CHECKING:
    pass


class User(BaseDocument):
    """User models based on mongoengine."""

    meta = BaseDocumentMeta | {"collection": "users"}

    kinde_user: str = StringField(max_length=50, required=True)
    first_name: str = StringField(max_length=255, required=False)
    last_name: str = StringField(max_length=255, required=False)
    email: str = StringField(max_length=255, required=False, unique=True)
    picture: str = StringField(max_length=255, required=False)

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
        if cls.objects.filter(email=kwargs["email"]).first():
            raise cls.CreateError("Email is taken.")
        return super().create(commit, **kwargs)

    def create_token(self) -> str:
        """Create a token for the user."""
        if self.kinde_user:
            sub = self.kinde_user
        else:
            sub = self.uid
        return jwt.encode({"sub": sub}, settings.SECRET_KEY, algorithm="HS256")

    @classmethod
    def decode_token(cls, token: str) -> dict:
        """Decode a token."""
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=["HS256"]
            )
        except jwt.InvalidSignatureError:
            payload = jwt.decode(token, "secret", algorithms=["HS256"])
            logging.info("User '%s' used an old token", payload["sub"][:8])
        return payload

    @classmethod
    def get_by_token(cls, token: str) -> Self | None:
        """Get a user by a token."""
        data = cls.decode_token(token)
        sub = data["sub"]
        user = cls.objects.filter(kinde_user=sub).first()
        if user:
            return user
        try:
            # Try to parse the sub as a UUID
            UUID(sub)
        except ValueError:
            return None
        try:
            return cls.objects.get(sub)
        except cls.DoesNotExist:
            return None
