"""Database component for Alexis."""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import ClassVar, Generic, TypeVar, overload
from uuid import UUID, uuid4

from mongoengine import (  # type: ignore[import]
    DateTimeField,
    Document,
    QuerySet,
    UUIDField,
    connect,
    disconnect,
    document,
)
from typing_extensions import Self

from alexis import logging
from alexis.config import settings
from alexis.utils import cast_fn, utcnow

DocumentT = TypeVar("DocumentT", bound="BaseDocument")


def document_query_set(
    cls: type[DocumentT], queryset: DocumentQuerySet[DocumentT]
) -> DocumentQuerySet[DocumentT]:
    """Document query set."""
    return queryset.order_by("-created_at")


class DocumentError(Exception):
    """Document error."""


class DocumentErrorMixin:
    """Mixins for common erros in documents."""

    class DoesNotExistError(DocumentError):
        """Document does not exist."""

    class CreateError(DocumentError):
        """Error creating document."""

    class UpdateError(DocumentError):
        """Error updating document."""

    class DeleteError(DocumentError):
        """Error deleting document."""


class DocumentQuerySet(QuerySet, Generic[DocumentT]):
    """Base query set."""

    @overload
    def __getitem__(self, key: int) -> DocumentT:
        ...

    @overload
    def __getitem__(self, key: slice) -> list[DocumentT]:
        ...

    def __getitem__(self, key: int | slice) -> DocumentT | list[DocumentT]:
        """Get item."""
        return super().__getitem__(key)

    def __iter__(self) -> Iterable[DocumentT]:
        """Iterate over the documents."""
        return super().__iter__()

    def first(self) -> DocumentT | None:
        """Get the first document."""
        return super().first()

    def order_by(self, *keys, __raw__=None) -> Self:
        """Order by."""
        return super().order_by(*keys, __raw__=__raw__)

    def get(self, id: UUID | str) -> DocumentT:
        """Get a document."""
        if isinstance(id, str):
            id = UUID(id)
        return super().get(id=id)

    def create(self, **kwargs) -> DocumentT:
        """Create a document."""
        return super().create(**kwargs)

    def with_id(self, id: UUID) -> DocumentT:
        """Get a document by id."""
        return super().with_id(id)

    def filter(self, *q_objs, **query) -> Self:
        """Filter documents."""
        return super().filter(*q_objs, **query)

    def one(self) -> DocumentT | None:
        """Get one document."""
        if self.count() < 2:
            return self.first()
        raise self._document.MultipleObjectsReturned(
            "2 or more items returned, instead of 1"
        )


BaseDocumentMeta = {
    "id_field": "id",
    "queryset_class": DocumentQuerySet,
    "ordering": ["-created_at"],
}


class DocumentMetaClass(document.TopLevelDocumentMetaclass):
    """Base metaclass for MongoDB."""

    def __new__(cls, name, bases, attrs):
        """Create a new class."""
        for error in DocumentError.__subclasses__():
            attrs[error.__name__] = type(
                error.__name__, (error,), {"__doc__": error.__doc__}
            )
        new_class = super().__new__(cls, name, bases, attrs)
        return new_class


class BaseDocument(Document, DocumentErrorMixin, metaclass=DocumentMetaClass):
    """Base document for MongoDB."""

    meta = BaseDocumentMeta | {"abstract": True}

    id: UUID = UUIDField(primary_key=True, default=uuid4)
    created_at: datetime = DateTimeField(default=utcnow)
    updated_at: datetime = DateTimeField(default=utcnow)

    @property
    def uid(self):
        """Return the UUID of the model as a string."""
        self.CreateError
        return str(self.id)

    def __repr__(self):
        """Return a string representation of the model."""
        return f"<{self.__class__.__name__}: {self.uid[:8]}>"

    objects: ClassVar[DocumentQuerySet[Self]]

    def update(self, commit=True, **kwargs):
        """Updates model."""
        logging.debug("Updating %s with attributes: %s", self, kwargs)
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        if commit:
            self.save()

    @cast_fn(Document.save)
    def save(self, *args, **kwargs):
        """Saves model."""
        logging.debug(f"Saving {self.__class__.__name__}...")
        super().save(*args, **kwargs)

    @cast_fn(Document.delete)
    def delete(self, *args, **kwargs):
        """Deletes model."""
        logging.debug(f"Deleting {self}...")
        super().delete(*args, **kwargs)

    @classmethod
    def create(cls, commit=True, **kwargs):
        """Creates model."""
        logging.debug("Creating %s with attributes: %s", cls.__name__, kwargs)
        instance = cls(**kwargs)
        if commit:
            instance.save(force_insert=True)
        return instance


__context = "app"


def scopefunc():
    """Returns the current app context."""
    return __context


connect(db=settings.MONGO_DATABASE, host=settings.MONGO_URI)


@asynccontextmanager
async def lifespan(app):
    """Get app lifespan."""
    try:
        yield
    finally:
        logging.debug("[lifespan] Disconnecting from the database...")
        disconnect()
