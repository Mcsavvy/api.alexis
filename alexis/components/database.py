"""Database component for Alexis."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import ClassVar, Generic, TypeVar, cast, overload
from uuid import UUID, uuid4

from fastapi import Request, Response
from mongoengine import (  # type: ignore[import]
    DateTimeField,
    Document,
    QuerySet,
    UUIDField,
    connect,
    disconnect,
    document,
)
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Query,
    mapped_column,
    scoped_session,
    sessionmaker,
)
from typing_extensions import Self, deprecated

from alexis import logging
from alexis.config import settings
from alexis.utils import cast_fn, pascal_to_snake, utcnow

Base: type[DeclarativeBase] = declarative_base()
metadata = Base.metadata
ModelT = TypeVar("ModelT", bound="BaseModel")
DocumentT = TypeVar("DocumentT", bound="BaseDocument")


class SQLAlchemy:
    """SQLAlchemy session manager."""

    def __init__(self, db_uri: str):
        """Initialize the SQLAlchemy session manager."""
        self.engine = create_engine(db_uri)
        self._session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_all(self):
        """Create all tables."""
        metadata.create_all(self.engine)

    def drop_all(self):
        """Drop all tables."""
        metadata.drop_all(self.engine)


class ModelError(Exception):
    """Model error."""


class ModelErrorMixin:
    """Model error mixin."""

    class DoesNotExistError(ModelError):
        """Does not exist."""

    class CreateError(ModelError):
        """Error creating."""

    class UpdateError(ModelError):
        """Error updating."""

    class DeleteError(ModelError):
        """Error deleting."""

    def __init_subclass__(cls) -> None:
        """Initialize subclass."""
        for error in (
            cls.DoesNotExistError,
            cls.CreateError,
            cls.UpdateError,
            cls.DeleteError,
        ):
            setattr(
                cls,
                error.__name__,
                type(error.__name__, (error,), {"__doc__": error.__doc__}),
            )


class BaseQuery(Query, Generic[ModelT]):
    """Base query."""

    @property
    def tables(self) -> set[type[BaseModel]]:
        """Get tables."""
        from sqlalchemy.sql.schema import Table

        _tables: set[type[BaseModel]] = set()
        for element in self._raw_columns:
            if isinstance(element, Table):
                table = cast(
                    type[BaseModel],
                    element._annotations["parententity"].class_,
                )
                _tables.add(table)
        return _tables

    @property
    def model(self) -> type[ModelT] | None:
        """Get model class."""
        tables = self.tables
        if len(tables) == 1:
            return tables.pop()
        return None

    def get(self, ident) -> ModelT:
        """Get a model by id."""
        if isinstance(ident, str):
            try:
                ident = UUID(ident)
            except ValueError:
                # maybe it's not a UUID
                pass
        rv = super().get(ident)
        if rv is None:
            if model := self.model:
                raise model.DoesNotExistError
            logging.warning("Multiple models used for one query")
            raise BaseModel.DoesNotExistError
        return rv

    def first(self) -> ModelT | None:
        """Get the first model."""
        return super().first()

    def all(self) -> list[ModelT]:
        """Get all models."""
        return super().all()


class BaseModel(Base, ModelErrorMixin):  # type: ignore[valid-type, misc]
    """Base model for the database."""

    __abstract__ = True
    __table_args__ = {"mysql_charset": "utf8mb4"}

    query: ClassVar[BaseQuery[Self]]

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=utcnow, onupdate=utcnow, nullable=False
    )

    @property
    def uid(self):
        """Return the UUID of the model as a string."""
        return str(self.id)

    def __repr__(self):
        """Return a string representation of the model."""
        return f"<{self.__class__.__name__}: {self.uid[:8]}>"

    def __init_subclass__(cls) -> None:
        """Initialize subclass."""
        super().__init_subclass__()
        if not hasattr(cls, "__tablename__"):
            cls.__tablename__ = pascal_to_snake(cls.__name__)

    @classmethod
    def create(cls, commit=True, **kwargs):
        """Creates model."""
        logging.debug("Creating %s with attributes: %s", cls.__name__, kwargs)
        instance = cls(**kwargs)
        if commit:
            instance.save()  # pragma: no cover
        return instance

    def update(self, commit=True, **kwargs):
        """Updates model."""
        logging.debug("Updating %s with attributes: %s", self, kwargs)
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        if commit:
            self.save()  # pragma: no cover
        return self

    def save(self):
        """Saves model."""
        logging.debug(f"Saving {self.__class__.__name__}...")
        session.add(self)
        session.commit()

    def delete(self, commit=True):
        """Deletes model."""
        logging.debug(f"Deleting {self}...")
        session.delete(self)
        if commit:
            session.commit()

    @classmethod
    @deprecated("Use `.query.get(id)` instead")
    def get(cls, id: UUID | str, throw=True) -> Self:
        """Get model by id."""
        logging.debug(f"Getting {cls.__name__} with id: {id}")
        if isinstance(id, str):
            id = UUID(id)
        instance = session.query(cls).get(id)
        if instance:
            return instance
        raise cls.DoesNotExistError(
            f"{cls.__name__} with id {id} does not exist"
        )


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
    created_at: Mapped[datetime] = DateTimeField(default=utcnow)
    updated_at: Mapped[datetime] = DateTimeField(default=utcnow)

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


db = SQLAlchemy(settings.SQLALCHEMY_DATABASE_URI)
session = scoped_session(db._session, scopefunc=scopefunc)
Base.query = session.query_property(BaseQuery)
connect(db=settings.MONGO_DATABASE, host=settings.MONGO_URI)


async def SessionMiddleware(  # noqa: N802
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    """Session middleware.

    This middleware removes the session after the request is finished.
    """
    response = await call_next(request)
    logging.debug("[middleware] Removing database session...")
    session.remove()
    return response


@asynccontextmanager
async def lifespan(app):
    """Get a session lifespan."""
    try:
        yield session
    finally:
        logging.debug("[lifespan] Removing database session...")
        session.remove()
        logging.debug("[lifespan] Disconnecting from the database...")
        disconnect()
