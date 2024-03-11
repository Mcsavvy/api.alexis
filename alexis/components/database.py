"""Database component for Alexis."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import ClassVar, Generic, TypeVar, cast
from uuid import UUID, uuid4

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
from typing_extensions import Self

from alexis import logging
from alexis.config import settings
from alexis.utils import pascal_to_snake, utcnow

Base: type[DeclarativeBase] = declarative_base()
metadata = Base.metadata
ModelT = TypeVar("ModelT", bound="BaseModel")


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
        return self.id.hex

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


db = SQLAlchemy(settings.SQLALCHEMY_DATABASE_URI)
session = scoped_session(db._session)
Base.query = session.query_property(BaseQuery)


@asynccontextmanager
async def lifespan(app):
    """Get a session lifespan."""
    try:
        logging.debug("Creating database session...")
        yield session
    finally:
        logging.debug("Closing database session...")
        session.remove()
