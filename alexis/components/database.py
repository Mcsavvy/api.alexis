"""Database component for Alexis."""

from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import Depends, Request, Response
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from typing_extensions import Self

from alexis import logging
from alexis.config import settings
from alexis.utils import LocalProxy, pascal_to_snake, utcnow

Base: type[DeclarativeBase] = declarative_base()
_session_ctx = ContextVar[Session]("session")
__contexts__: list[Session] = []
metadata = Base.metadata


class SQLAlchemy:
    """SQLAlchemy session manager."""

    def __init__(self, db_uri: str):
        """Initialize the SQLAlchemy session manager."""
        self.engine = create_engine(db_uri)
        self._session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    @contextmanager
    def scoped_session(self):
        """Get a new SQLAlchemy session."""
        session = self._session()
        try:
            yield session
        finally:
            session.close()

    def get_or_create_session(self) -> tuple[Session, bool]:
        """Get or create a new session."""
        if __contexts__:
            session = __contexts__[-1]
            return session, False
        session = self._session()
        __contexts__.append(session)
        return session, True

    def execute(self, function, *args, **kwargs):
        """Execute a function using a new session and handle transactions."""
        with self.scoped_session() as db:
            try:
                result = function(db, *args, **kwargs)
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise

    def create_all(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)

    def drop_all(self):
        """Drop all tables."""
        Base.metadata.drop_all(self.engine)


class ModelError(Exception):
    """Model error."""

    def __init__(self, message: str):
        """Initialize the model error."""
        self.message = message
        super().__init__(message)


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


class BaseModel(Base, ModelErrorMixin):  # type: ignore[valid-type, misc]
    """Base model for the database."""

    __abstract__ = True
    __table_args__ = {"mysql_charset": "utf8mb4"}

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


def get_session():
    """SQLAlchemy session dependency."""
    session, created = db.get_or_create_session()
    token = _session_ctx.set(session)
    try:
        yield session
    finally:
        if created:
            __contexts__.remove(session)
        _session_ctx.reset(token)
        session.close()


async def SessionMiddleware(  # noqa: N802
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Session middleware."""
    logging.debug("Loading session middleware...")
    logging.debug("Creating session...")
    session, created = db.get_or_create_session()
    token = _session_ctx.set(session)
    response = await call_next(request)
    logging.debug("Closing session...")
    if created:
        __contexts__.remove(session)
    _session_ctx.reset(token)
    session.close()
    return response


SessionDependency = Depends(get_session)
session: Session = LocalProxy(  # type: ignore[assignment,misc]
    _session_ctx,
    "Working outside of session context.",
    factory=lambda: db.get_or_create_session()[0],
)
