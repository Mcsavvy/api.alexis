"""Database component for Alexis."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import UUID, uuid4

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
from alexis.utils import LocalProxy

Base: type[DeclarativeBase] = declarative_base()
_session_ctx = ContextVar[Session]("session")
session: Session = LocalProxy(  # type: ignore[assignment,misc]
    _session_ctx, "Working outside of session context."
)


class SQLAlchemy:
    """SQLAlchemy session manager."""

    def __init__(self, db_uri: str):
        """Initialize the SQLAlchemy session manager."""
        self.engine = create_engine(db_uri)
        self._session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    @contextmanager
    def session(self):
        """Get a new SQLAlchemy session."""
        session = self._session()
        token = _session_ctx.set(session)
        try:
            yield session
        finally:
            _session_ctx.reset(token)
            session.close()

    def execute(self, function, *args, **kwargs):
        """Execute a function using a new session and handle transactions."""
        with self.session() as db:
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


def pascal_to_snake(name: str) -> str:
    """Convert a pascal case string to snake case."""
    return "".join(
        ["_" + i.lower() if i.isupper() else i for i in name]
    ).lstrip("_")  # noqa: E501


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(tz=timezone.utc)


class BaseModel(Base):  # type: ignore[valid-type, misc]
    """Base model for the database."""

    class DoesNotExistError(Exception):
        """Does not exist."""

    class CreateError(Exception):
        """Error creating."""

    class UpdateError(Exception):
        """Error updating."""

    class DeleteError(Exception):
        """Error deleting."""

    __abstract__ = True
    __table_args__ = {"mysql_charset": "utf8mb4"}

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
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

        cls.DoesNotExistError = type(  # type: ignore[assignment,misc]
            f"{cls.__name__}.DoesNotExist",
            (cls.DoesNotExistError,),
            {"__doc__": f"{cls.__name__} does not exist"},
        )
        cls.CreateError = type(  # type: ignore[assignment,misc]
            f"{cls.__name__}.CreateError",
            (cls.CreateError,),
            {"__doc__": f"Error creating {cls.__name__}"},
        )
        cls.UpdateError = type(  # type: ignore[assignment,misc]
            f"{cls.__name__}.UpdateError",
            (cls.UpdateError,),
            {"__doc__": f"Error updating {cls.__name__}"},
        )
        cls.DeleteError = type(  # type: ignore[assignment,misc]
            f"{cls.__name__}.DeleteError",
            (cls.DeleteError,),
            {"__doc__": f"Error deleting {cls.__name__}"},
        )

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
    def get(cls, id: UUID) -> Self | None:
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
