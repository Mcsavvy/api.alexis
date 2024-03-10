"""initial setup

Revision ID: 3f16f67e18c3
Revises: 
Create Date: 2024-03-10 22:02:35.180449

"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.engine import reflection
from alexis.config import settings
from alexis.utils import cast_fn

# revision identifiers, used by Alembic.
revision: str = "3f16f67e18c3"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def DefaultColumns():
    from alexis.utils import utcnow

    return [
        sa.Column("id", sa.Uuid, primary_key=True, default=uuid4),
        sa.Column("created_at", sa.DateTime, nullable=False, default=utcnow),
        sa.Column(
            "updated_at",
            sa.DateTime,
            nullable=False,
            default=utcnow,
            onupdate=utcnow,
        ),
    ]


def _has_table(table_name):
    """Check if a table exists in the database."""
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    inspector = reflection.Inspector.from_engine(engine)
    tables = inspector.get_table_names()
    return table_name in tables


_original_create_table = op.create_table


@cast_fn(op.create_table)
def create_table(table_name, *args, **kwargs):
    if not _has_table(table_name):
        _original_create_table(table_name, *args, **kwargs)


op.create_table = create_table


def upgrade() -> None:
    from alexis.chat.models import ChatType

    # user table
    op.create_table(
        "user",
        *DefaultColumns(),
        sa.Column("kinde_user", sa.String(50), nullable=False),
        sa.Column("first_name", sa.String(255), nullable=True),
        sa.Column("last_name", sa.String(255), nullable=True),
        sa.Column("email", sa.String(255), nullable=True, unique=True),
        sa.Column("picture", sa.String(255), nullable=True),
    )
    # thread table
    op.create_table(
        "thread",
        *DefaultColumns(),
        sa.Column("title", sa.String(80), nullable=False),
        sa.Column("user_id", sa.Uuid, sa.ForeignKey("user.id"), nullable=False),
        sa.Column("project", sa.Integer, nullable=False),
        sa.Column("closed", sa.Boolean, nullable=False, default=False),
    )
    # chat table
    op.create_table(
        "chat",
        *DefaultColumns(),
        sa.Column(
            "thread_id", sa.Uuid, sa.ForeignKey("thread.id"), nullable=False
        ),
        sa.Column(
            "previous_chat_id", sa.Uuid, sa.ForeignKey("chat.id"), nullable=True
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("cost", sa.Integer, nullable=False, default=0),
        sa.Column("chat_type", sa.Enum(ChatType), nullable=False),
        sa.Column("sent_time", sa.DateTime, nullable=False),
        sa.Column("order", sa.Integer, nullable=False, default=0),
        sa.UniqueConstraint(
            "thread_id", "order", name="unique_order_per_thread"
        ),
    )


def downgrade() -> None:
    op.drop_table("chat")
    op.drop_table("thread")
    op.drop_table("user")
