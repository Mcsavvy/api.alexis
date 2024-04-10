"""Alexis CLI component."""
import click


@click.group(name="alexis")
def cli():
    """Alexis CLI."""
    pass


@cli.command(name="shell")
def shell():
    """Start a IPython shell with the app context."""
    from IPython.terminal.embed import InteractiveShellEmbed

    from alexis.app import create_app
    from alexis.components import redis
    from alexis.components.contexts import ProjectContext, TaskContext
    from alexis.components.storage import default_storage
    from alexis.config import settings
    from alexis.logging import get_logger
    from alexis.models import (
        Chat,
        ChatType,
        Thread,
        User,
    )

    app = create_app()
    logger = get_logger()

    shell = InteractiveShellEmbed()
    local_ns = {
        "app": app,
        "User": User,
        "Chat": Chat,
        "ChatType": ChatType,
        "Thread": Thread,
        "redis": redis,
        "settings": settings,
        "logger": logger,
        "storage": default_storage,
        "ProjectContext": ProjectContext,
        "TaskContext": TaskContext,
    }
    shell(local_ns=local_ns)
