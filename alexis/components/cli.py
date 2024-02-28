"""Alexis CLI component."""
import click


@click.group(name="alexis")
def cli():
    """Alexis CLI."""
    pass


@cli.group(name="db")
def db():
    """Database commands."""
    pass


@db.command(name="create-all")
def create_all():
    """Create all tables."""
    from alexis.auth import models  # noqa
    from alexis.components.database import db

    db.create_all()
    click.echo("All tables created.")


@db.command(name="drop-all")
def drop_all():
    """Drop all tables."""
    from alexis.auth import models  # noqa
    from alexis.components.database import db

    db.drop_all()
    click.echo("All tables dropped.")
