"""Alexis App."""

from pathlib import Path

from fastapi import APIRouter, FastAPI

from alexis import logging
from alexis.config import settings
from alexis.utils import cast_fn, load_entry_point

DESCRIPTION = """The ALX Learners Copilot."""


class AlexisApp(FastAPI):
    """Alexis App."""

    @cast_fn(FastAPI.__init__)
    def __init__(self, **kwargs):
        """Initialize the Alexis App."""
        self.settings = settings
        version = (Path(__file__).parent / "VERSION").read_text().strip()
        description = f"{DESCRIPTION}"
        title = "Alexis"
        kwargs.setdefault("title", title)
        kwargs.setdefault("description", description)
        kwargs.setdefault("version", version)
        super().__init__(**kwargs)
        self._load_routes()

    def _load_routes(self):
        """Load the routes."""
        for router in self.settings.ROUTERS:
            logging.debug(f"Loading router: {router}")
            r = load_entry_point(router, APIRouter)
            self.include_router(r)

    def __repr__(self):
        """Get the string representation of the Alexis App."""
        return f"<AlexisApp[{self.settings.env}]: {self.title} v{self.version}>"


def create_app() -> AlexisApp:
    """Create the Alexis App."""
    return AlexisApp()
