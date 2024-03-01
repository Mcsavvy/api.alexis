"""Alexis App."""

from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.applications import BaseHTTPMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from langserve import add_routes  # type: ignore[import-untyped]

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
        kwargs["dependencies"] = (
            kwargs.get("dependencies", []) + self._get_global_dependencies()
        )
        super().__init__(**kwargs)

    def _get_global_dependencies(self):
        """Get the global dependencies."""
        return [
            load_entry_point(dep)
            for dep in self.settings.get("DEPENDENCIES", [])
        ]

    def _load_middlewares(self):
        """Get the middlewares."""
        for middleware in self.settings.get("MIDDLEWARES", []):
            logging.debug(f"Loading middleware: '{middleware}'")
            m = load_entry_point(middleware)
            # prevent adding the same middleware twice
            for _m in self.user_middleware:
                if (
                    isinstance(_m, BaseHTTPMiddleware)
                    and _m.kwargs["dispatch"] == m
                ):
                    break
            else:
                self.add_middleware(BaseHTTPMiddleware, dispatch=m)

    def _load_routes(self):
        """Load the routes."""
        for router in self.settings.ROUTERS:
            logging.debug(f"Loading router: {router}")
            r = load_entry_point(router, APIRouter)
            self.include_router(r)

    def _load_chains(self):
        """Load the chains."""
        from alexis.chat.chains import AlexisChain, ContextInput, user_injection

        add_routes(
            self,
            AlexisChain,
            path="/alexis",
            input_type=ContextInput,
            output_type=str,
            per_req_config_modifier=user_injection,
        )

    def _enable_cors(self):
        """Enable CORS."""
        from fastapi.middleware.cors import CORSMiddleware

        self.add_middleware(
            CORSMiddleware,
            allow_origins=settings.get("CORS_ALLOWED_ORIGINS", []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def __repr__(self):
        """Get the string representation of the Alexis App."""
        return f"<AlexisApp[{self.settings.env}]: {self.title} v{self.version}>"


async def redirect_root_to_docs():
    """Redirects the root URL to the /docs URL."""
    return RedirectResponse("/docs")


async def api_documentation(request: Request):
    """API documentation."""
    return HTMLResponse(
        (Path(__file__).parent / "templates" / "elements.html").read_text()
    )


def create_app() -> AlexisApp:
    """Create the Alexis App."""
    app = AlexisApp(docs_url=None, redoc_url=None)
    app.add_api_route("/", redirect_root_to_docs, include_in_schema=False)
    app.add_api_route("/docs", api_documentation, include_in_schema=False)
    app._load_routes()
    app._load_chains()
    app._enable_cors()
    app._load_middlewares()
    return app
