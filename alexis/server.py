"""Application entry point. This is the main file that starts the server."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes  # type: ignore[import-untyped]

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    """Redirects the root URL to the /docs URL."""
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
