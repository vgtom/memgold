"""ASGI entrypoint for the memgold HTTP API."""

from __future__ import annotations

from fastapi import FastAPI

from memgold.api.routes import router


def create_app() -> FastAPI:
    """Application factory for tests and embedded deployments."""
    application = FastAPI(
        title="memgold",
        version="0.1.0",
        summary="AI memory layer API (scaffolding)",
    )
    application.include_router(router)
    return application


app = create_app()


def run() -> None:
    """CLI entry via ``uv run memgold-api``."""
    import uvicorn

    uvicorn.run(
        "memgold.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
