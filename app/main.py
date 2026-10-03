"""
FastAPI application factory and lifespan handler for RepoTriage.

The lifespan context manager initializes all shared resources at startup
and cleans them up on shutdown:
- Settings loaded from environment / .env
- httpx.AsyncClient for GitHub API calls
- GitHubClient wrapping the HTTP client
- LLMProvider created via the factory
- PromptConfig loaded and validated from prompts.yaml

All resources are stored on app.state and accessed by route handlers.
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import httpx
from fastapi import FastAPI

from app.api.routes import health_router, router
from app.clients.github_client import GitHubClient
from app.core.config import Settings
from app.middleware.error_handlers import register_error_handlers
from app.providers.factory import create_llm_provider
from app.services.prompt_loader import load_prompt_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down application resources.

    Startup:
        1. Load Settings from environment.
        2. Create httpx.AsyncClient for GitHub.
        3. Wrap it in GitHubClient.
        4. Create LLM provider via factory.
        5. Load and validate prompt configuration.
        6. Store all on app.state.

    Shutdown:
        1. Close the httpx.AsyncClient.
    """
    logger.info("Starting RepoTriage application")

    # 1. Settings
    settings = Settings()
    app.state.settings = settings

    # 2. HTTP client for GitHub
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.github_timeout_seconds),
    )

    # 3. GitHub client
    app.state.github_client = GitHubClient(
        http_client=http_client,
        settings=settings,
    )

    # 4. LLM provider
    app.state.llm_provider = create_llm_provider(settings)
    logger.info(
        "LLM provider initialized",
        extra={"provider": settings.llm_provider, "model": settings.llm_model},
    )

    # 5. Prompt configuration
    app.state.prompt_config = load_prompt_config(settings.prompts_file)
    logger.info("Prompt configuration loaded", extra={"file": settings.prompts_file})

    logger.info("RepoTriage startup complete")

    yield

    # Shutdown
    logger.info("Shutting down RepoTriage")
    await http_client.aclose()
    logger.info("HTTP client closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns a fully wired app with lifespan, routes, and error handlers.
    """
    app = FastAPI(
        title="RepoTriage",
        description=(
            "Live GitHub Issue Analyzer — fetches open issues from any "
            "public repository and uses an LLM to classify, prioritize, "
            "and summarize each one."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)
    app.include_router(health_router)

    # Register exception handlers
    register_error_handlers(app)

    return app


# Module-level app instance for uvicorn
app = create_app()
