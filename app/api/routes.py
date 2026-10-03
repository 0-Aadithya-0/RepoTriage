"""
API routes for RepoTriage.

Provides:
- POST /api/v1/analyze — Analyze open issues in a GitHub repository.
- GET  /health         — Health check endpoint.
"""

import logging

from fastapi import APIRouter, Request

from app.models.api import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.services.analyzer import analyze_repository

logger = logging.getLogger(__name__)

# API router with versioned prefix
router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Separate router for health check (no prefix)
health_router = APIRouter(tags=["health"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze GitHub repository issues",
    description=(
        "Fetches the newest open issues from the specified GitHub repository "
        "and analyzes each one using the configured LLM provider. "
        "Returns structured triage results with category, priority, and summary."
    ),
)
async def analyze_issues(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Analyze open issues in a GitHub repository.

    Retrieves dependencies from app state (set during lifespan startup)
    and delegates to the orchestrator.
    """
    state = request.app.state

    logger.info(
        "Received analysis request",
        extra={
            "owner": body.owner,
            "repo": body.repo,
            "limit": body.limit,
        },
    )

    response = await analyze_repository(
        owner=body.owner,
        repo=body.repo,
        limit=body.limit,
        github_client=state.github_client,
        llm_provider=state.llm_provider,
        prompt_config=state.prompt_config,
        max_concurrency=state.settings.llm_max_concurrency,
    )

    return response


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the application health status.",
)
async def health_check() -> HealthResponse:
    """Simple health check endpoint."""
    return HealthResponse(status="ok")
