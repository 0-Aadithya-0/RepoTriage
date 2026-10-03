"""
Tests for the FastAPI API routes.

Uses httpx.AsyncClient with the FastAPI app directly (ASGITransport),
injecting mock dependencies into app.state to avoid real API calls.

Covers:
- POST /api/v1/analyze: successful analysis, validation errors, empty repo.
- GET /health: returns ok.
- Exception handler integration: domain errors produce correct HTTP responses.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import health_router, router
from app.core.exceptions import (
    LLMTimeoutError,
    RepositoryNotFoundError,
)
from app.middleware.error_handlers import register_error_handlers
from app.models.analysis import IssueAnalysis, IssueCategory, IssuePriority
from app.models.api import AnalyzeResponse
from app.models.github import GitHubIssue
from app.models.prompts import FewShotExample, PromptConfig
from app.services.analyzer import analyze_repository


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_prompt_config() -> PromptConfig:
    return PromptConfig(
        version="1.0",
        system_role="You are an AI.",
        task="Classify the issue.",
        classification_rules="Use Bug for bugs.",
        priority_rules="High for urgent.",
        summary_rules="One sentence.",
        few_shot_examples=[
            FewShotExample(
                title="Test issue",
                body="Test body for this example issue.",
                category=IssueCategory.BUG,
                priority=IssuePriority.HIGH,
                summary="Test summary for this example issue.",
            ),
        ],
    )


def _make_test_app(
    *,
    github_client: MagicMock | None = None,
    llm_provider: MagicMock | None = None,
) -> FastAPI:
    """Create a test FastAPI app with mocked dependencies in app.state."""
    app = FastAPI()
    app.include_router(router)
    app.include_router(health_router)
    register_error_handlers(app)

    # Directly set the state for tests
    app.state.settings = MagicMock()
    app.state.settings.llm_max_concurrency = 2
    app.state.github_client = github_client or MagicMock()
    app.state.llm_provider = llm_provider or MagicMock()
    app.state.prompt_config = _make_prompt_config()

    return app


def _make_issue(number: int) -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=f"Issue #{number}",
        body=f"Body for issue #{number}.",
        html_url=f"https://github.com/owner/repo/issues/{number}",
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    async def test_health_returns_ok(self) -> None:
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Analyze endpoint — success
# ---------------------------------------------------------------------------


class TestAnalyzeSuccess:

    async def test_successful_analysis(self) -> None:
        """POST /api/v1/analyze returns analyzed issues on success."""
        issues = [_make_issue(1), _make_issue(2)]
        analysis = IssueAnalysis(
            category=IssueCategory.BUG,
            priority_level=IssuePriority.MEDIUM,
            tldr_summary="This is a test analysis summary.",
        )

        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(return_value=issues)

        mock_llm = MagicMock()
        mock_llm.analyze_issue = AsyncMock(return_value=analysis)

        app = _make_test_app(github_client=mock_github, llm_provider=mock_llm)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "tiangolo", "repo": "fastapi", "limit": 2},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["repository"] == "tiangolo/fastapi"
        assert data["analyzed_count"] == 2
        assert data["failed_count"] == 0
        assert len(data["issues"]) == 2
        assert data["issues"][0]["issue_number"] == 1
        assert data["issues"][0]["category"] == "Bug"

    async def test_empty_repository(self) -> None:
        """An empty repo returns zero counts."""
        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(return_value=[])

        app = _make_test_app(github_client=mock_github)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "empty-repo"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["analyzed_count"] == 0
        assert data["failed_count"] == 0
        assert data["issues"] == []

    async def test_default_limit_is_10(self) -> None:
        """When limit is not provided, it defaults to 10."""
        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(return_value=[])

        app = _make_test_app(github_client=mock_github)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "repo"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["requested_limit"] == 10
        mock_github.fetch_issues.assert_called_once_with("owner", "repo", limit=10)


# ---------------------------------------------------------------------------
# Analyze endpoint — validation errors
# ---------------------------------------------------------------------------


class TestAnalyzeValidation:

    async def test_missing_owner(self) -> None:
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"repo": "fastapi"},
            )
        assert resp.status_code == 422

    async def test_invalid_owner_format(self) -> None:
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "-invalid", "repo": "fastapi"},
            )
        assert resp.status_code == 422

    async def test_limit_too_high(self) -> None:
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "repo", "limit": 999},
            )
        assert resp.status_code == 422

    async def test_limit_zero(self) -> None:
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "repo", "limit": 0},
            )
        assert resp.status_code == 422

    async def test_empty_body(self) -> None:
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/analyze", content=b"")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Analyze endpoint — error handler integration
# ---------------------------------------------------------------------------


class TestAnalyzeErrorIntegration:

    async def test_repo_not_found_returns_404(self) -> None:
        """RepositoryNotFoundError from GitHub maps to HTTP 404."""
        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(
            side_effect=RepositoryNotFoundError("Not found")
        )

        app = _make_test_app(github_client=mock_github)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "nonexistent"},
            )

        assert resp.status_code == 404
        data = resp.json()
        assert data["error"] == "repository_not_found"

    async def test_path_traversal_in_repo_blocked(self) -> None:
        """A repo name with '..' is rejected by validation."""
        app = _make_test_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "../../etc"},
            )
        assert resp.status_code == 422

    async def test_llm_rate_limit_returns_429(self) -> None:
        """LLMRateLimitError when GitHub is ok but LLM fails globally."""
        from app.core.exceptions import LLMRateLimitError

        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(
            side_effect=LLMRateLimitError("Too many LLM requests", retry_after=45)
        )
        app = _make_test_app(github_client=mock_github)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "repo"},
            )
        assert resp.status_code == 429
        assert resp.json()["error"] == "llm_rate_limit"
        assert resp.headers.get("retry-after") == "45"

    async def test_partial_failure_response_shape(self) -> None:
        """A mixed success/failure batch returns both issues and failures fields."""
        from app.core.exceptions import LLMServerError

        issues = [_make_issue(1), _make_issue(2)]
        call_count = 0

        async def _mixed(issue: GitHubIssue, _pc):
            nonlocal call_count
            call_count += 1
            if issue.number == 2:
                raise LLMServerError("Provider down")
            return IssueAnalysis(
                category=IssueCategory.BUG,
                priority_level=IssuePriority.HIGH,
                tldr_summary="Issue #1 is a high priority bug.",
            )

        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(return_value=issues)
        mock_llm = MagicMock()
        mock_llm.analyze_issue = _mixed

        app = _make_test_app(github_client=mock_github, llm_provider=mock_llm)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/analyze",
                json={"owner": "owner", "repo": "repo", "limit": 2},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["analyzed_count"] == 1
        assert data["failed_count"] == 1
        assert data["issues"][0]["issue_number"] == 1
        assert data["failures"][0]["issue_number"] == 2
        assert "Provider down" in data["failures"][0]["error"]
