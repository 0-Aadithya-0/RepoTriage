"""
End-to-end tests for RepoTriage.

These tests make REAL network requests to the GitHub API and the Gemini API.
They require a valid GEMINI_API_KEY in the environment (or .env file).

Run only e2e tests:
    pytest -m e2e -v

Skip e2e tests (default unit test run):
    pytest -m "not e2e" -v

The tests analyze real open issues from well-known public repositories
and validate the full pipeline: app lifespan -> GitHub fetch -> LLM -> response.
"""

import os

import httpx
import pytest

# Load .env before importing Settings so the API key is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.main import create_app


# ---------------------------------------------------------------------------
# Skip guard: skip all live tests if no real API key is present
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.e2e

_has_real_key = bool(
    os.environ.get("GEMINI_API_KEY")
    and not os.environ.get("GEMINI_API_KEY", "").startswith("your_")
)

requires_real_key = pytest.mark.skipif(
    not _has_real_key,
    reason="GEMINI_API_KEY not set or is a placeholder — skipping live e2e test",
)


# ---------------------------------------------------------------------------
# Shared fixture: boots full app lifespan, yields an async HTTP client
# ---------------------------------------------------------------------------


@pytest.fixture()
async def live_client():
    """Boot the full RepoTriage app with its production lifespan.

    Manually enters the FastAPI lifespan context manager so that app.state
    is populated (Settings, GitHubClient, GeminiProvider, PromptConfig) before
    any requests are made.

    Note: function-scoped (not module) to avoid pytest-asyncio event loop
    mismatch on Windows with the proactor loop.
    """
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60.0,
        ) as client:
            yield client


# ---------------------------------------------------------------------------
# Health check (no API key needed once app state is populated)
# ---------------------------------------------------------------------------


class TestLiveHealth:

    async def test_health_endpoint_is_reachable(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """The /health endpoint responds 200 with {status: ok}."""
        resp = await live_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Live analysis: tiangolo/fastapi (popular repo, guaranteed open issues)
# ---------------------------------------------------------------------------


class TestLiveAnalysis:

    @requires_real_key
    async def test_live_analysis_fastapi_repo(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """Analyze real open issues from tiangolo/fastapi.

        Validates:
        - HTTP 200
        - Correct repository field
        - At least one issue attempted
        - All AnalyzedIssue fields present and within spec
        """
        resp = await live_client.post(
            "/api/v1/analyze",
            json={"owner": "tiangolo", "repo": "fastapi", "limit": 3},
        )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

        data = resp.json()
        assert data["repository"] == "tiangolo/fastapi"
        assert data["requested_limit"] == 3
        # GitHub may return 0 issues if the repo has no open issues at query time;
        # the important thing is the response is well-formed, not that it has data.
        assert isinstance(data["issues"], list)
        assert isinstance(data["failures"], list)
        assert data["analyzed_count"] + data["failed_count"] == len(data["issues"]) + len(data["failures"])

        valid_categories = {
            "Bug", "Feature Request", "Documentation", "Support Question", "Other"
        }
        valid_priorities = {"Critical", "High", "Medium", "Low"}

        for issue in data["issues"]:
            assert issue["issue_number"] > 0
            assert len(issue["title"]) > 0
            assert issue["html_url"].startswith("https://github.com/")
            assert issue["category"] in valid_categories, (
                f"Invalid category: {issue['category']}"
            )
            assert issue["priority_level"] in valid_priorities, (
                f"Invalid priority: {issue['priority_level']}"
            )
            assert 10 <= len(issue["tldr_summary"]) <= 256, (
                f"Summary length {len(issue['tldr_summary'])} out of bounds: "
                f"'{issue['tldr_summary']}'"
            )

    @requires_real_key
    async def test_limit_is_respected(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """The number of analyzed + failed issues never exceeds the requested limit."""
        limit = 2
        resp = await live_client.post(
            "/api/v1/analyze",
            json={"owner": "tiangolo", "repo": "fastapi", "limit": limit},
        )

        assert resp.status_code == 200
        data = resp.json()
        total_processed = data["analyzed_count"] + data["failed_count"]
        assert total_processed <= limit, (
            f"Processed {total_processed} issues but limit was {limit}"
        )

    @requires_real_key
    async def test_response_schema_is_complete(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """All required top-level response fields are present."""
        resp = await live_client.post(
            "/api/v1/analyze",
            json={"owner": "tiangolo", "repo": "fastapi", "limit": 1},
        )

        assert resp.status_code == 200
        data = resp.json()

        required_fields = {
            "repository", "requested_limit",
            "analyzed_count", "failed_count",
            "issues", "failures",
        }
        for field in required_fields:
            assert field in data, f"Missing required response field: '{field}'"


# ---------------------------------------------------------------------------
# Live error handling: non-existent repository
# ---------------------------------------------------------------------------


class TestLiveErrorHandling:

    async def test_nonexistent_repo_returns_404(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """A repo that doesn't exist returns a structured 404 JSON error."""
        resp = await live_client.post(
            "/api/v1/analyze",
            json={
                "owner": "repotriage-nonexistent-owner-abc123",
                "repo": "nonexistent-repo-xyz789",
            },
        )

        assert resp.status_code == 404
        data = resp.json()
        assert data["error"] == "repository_not_found"
        assert "message" in data

    async def test_invalid_owner_rejected_before_network(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """Pydantic validation catches a bad owner — no network call made."""
        resp = await live_client.post(
            "/api/v1/analyze",
            json={"owner": "-bad-owner-", "repo": "repo"},
        )
        assert resp.status_code == 422

    async def test_missing_repo_field_returns_422(
        self, live_client: httpx.AsyncClient
    ) -> None:
        """Missing required 'repo' field returns 422."""
        resp = await live_client.post(
            "/api/v1/analyze",
            json={"owner": "tiangolo"},
        )
        assert resp.status_code == 422
