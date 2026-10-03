"""
Tests for the GitHub client.

All tests use httpx.MockTransport to simulate GitHub API responses.
No real network requests are made. Tests cover:
- Successful issue retrieval
- Pull request filtering
- Null body handling
- Newest-first ordering preservation
- Fewer than 10 real issues
- 404 → RepositoryNotFoundError
- 403 rate limit → GitHubRateLimitError with retry_after
- 403 permission denied → GitHubAuthenticationError
- 401 → GitHubAuthenticationError
- Timeout → GitHubTimeoutError
- 5xx → GitHubServerError
- Optional token changes headers
- Body truncation
"""

import json
import os
from typing import Any

import httpx
import pytest

from app.clients.github_client import GitHubClient, _TRUNCATION_MARKER
from app.core.config import Settings
from app.core.exceptions import (
    GitHubAuthenticationError,
    GitHubError,
    GitHubRateLimitError,
    GitHubServerError,
    GitHubTimeoutError,
    RepositoryNotFoundError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    number: int,
    *,
    title: str | None = None,
    body: str | None = "Issue body text",
    is_pr: bool = False,
) -> dict[str, Any]:
    """Create a fake GitHub API issue/PR item."""
    item: dict[str, Any] = {
        "number": number,
        "title": title or f"Issue #{number}",
        "body": body,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
    }
    if is_pr:
        item["pull_request"] = {
            "url": f"https://api.github.com/repos/owner/repo/pulls/{number}"
        }
    return item


def _make_transport(
    *,
    status_code: int = 200,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
    raise_timeout: bool = False,
) -> httpx.MockTransport:
    """Create a MockTransport that returns a fixed response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if raise_timeout:
            raise httpx.ReadTimeout(
                "Simulated timeout",
                request=request,
            )
        return httpx.Response(
            status_code=status_code,
            json=json_body if json_body is not None else [],
            headers=headers or {},
        )

    return httpx.MockTransport(handler)


def _make_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with test defaults."""
    env_defaults = {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "sk-test-key",
        "GITHUB_API_BASE_URL": "https://api.github.com",
        "GITHUB_API_VERSION": "2026-03-10",
        "GITHUB_TIMEOUT_SECONDS": "15.0",
        "MAX_ISSUE_BODY_LENGTH": "4096",
    }
    env_defaults.update(overrides)
    # Temporarily set env vars and create settings
    original = {}
    for k, v in env_defaults.items():
        original[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        return Settings(_env_file=None)  # type: ignore[call-arg]
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_client(
    transport: httpx.MockTransport,
    settings: Settings | None = None,
) -> GitHubClient:
    """Create a GitHubClient with a mock transport."""
    if settings is None:
        settings = _make_settings()
    http = httpx.AsyncClient(transport=transport)
    return GitHubClient(http_client=http, settings=settings)


# ---------------------------------------------------------------------------
# Successful retrieval
# ---------------------------------------------------------------------------


class TestSuccessfulFetch:
    """Tests for successful issue retrieval."""

    async def test_returns_issues_in_order(self) -> None:
        """Issues are returned in the same order as the API response."""
        items = [_make_issue(10), _make_issue(9), _make_issue(8)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo", limit=10)

        assert len(issues) == 3
        assert [i.number for i in issues] == [10, 9, 8]

    async def test_respects_limit(self) -> None:
        """Only `limit` issues are returned even when more are available."""
        items = [_make_issue(i) for i in range(20, 0, -1)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo", limit=5)

        assert len(issues) == 5
        assert issues[0].number == 20
        assert issues[4].number == 16

    async def test_parses_issue_fields(self) -> None:
        """All fields are correctly mapped from the API response."""
        items = [
            {
                "number": 42,
                "title": "Fix the bug",
                "body": "Detailed bug description here.",
                "html_url": "https://github.com/owner/repo/issues/42",
            }
        ]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo")

        assert issues[0].number == 42
        assert issues[0].title == "Fix the bug"
        assert issues[0].body == "Detailed bug description here."
        assert issues[0].html_url == "https://github.com/owner/repo/issues/42"

    async def test_empty_repository(self) -> None:
        """An empty issue list is handled gracefully."""
        transport = _make_transport(json_body=[])
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo")

        assert issues == []

    async def test_fewer_than_limit_available(self) -> None:
        """When fewer issues exist than `limit`, all are returned."""
        items = [_make_issue(3), _make_issue(2)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo", limit=10)

        assert len(issues) == 2


# ---------------------------------------------------------------------------
# Pull request filtering
# ---------------------------------------------------------------------------


class TestPullRequestFiltering:
    """Tests for excluding pull requests from the response."""

    async def test_pull_requests_are_excluded(self) -> None:
        """Items with a `pull_request` key are filtered out."""
        items = [
            _make_issue(10),
            _make_issue(9, is_pr=True),
            _make_issue(8),
            _make_issue(7, is_pr=True),
            _make_issue(6),
        ]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo")

        assert len(issues) == 3
        assert [i.number for i in issues] == [10, 8, 6]

    async def test_all_pull_requests_returns_empty(self) -> None:
        """If every item is a PR, an empty list is returned."""
        items = [_make_issue(i, is_pr=True) for i in range(5, 0, -1)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo")

        assert issues == []

    async def test_limit_applies_after_filtering(self) -> None:
        """The limit counts real issues, not including PRs."""
        items = [
            _make_issue(10),
            _make_issue(9, is_pr=True),
            _make_issue(8),
            _make_issue(7, is_pr=True),
            _make_issue(6),
            _make_issue(5),
        ]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo", limit=2)

        assert len(issues) == 2
        assert [i.number for i in issues] == [10, 8]


# ---------------------------------------------------------------------------
# Null body handling
# ---------------------------------------------------------------------------


class TestNullBodyHandling:
    """Tests for issues with null or missing body."""

    async def test_null_body_becomes_empty_string(self) -> None:
        """GitHub's null body is converted to empty string."""
        items = [_make_issue(1, body=None)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo")

        assert issues[0].body == ""

    async def test_missing_body_key_becomes_empty_string(self) -> None:
        """If the body key is absent entirely, it defaults to empty string."""
        items = [{"number": 1, "title": "No body", "html_url": "http://x.com/1"}]
        transport = _make_transport(json_body=items)
        client = _make_client(transport)

        issues = await client.fetch_issues("owner", "repo")

        assert issues[0].body == ""


# ---------------------------------------------------------------------------
# Body truncation
# ---------------------------------------------------------------------------


class TestBodyTruncation:
    """Tests for body length truncation."""

    async def test_long_body_is_truncated(self) -> None:
        """Bodies exceeding max_issue_body_length are truncated with marker."""
        settings = _make_settings(MAX_ISSUE_BODY_LENGTH="256")
        long_body = "x" * 500
        items = [_make_issue(1, body=long_body)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport, settings=settings)

        issues = await client.fetch_issues("owner", "repo")

        assert issues[0].body.endswith(_TRUNCATION_MARKER)
        # The content portion should be exactly 256 chars
        content = issues[0].body[: -len(_TRUNCATION_MARKER)]
        assert len(content) == 256

    async def test_body_at_limit_is_not_truncated(self) -> None:
        """Bodies at exactly the limit are not truncated."""
        settings = _make_settings(MAX_ISSUE_BODY_LENGTH="256")
        exact_body = "x" * 256
        items = [_make_issue(1, body=exact_body)]
        transport = _make_transport(json_body=items)
        client = _make_client(transport, settings=settings)

        issues = await client.fetch_issues("owner", "repo")

        assert issues[0].body == exact_body
        assert _TRUNCATION_MARKER not in issues[0].body


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """Tests for HTTP status code to domain exception mapping."""

    async def test_404_raises_repository_not_found(self) -> None:
        transport = _make_transport(status_code=404)
        client = _make_client(transport)

        with pytest.raises(RepositoryNotFoundError, match="not found"):
            await client.fetch_issues("owner", "nonexistent")

    async def test_401_raises_authentication_error(self) -> None:
        transport = _make_transport(status_code=401)
        client = _make_client(transport)

        with pytest.raises(GitHubAuthenticationError, match="authentication"):
            await client.fetch_issues("owner", "repo")

    async def test_403_rate_limit_raises_rate_limit_error(self) -> None:
        """403 with x-ratelimit-remaining: 0 maps to GitHubRateLimitError."""
        transport = _make_transport(
            status_code=403,
            headers={
                "x-ratelimit-remaining": "0",
                "retry-after": "60",
            },
        )
        client = _make_client(transport)

        with pytest.raises(GitHubRateLimitError) as exc_info:
            await client.fetch_issues("owner", "repo")
        assert exc_info.value.retry_after == 60

    async def test_403_rate_limit_without_retry_after(self) -> None:
        """Rate limit error without Retry-After header still works."""
        transport = _make_transport(
            status_code=403,
            headers={"x-ratelimit-remaining": "0"},
        )
        client = _make_client(transport)

        with pytest.raises(GitHubRateLimitError) as exc_info:
            await client.fetch_issues("owner", "repo")
        assert exc_info.value.retry_after is None

    async def test_403_permission_denied_raises_auth_error(self) -> None:
        """403 without rate limit indicators is a permission error."""
        transport = _make_transport(
            status_code=403,
            headers={"x-ratelimit-remaining": "50"},
        )
        client = _make_client(transport)

        with pytest.raises(GitHubAuthenticationError, match="forbidden"):
            await client.fetch_issues("owner", "repo")

    async def test_500_raises_server_error(self) -> None:
        transport = _make_transport(status_code=500)
        client = _make_client(transport)

        with pytest.raises(GitHubServerError, match="500"):
            await client.fetch_issues("owner", "repo")

    async def test_502_raises_server_error(self) -> None:
        transport = _make_transport(status_code=502)
        client = _make_client(transport)

        with pytest.raises(GitHubServerError, match="502"):
            await client.fetch_issues("owner", "repo")

    async def test_503_raises_server_error(self) -> None:
        transport = _make_transport(status_code=503)
        client = _make_client(transport)

        with pytest.raises(GitHubServerError, match="503"):
            await client.fetch_issues("owner", "repo")

    async def test_422_raises_github_error(self) -> None:
        transport = _make_transport(status_code=422)
        client = _make_client(transport)

        with pytest.raises(GitHubError, match="422"):
            await client.fetch_issues("owner", "repo")


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Tests for timeout behavior."""

    async def test_timeout_raises_github_timeout_error(self) -> None:
        transport = _make_transport(raise_timeout=True)
        client = _make_client(transport)

        with pytest.raises(GitHubTimeoutError, match="timed out"):
            await client.fetch_issues("owner", "repo")


# ---------------------------------------------------------------------------
# Authentication headers
# ---------------------------------------------------------------------------


class TestAuthenticationHeaders:
    """Tests for optional token handling in request headers."""

    async def test_no_token_sends_no_authorization(self) -> None:
        """When GITHUB_TOKEN is absent, no Authorization header is sent."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        # No GITHUB_TOKEN in env
        settings = _make_settings()
        client = _make_client(transport, settings=settings)

        await client.fetch_issues("owner", "repo")

        assert "authorization" not in captured_headers

    async def test_token_sends_bearer_authorization(self) -> None:
        """When GITHUB_TOKEN is set, Bearer auth is used."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        settings = _make_settings(GITHUB_TOKEN="ghp_test_token_123")
        client = _make_client(transport, settings=settings)

        await client.fetch_issues("owner", "repo")

        assert captured_headers.get("authorization") == "Bearer ghp_test_token_123"

    async def test_required_headers_are_always_present(self) -> None:
        """Accept, X-GitHub-Api-Version, and User-Agent headers are sent."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        client = _make_client(transport)

        await client.fetch_issues("owner", "repo")

        assert captured_headers["accept"] == "application/vnd.github+json"
        assert captured_headers["x-github-api-version"] == "2026-03-10"
        assert captured_headers["user-agent"] == "RepoTriage/1.0"


# ---------------------------------------------------------------------------
# Query parameters
# ---------------------------------------------------------------------------


class TestQueryParameters:
    """Tests for correct query parameter construction."""

    async def test_correct_query_parameters(self) -> None:
        """Verify the query parameters sent to GitHub."""
        captured_url: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_url.append(request.url)
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        client = _make_client(transport)

        await client.fetch_issues("owner", "repo")

        url = captured_url[0]
        assert url.params["state"] == "open"
        assert url.params["sort"] == "created"
        assert url.params["direction"] == "desc"
        assert url.params["per_page"] == "30"

    async def test_correct_url_path(self) -> None:
        """The URL path includes owner and repo."""
        captured_url: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_url.append(request.url)
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        client = _make_client(transport)

        await client.fetch_issues("tiangolo", "fastapi")

        assert "/repos/tiangolo/fastapi/issues" in str(captured_url[0])


# ---------------------------------------------------------------------------
# Unexpected response shape
# ---------------------------------------------------------------------------


    async def test_non_list_response_raises_github_error(self) -> None:
        """If GitHub returns an object instead of a list, we raise clearly."""
        transport = _make_transport(
            json_body={"message": "Not Found"},
            status_code=200,
        )
        client = _make_client(transport)

        with pytest.raises(GitHubError, match="expected list"):
            await client.fetch_issues("owner", "repo")


# ---------------------------------------------------------------------------
# Uncovered paths: httpx.HTTPError, unexpected status, non-int retry-after
# ---------------------------------------------------------------------------


class TestAdditionalErrorPaths:
    """Tests for previously uncovered error-handling branches."""

    async def test_generic_http_error_raises_github_error(self) -> None:
        """A non-timeout httpx.HTTPError maps to a plain GitHubError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError(
                "Connection reset", request=request
            )

        transport = httpx.MockTransport(handler)
        client = _make_client(transport)

        with pytest.raises(GitHubError, match="RemoteProtocolError"):
            await client.fetch_issues("owner", "repo")

    async def test_unexpected_status_code_raises_github_error(self) -> None:
        """A status like 418 (not explicitly handled) maps to a generic GitHubError."""
        transport = _make_transport(status_code=418)
        client = _make_client(transport)

        with pytest.raises(GitHubError, match="Unexpected GitHub API response \\(HTTP 418\\)"):
            await client.fetch_issues("owner", "repo")

    async def test_non_integer_retry_after_header_is_ignored(self) -> None:
        """A non-integer Retry-After header does not crash — retry_after is None."""
        transport = _make_transport(
            status_code=403,
            headers={
                "x-ratelimit-remaining": "0",
                "retry-after": "not-a-number",
            },
        )
        client = _make_client(transport)

        with pytest.raises(GitHubRateLimitError) as exc_info:
            await client.fetch_issues("owner", "repo")
        assert exc_info.value.retry_after is None
