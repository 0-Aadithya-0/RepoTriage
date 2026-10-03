"""
Tests for the exception-to-HTTP error handlers.

Verifies each domain exception maps to the correct HTTP status code
and JSON error body. Uses direct handler invocation (no full app needed).
"""

from unittest.mock import MagicMock

import pytest
from fastapi import Request

from app.core.exceptions import (
    ConfigurationError,
    GitHubAuthenticationError,
    GitHubRateLimitError,
    GitHubServerError,
    GitHubTimeoutError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
    RepositoryNotFoundError,
    RepoTriageError,
)
from app.middleware.error_handlers import (
    config_error_handler,
    generic_error_handler,
    github_auth_handler,
    github_rate_limit_handler,
    github_server_error_handler,
    github_timeout_handler,
    llm_auth_handler,
    llm_parse_error_handler,
    llm_rate_limit_handler,
    llm_refusal_handler,
    llm_server_error_handler,
    llm_timeout_handler,
    repository_not_found_handler,
    repotriage_error_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_request() -> MagicMock:
    """Create a minimal mock Request."""
    mock = MagicMock(spec=Request)
    mock.url = "http://test/api/v1/analyze"
    return mock


# ---------------------------------------------------------------------------
# GitHub error handlers
# ---------------------------------------------------------------------------


class TestGitHubErrorHandlers:

    async def test_repository_not_found(self) -> None:
        exc = RepositoryNotFoundError("Repository not found")
        resp = await repository_not_found_handler(_fake_request(), exc)
        assert resp.status_code == 404
        assert resp.body is not None
        body = resp.body.decode()
        assert "repository_not_found" in body

    async def test_github_auth_error(self) -> None:
        exc = GitHubAuthenticationError("Token invalid")
        resp = await github_auth_handler(_fake_request(), exc)
        assert resp.status_code == 401
        body = resp.body.decode()
        assert "github_authentication_error" in body

    async def test_github_rate_limit(self) -> None:
        exc = GitHubRateLimitError("Rate limited", retry_after=60)
        resp = await github_rate_limit_handler(_fake_request(), exc)
        assert resp.status_code == 429
        body = resp.body.decode()
        assert "github_rate_limit" in body
        assert "60" in body
        assert resp.headers.get("retry-after") == "60"

    async def test_github_rate_limit_no_retry_after(self) -> None:
        exc = GitHubRateLimitError("Rate limited")
        resp = await github_rate_limit_handler(_fake_request(), exc)
        assert resp.status_code == 429
        assert "retry-after" not in resp.headers

    async def test_github_timeout(self) -> None:
        exc = GitHubTimeoutError("Timed out")
        resp = await github_timeout_handler(_fake_request(), exc)
        assert resp.status_code == 504

    async def test_github_server_error(self) -> None:
        exc = GitHubServerError("Server error 500")
        resp = await github_server_error_handler(_fake_request(), exc)
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# LLM error handlers
# ---------------------------------------------------------------------------


class TestLLMErrorHandlers:

    async def test_llm_auth_error(self) -> None:
        exc = LLMAuthenticationError("Bad key")
        resp = await llm_auth_handler(_fake_request(), exc)
        assert resp.status_code == 401
        body = resp.body.decode()
        assert "llm_authentication_error" in body
        # Must NOT leak the actual error message (could contain key info)
        assert "Bad key" not in body

    async def test_llm_rate_limit(self) -> None:
        exc = LLMRateLimitError("Too many requests", retry_after=30)
        resp = await llm_rate_limit_handler(_fake_request(), exc)
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "30"

    async def test_llm_timeout(self) -> None:
        exc = LLMTimeoutError("Timed out after 30s")
        resp = await llm_timeout_handler(_fake_request(), exc)
        assert resp.status_code == 504

    async def test_llm_server_error(self) -> None:
        exc = LLMServerError("Provider down")
        resp = await llm_server_error_handler(_fake_request(), exc)
        assert resp.status_code == 502

    async def test_llm_parse_error(self) -> None:
        exc = LLMResponseParseError("Could not parse JSON")
        resp = await llm_parse_error_handler(_fake_request(), exc)
        assert resp.status_code == 502
        body = resp.body.decode()
        assert "llm_parse_error" in body

    async def test_llm_refusal(self) -> None:
        exc = LLMRefusalError("Safety filter")
        resp = await llm_refusal_handler(_fake_request(), exc)
        assert resp.status_code == 422
        body = resp.body.decode()
        assert "llm_refusal" in body


# ---------------------------------------------------------------------------
# Generic / catch-all handlers
# ---------------------------------------------------------------------------


class TestGenericHandlers:

    async def test_config_error(self) -> None:
        exc = ConfigurationError("Missing GEMINI_API_KEY")
        resp = await config_error_handler(_fake_request(), exc)
        assert resp.status_code == 500
        body = resp.body.decode()
        assert "configuration_error" in body

    async def test_repotriage_catch_all(self) -> None:
        exc = RepoTriageError("Something weird")
        resp = await repotriage_error_handler(_fake_request(), exc)
        assert resp.status_code == 500
        body = resp.body.decode()
        assert "internal_error" in body

    async def test_generic_exception_handler(self) -> None:
        exc = RuntimeError("Totally unexpected")
        resp = await generic_error_handler(_fake_request(), exc)
        assert resp.status_code == 500
        body = resp.body.decode()
        assert "internal_error" in body
        # Must NOT leak the actual error message
        assert "Totally unexpected" not in body

    async def test_generic_handler_never_leaks_internals(self) -> None:
        exc = ValueError("secret_api_key=abc123")
        resp = await generic_error_handler(_fake_request(), exc)
        body = resp.body.decode()
        assert "secret_api_key" not in body
        assert "abc123" not in body
