"""
Shared test fixtures for the RepoTriage test suite.

Fixtures defined here are available to all tests without explicit imports.
"""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Provide a clean environment with no RepoTriage env vars set.

    Removes all known RepoTriage environment variables before the test
    and restores the original environment afterward.
    """
    repotriage_vars = [
        "APP_NAME",
        "APP_ENVIRONMENT",
        "LOG_LEVEL",
        "GITHUB_TOKEN",
        "GITHUB_API_BASE_URL",
        "GITHUB_API_VERSION",
        "GITHUB_TIMEOUT_SECONDS",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_CONCURRENCY",
        "LLM_MAX_RETRIES",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "PROMPTS_FILE",
        "MAX_ISSUE_BODY_LENGTH",
    ]
    original = {k: os.environ.get(k) for k in repotriage_vars}

    for var in repotriage_vars:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    for var, value in original.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


@pytest.fixture
def env_with_gemini(clean_env: None) -> None:
    """Set up environment for Gemini provider testing."""
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = "test-gemini-key-not-real"


@pytest.fixture
def no_dotenv() -> Generator[None, None, None]:
    """Prevent pydantic-settings from loading any .env file during tests.

    This ensures tests are deterministic and not affected by a developer's
    local .env file.
    """
    with patch.dict(os.environ, {}, clear=False):
        yield


# ---------------------------------------------------------------------------
# Fake LLM Provider for service-layer testing
# ---------------------------------------------------------------------------

from app.models.analysis import IssueAnalysis, IssueCategory, IssuePriority
from app.models.github import GitHubIssue
from app.models.prompts import PromptConfig


class FakeLLMProvider:
    """A protocol-satisfying fake LLM provider for testing.

    Returns deterministic IssueAnalysis results. Can be configured
    to raise exceptions for failure testing.

    Attributes:
        call_count: Number of times analyze_issue was called.
        calls: List of (issue, prompt_config) tuples for each call.
        exception_to_raise: If set, raised instead of returning a result.
        delay_seconds: If > 0, simulates processing delay (for concurrency tests).
    """

    def __init__(
        self,
        *,
        exception_to_raise: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.call_count = 0
        self.calls: list[tuple[GitHubIssue, PromptConfig]] = []
        self.exception_to_raise = exception_to_raise
        self.delay_seconds = delay_seconds

    async def analyze_issue(
        self,
        issue: GitHubIssue,
        prompt_config: PromptConfig,
    ) -> IssueAnalysis:
        """Return a deterministic analysis or raise the configured exception."""
        import asyncio

        self.call_count += 1
        self.calls.append((issue, prompt_config))

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        return IssueAnalysis(
            category=IssueCategory.BUG,
            priority_level=IssuePriority.MEDIUM,
            tldr_summary=f"Analysis of issue #{issue.number}: {issue.title}",
        )


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """Provide a fresh FakeLLMProvider instance."""
    return FakeLLMProvider()
