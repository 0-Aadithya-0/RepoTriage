"""
Tests for the analysis orchestrator.

Uses FakeLLMProvider and a mock GitHubClient to verify:
- Full pipeline: fetch → analyze → response mapping.
- Partial-failure semantics: individual LLM failures produce IssueFailure entries.
- Empty repository handling.
- Bounded concurrency enforcement.
- Correct field mapping from internal models to API response models.
- Unexpected exception handling (defensive catch-all).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import (
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    RepositoryNotFoundError,
)
from app.models.analysis import IssueAnalysis, IssueCategory, IssuePriority
from app.models.github import GitHubIssue
from app.models.prompts import FewShotExample, PromptConfig
from app.services.analyzer import AnalysisResult, analyze_repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompt_config() -> PromptConfig:
    """A valid PromptConfig for testing."""
    return PromptConfig(
        version="1.0",
        system_role="You are an AI triage assistant.",
        task="Classify the issue.",
        classification_rules="Use Bug for bugs.",
        priority_rules="High for urgent.",
        summary_rules="One sentence.",
        few_shot_examples=[
            FewShotExample(
                title="App crashes on startup",
                body="The application crashes immediately after launch.",
                category=IssueCategory.BUG,
                priority=IssuePriority.HIGH,
                summary="Application crashes on startup due to unhandled exception.",
            ),
        ],
    )


def _make_issue(number: int, title: str = "") -> GitHubIssue:
    """Create a test GitHubIssue."""
    return GitHubIssue(
        number=number,
        title=title or f"Issue #{number}",
        body=f"Body for issue #{number}.",
        html_url=f"https://github.com/owner/repo/issues/{number}",
    )


def _make_mock_github_client(issues: list[GitHubIssue]) -> MagicMock:
    """Create a mock GitHubClient that returns the given issues."""
    mock = MagicMock()
    mock.fetch_issues = AsyncMock(return_value=issues)
    return mock


# ---------------------------------------------------------------------------
# AnalysisResult unit tests
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """Tests for the AnalysisResult internal container."""

    def test_success_result(self) -> None:
        issue = _make_issue(1)
        analysis = IssueAnalysis(
            category=IssueCategory.BUG,
            priority_level=IssuePriority.HIGH,
            tldr_summary="A high priority bug.",
        )
        result = AnalysisResult(issue, analysis=analysis)

        assert result.is_success is True
        assert result.analysis == analysis
        assert result.error is None

    def test_failure_result(self) -> None:
        issue = _make_issue(1)
        result = AnalysisResult(issue, error="LLM timed out")

        assert result.is_success is False
        assert result.analysis is None
        assert result.error == "LLM timed out"


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Tests for the complete analyze_repository pipeline."""

    async def test_successful_analysis_of_multiple_issues(
        self, fake_provider, prompt_config: PromptConfig
    ) -> None:
        """All issues analyzed successfully produce AnalyzedIssue entries."""
        issues = [_make_issue(1), _make_issue(2), _make_issue(3)]
        mock_github = _make_mock_github_client(issues)

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=3,
            github_client=mock_github,
            llm_provider=fake_provider,
            prompt_config=prompt_config,
        )

        assert response.repository == "owner/repo"
        assert response.requested_limit == 3
        assert response.analyzed_count == 3
        assert response.failed_count == 0
        assert len(response.issues) == 3
        assert len(response.failures) == 0

        # Verify fields are correctly mapped
        assert response.issues[0].issue_number == 1
        assert response.issues[0].title == "Issue #1"
        assert response.issues[0].html_url == "https://github.com/owner/repo/issues/1"
        assert response.issues[0].category == IssueCategory.BUG
        assert response.issues[0].priority_level == IssuePriority.MEDIUM

        # Verify the fake provider was called for each issue
        assert fake_provider.call_count == 3

    async def test_github_client_called_with_correct_args(
        self, fake_provider, prompt_config: PromptConfig
    ) -> None:
        """GitHubClient.fetch_issues is called with the correct arguments."""
        mock_github = _make_mock_github_client([])

        await analyze_repository(
            owner="tiangolo",
            repo="fastapi",
            limit=5,
            github_client=mock_github,
            llm_provider=fake_provider,
            prompt_config=prompt_config,
        )

        mock_github.fetch_issues.assert_called_once_with(
            "tiangolo", "fastapi", limit=5
        )


# ---------------------------------------------------------------------------
# Empty repository
# ---------------------------------------------------------------------------


class TestEmptyRepository:
    """Tests for when the repository has no open issues."""

    async def test_empty_repo_returns_zero_counts(
        self, fake_provider, prompt_config: PromptConfig
    ) -> None:
        mock_github = _make_mock_github_client([])

        response = await analyze_repository(
            owner="owner",
            repo="empty-repo",
            limit=10,
            github_client=mock_github,
            llm_provider=fake_provider,
            prompt_config=prompt_config,
        )

        assert response.analyzed_count == 0
        assert response.failed_count == 0
        assert response.issues == []
        assert response.failures == []
        assert fake_provider.call_count == 0


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


class TestPartialFailure:
    """Tests for partial-failure semantics."""

    async def test_single_llm_failure_does_not_abort_batch(
        self, prompt_config: PromptConfig
    ) -> None:
        """One failing issue should not prevent others from succeeding."""
        issues = [_make_issue(1), _make_issue(2), _make_issue(3)]
        mock_github = _make_mock_github_client(issues)

        call_count = 0

        async def _conditional_analyze(
            issue: GitHubIssue, _prompt_config: PromptConfig
        ) -> IssueAnalysis:
            nonlocal call_count
            call_count += 1
            if issue.number == 2:
                raise LLMTimeoutError("Request timed out after 30s")
            return IssueAnalysis(
                category=IssueCategory.BUG,
                priority_level=IssuePriority.LOW,
                tldr_summary=f"Analysis of issue #{issue.number}.",
            )

        mock_provider = MagicMock()
        mock_provider.analyze_issue = _conditional_analyze

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=3,
            github_client=mock_github,
            llm_provider=mock_provider,
            prompt_config=prompt_config,
        )

        assert response.analyzed_count == 2
        assert response.failed_count == 1
        assert len(response.issues) == 2
        assert len(response.failures) == 1

        # The failed issue is #2
        assert response.failures[0].issue_number == 2
        assert "timed out" in response.failures[0].error

        # Successful issues are #1 and #3
        successful_numbers = [i.issue_number for i in response.issues]
        assert 1 in successful_numbers
        assert 3 in successful_numbers

    async def test_all_llm_failures(
        self, prompt_config: PromptConfig
    ) -> None:
        """If all LLM calls fail, all become failures in the response."""
        from tests.conftest import FakeLLMProvider

        issues = [_make_issue(1), _make_issue(2)]
        mock_github = _make_mock_github_client(issues)
        failing_provider = FakeLLMProvider(
            exception_to_raise=LLMServerError("Internal error")
        )

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=2,
            github_client=mock_github,
            llm_provider=failing_provider,
            prompt_config=prompt_config,
        )

        assert response.analyzed_count == 0
        assert response.failed_count == 2
        assert all("Internal error" in f.error for f in response.failures)

    async def test_rate_limit_error_produces_failure(
        self, prompt_config: PromptConfig
    ) -> None:
        """LLMRateLimitError on a single issue becomes an IssueFailure."""
        from tests.conftest import FakeLLMProvider

        issues = [_make_issue(1)]
        mock_github = _make_mock_github_client(issues)
        provider = FakeLLMProvider(
            exception_to_raise=LLMRateLimitError("Rate limit exceeded")
        )

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=1,
            github_client=mock_github,
            llm_provider=provider,
            prompt_config=prompt_config,
        )

        assert response.analyzed_count == 0
        assert response.failed_count == 1
        assert "Rate limit" in response.failures[0].error


# ---------------------------------------------------------------------------
# Unexpected exception handling
# ---------------------------------------------------------------------------


class TestUnexpectedExceptions:
    """Tests for defensive catch-all of non-LLMError exceptions."""

    async def test_unexpected_error_becomes_failure(
        self, prompt_config: PromptConfig
    ) -> None:
        """A non-LLMError exception is caught and reported as a failure."""
        issues = [_make_issue(1)]
        mock_github = _make_mock_github_client(issues)

        async def _explode(issue: GitHubIssue, _pc: PromptConfig) -> IssueAnalysis:
            raise RuntimeError("Something completely unexpected")

        mock_provider = MagicMock()
        mock_provider.analyze_issue = _explode

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=1,
            github_client=mock_github,
            llm_provider=mock_provider,
            prompt_config=prompt_config,
        )

        assert response.analyzed_count == 0
        assert response.failed_count == 1
        assert "Unexpected error" in response.failures[0].error


# ---------------------------------------------------------------------------
# GitHub error propagation
# ---------------------------------------------------------------------------


class TestGitHubErrorPropagation:
    """Tests that GitHub errors propagate up (not caught by orchestrator)."""

    async def test_repo_not_found_propagates(
        self, fake_provider, prompt_config: PromptConfig
    ) -> None:
        """RepositoryNotFoundError from GitHub is not caught."""
        mock_github = MagicMock()
        mock_github.fetch_issues = AsyncMock(
            side_effect=RepositoryNotFoundError("Not found")
        )

        with pytest.raises(RepositoryNotFoundError, match="Not found"):
            await analyze_repository(
                owner="owner",
                repo="nonexistent",
                limit=5,
                github_client=mock_github,
                llm_provider=fake_provider,
                prompt_config=prompt_config,
            )


# ---------------------------------------------------------------------------
# Concurrency enforcement
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for bounded concurrency via semaphore."""

    async def test_concurrency_is_bounded(
        self, prompt_config: PromptConfig
    ) -> None:
        """No more than max_concurrency LLM calls run simultaneously."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _tracking_analyze(
            issue: GitHubIssue, _pc: PromptConfig
        ) -> IssueAnalysis:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            # Simulate some work
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return IssueAnalysis(
                category=IssueCategory.BUG,
                priority_level=IssuePriority.LOW,
                tldr_summary=f"Analysis of issue #{issue.number}.",
            )

        issues = [_make_issue(i) for i in range(1, 7)]  # 6 issues
        mock_github = _make_mock_github_client(issues)

        mock_provider = MagicMock()
        mock_provider.analyze_issue = _tracking_analyze

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=6,
            github_client=mock_github,
            llm_provider=mock_provider,
            prompt_config=prompt_config,
            max_concurrency=2,
        )

        assert response.analyzed_count == 6
        assert max_concurrent <= 2, (
            f"Expected at most 2 concurrent calls, got {max_concurrent}"
        )

    async def test_order_preserved(
        self, prompt_config: PromptConfig
    ) -> None:
        """Results are returned in the same order as the input issues."""
        from tests.conftest import FakeLLMProvider

        issues = [_make_issue(10), _make_issue(9), _make_issue(8)]
        mock_github = _make_mock_github_client(issues)
        provider = FakeLLMProvider()

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=3,
            github_client=mock_github,
            llm_provider=provider,
            prompt_config=prompt_config,
        )

        result_numbers = [i.issue_number for i in response.issues]
        assert result_numbers == [10, 9, 8]


# ---------------------------------------------------------------------------
# Field mapping accuracy
# ---------------------------------------------------------------------------


class TestFieldMapping:
    """Tests that internal models are correctly mapped to API response models."""

    async def test_analyzed_issue_fields(
        self, fake_provider, prompt_config: PromptConfig
    ) -> None:
        """All AnalyzedIssue fields are populated from the right sources."""
        issues = [
            GitHubIssue(
                number=42,
                title="Fix the crash",
                body="Detailed description.",
                html_url="https://github.com/owner/repo/issues/42",
            )
        ]
        mock_github = _make_mock_github_client(issues)

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=1,
            github_client=mock_github,
            llm_provider=fake_provider,
            prompt_config=prompt_config,
        )

        item = response.issues[0]
        # From GitHubIssue
        assert item.issue_number == 42
        assert item.title == "Fix the crash"
        assert item.html_url == "https://github.com/owner/repo/issues/42"
        # From IssueAnalysis (via FakeLLMProvider)
        assert item.category == IssueCategory.BUG
        assert item.priority_level == IssuePriority.MEDIUM
        assert "Fix the crash" in item.tldr_summary

    async def test_failure_fields(
        self, prompt_config: PromptConfig
    ) -> None:
        """IssueFailure fields are populated correctly on error."""
        from tests.conftest import FakeLLMProvider

        issues = [
            GitHubIssue(
                number=99,
                title="Important issue",
                body="Details.",
                html_url="https://github.com/owner/repo/issues/99",
            )
        ]
        mock_github = _make_mock_github_client(issues)
        provider = FakeLLMProvider(
            exception_to_raise=LLMServerError("Provider is down")
        )

        response = await analyze_repository(
            owner="owner",
            repo="repo",
            limit=1,
            github_client=mock_github,
            llm_provider=provider,
            prompt_config=prompt_config,
        )

        failure = response.failures[0]
        assert failure.issue_number == 99
        assert failure.title == "Important issue"
        assert failure.html_url == "https://github.com/owner/repo/issues/99"
        assert "Provider is down" in failure.error
