"""
Tests for Pydantic models: analysis enums, GitHub issue model, and API models.

Validates enum constraints, field validation, serialization, and edge cases.
"""

import pytest
from pydantic import ValidationError

from app.models.analysis import IssueAnalysis, IssueCategory, IssuePriority
from app.models.github import GitHubIssue
from app.models.api import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzedIssue,
    HealthResponse,
    IssueFailure,
    MAX_ANALYSIS_LIMIT,
)


# =========================================================================== #
# IssueCategory Enum
# =========================================================================== #


class TestIssueCategory:
    """Tests for the IssueCategory enum."""

    def test_all_categories_exist(self) -> None:
        expected = {"Bug", "Feature Request", "Documentation", "Support Question", "Other"}
        actual = {c.value for c in IssueCategory}
        assert actual == expected

    def test_category_string_values(self) -> None:
        assert IssueCategory.BUG == "Bug"
        assert IssueCategory.FEATURE_REQUEST == "Feature Request"
        assert IssueCategory.DOCUMENTATION == "Documentation"
        assert IssueCategory.SUPPORT_QUESTION == "Support Question"
        assert IssueCategory.OTHER == "Other"

    def test_category_is_str_enum(self) -> None:
        """StrEnum values should be directly usable as strings."""
        assert isinstance(IssueCategory.BUG, str)
        assert f"Category: {IssueCategory.BUG}" == "Category: Bug"

    def test_category_json_serialization(self) -> None:
        """Categories should serialize as their string values in JSON."""
        analysis = IssueAnalysis(
            category=IssueCategory.BUG,
            priority_level=IssuePriority.HIGH,
            tldr_summary="A test summary that is long enough to pass validation.",
        )
        data = analysis.model_dump()
        assert data["category"] == "Bug"


# =========================================================================== #
# IssuePriority Enum
# =========================================================================== #


class TestIssuePriority:
    """Tests for the IssuePriority enum."""

    def test_all_priorities_exist(self) -> None:
        expected = {"Critical", "High", "Medium", "Low"}
        actual = {p.value for p in IssuePriority}
        assert actual == expected

    def test_priority_string_values(self) -> None:
        assert IssuePriority.CRITICAL == "Critical"
        assert IssuePriority.HIGH == "High"
        assert IssuePriority.MEDIUM == "Medium"
        assert IssuePriority.LOW == "Low"

    def test_priority_is_str_enum(self) -> None:
        assert isinstance(IssuePriority.LOW, str)


# =========================================================================== #
# IssueAnalysis Model
# =========================================================================== #


class TestIssueAnalysis:
    """Tests for the IssueAnalysis Pydantic model (LLM structured output schema)."""

    def test_valid_analysis(self) -> None:
        analysis = IssueAnalysis(
            category=IssueCategory.FEATURE_REQUEST,
            priority_level=IssuePriority.MEDIUM,
            tldr_summary="Request to add WebSocket support for real-time updates.",
        )
        assert analysis.category == IssueCategory.FEATURE_REQUEST
        assert analysis.priority_level == IssuePriority.MEDIUM
        assert "WebSocket" in analysis.tldr_summary

    def test_summary_too_short(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            IssueAnalysis(
                category=IssueCategory.BUG,
                priority_level=IssuePriority.LOW,
                tldr_summary="Short",
            )
        assert "tldr_summary" in str(exc_info.value)

    def test_summary_too_long(self) -> None:
        with pytest.raises(ValidationError):
            IssueAnalysis(
                category=IssueCategory.BUG,
                priority_level=IssuePriority.LOW,
                tldr_summary="x" * 257,
            )

    def test_summary_at_max_length(self) -> None:
        analysis = IssueAnalysis(
            category=IssueCategory.BUG,
            priority_level=IssuePriority.LOW,
            tldr_summary="x" * 256,
        )
        assert len(analysis.tldr_summary) == 256

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueAnalysis(
                category="InvalidCategory",  # type: ignore[arg-type]
                priority_level=IssuePriority.HIGH,
                tldr_summary="A valid summary that is long enough for the test.",
            )

    def test_invalid_priority_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueAnalysis(
                category=IssueCategory.BUG,
                priority_level="Urgent",  # type: ignore[arg-type]
                tldr_summary="A valid summary that is long enough for the test.",
            )

    def test_json_round_trip(self) -> None:
        original = IssueAnalysis(
            category=IssueCategory.DOCUMENTATION,
            priority_level=IssuePriority.LOW,
            tldr_summary="Missing documentation for the retry configuration options.",
        )
        json_str = original.model_dump_json()
        restored = IssueAnalysis.model_validate_json(json_str)
        assert restored == original

    def test_from_dict_with_string_enum_values(self) -> None:
        """Simulates parsing LLM output that returns enum values as strings."""
        data = {
            "category": "Bug",
            "priority_level": "Critical",
            "tldr_summary": "Security vulnerability in authentication handler allows bypass.",
        }
        analysis = IssueAnalysis.model_validate(data)
        assert analysis.category == IssueCategory.BUG
        assert analysis.priority_level == IssuePriority.CRITICAL


# =========================================================================== #
# GitHubIssue Model
# =========================================================================== #


class TestGitHubIssue:
    """Tests for the GitHubIssue internal data model."""

    def test_valid_issue(self) -> None:
        issue = GitHubIssue(
            number=42,
            title="Fix login timeout",
            body="The login page times out after 5 seconds.",
            html_url="https://github.com/owner/repo/issues/42",
        )
        assert issue.number == 42
        assert issue.title == "Fix login timeout"

    def test_null_body_defaults_to_empty_string(self) -> None:
        """GitHub API may return null for issue body."""
        issue = GitHubIssue(
            number=1,
            title="No body issue",
            body=None,  # type: ignore[arg-type]
            html_url="https://github.com/owner/repo/issues/1",
        )
        assert issue.body == ""

    def test_missing_body_defaults_to_empty_string(self) -> None:
        """When body is not provided at all, default applies."""
        issue = GitHubIssue(
            number=1,
            title="No body issue",
            html_url="https://github.com/owner/repo/issues/1",
        )
        assert issue.body == ""

    def test_issue_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            GitHubIssue(
                number=0,
                title="Invalid",
                html_url="https://github.com/owner/repo/issues/0",
            )

    def test_title_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            GitHubIssue(
                number=1,
                title="",
                html_url="https://github.com/owner/repo/issues/1",
            )

    def test_body_length_limit(self) -> None:
        """Body is capped at 65536 characters at the model level."""
        with pytest.raises(ValidationError):
            GitHubIssue(
                number=1,
                title="Large body",
                body="x" * 65537,
                html_url="https://github.com/owner/repo/issues/1",
            )


# =========================================================================== #
# AnalyzeRequest Model
# =========================================================================== #


class TestAnalyzeRequest:
    """Tests for the API request model validation."""

    def test_valid_request(self) -> None:
        req = AnalyzeRequest(owner="tiangolo", repo="fastapi")
        assert req.owner == "tiangolo"
        assert req.repo == "fastapi"
        assert req.limit == 10  # default

    def test_custom_limit(self) -> None:
        req = AnalyzeRequest(owner="python", repo="cpython", limit=5)
        assert req.limit == 5

    def test_limit_max_is_enforced(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="owner", repo="repo", limit=MAX_ANALYSIS_LIMIT + 1)

    def test_limit_min_is_enforced(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="owner", repo="repo", limit=0)

    def test_owner_with_hyphens(self) -> None:
        req = AnalyzeRequest(owner="my-org", repo="my-repo")
        assert req.owner == "my-org"

    def test_owner_cannot_start_with_hyphen(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="-invalid", repo="repo")

    def test_owner_cannot_end_with_hyphen(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="invalid-", repo="repo")

    def test_owner_too_long(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="a" * 40, repo="repo")

    def test_owner_empty(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="", repo="repo")

    def test_repo_with_dots_and_underscores(self) -> None:
        req = AnalyzeRequest(owner="owner", repo="my_repo.v2")
        assert req.repo == "my_repo.v2"

    def test_repo_path_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="owner", repo="../etc/passwd")

    def test_repo_double_dots_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="owner", repo="repo..name")

    def test_repo_empty(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="owner", repo="")

    def test_repo_special_chars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(owner="owner", repo="repo/name")

    def test_single_char_owner(self) -> None:
        req = AnalyzeRequest(owner="a", repo="repo")
        assert req.owner == "a"


# =========================================================================== #
# Response Models
# =========================================================================== #


class TestAnalyzeResponse:
    """Tests for the API response models."""

    def test_full_success_response(self) -> None:
        response = AnalyzeResponse(
            repository="tiangolo/fastapi",
            requested_limit=10,
            analyzed_count=2,
            failed_count=0,
            issues=[
                AnalyzedIssue(
                    issue_number=123,
                    title="Example issue",
                    html_url="https://github.com/tiangolo/fastapi/issues/123",
                    category=IssueCategory.BUG,
                    priority_level=IssuePriority.HIGH,
                    tldr_summary="Authentication middleware raises TypeError on empty headers.",
                ),
            ],
            failures=[],
        )
        assert response.analyzed_count == 2
        assert response.failed_count == 0
        assert len(response.failures) == 0

    def test_partial_failure_response(self) -> None:
        response = AnalyzeResponse(
            repository="owner/repo",
            requested_limit=5,
            analyzed_count=4,
            failed_count=1,
            issues=[],
            failures=[
                IssueFailure(
                    issue_number=99,
                    title="Failed issue",
                    html_url="https://github.com/owner/repo/issues/99",
                    error="LLM analysis timed out after 3 attempts",
                ),
            ],
        )
        assert response.failed_count == 1
        assert response.failures[0].error == "LLM analysis timed out after 3 attempts"

    def test_json_serialization(self) -> None:
        response = AnalyzeResponse(
            repository="owner/repo",
            requested_limit=10,
            analyzed_count=0,
            failed_count=0,
        )
        data = response.model_dump()
        assert data["repository"] == "owner/repo"
        assert data["issues"] == []
        assert data["failures"] == []


class TestHealthResponse:
    """Tests for the health endpoint response model."""

    def test_default_status(self) -> None:
        health = HealthResponse()
        assert health.status == "ok"

    def test_json_output(self) -> None:
        data = HealthResponse().model_dump()
        assert data == {"status": "ok"}
