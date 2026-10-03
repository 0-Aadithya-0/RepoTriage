"""
API request and response models for the RepoTriage FastAPI endpoints.

These models define the public contract of the API. Internal models
(GitHubIssue, IssueAnalysis) are mapped into these response shapes
by the service and route layers.
"""

import re

from pydantic import BaseModel, Field, field_validator

from app.models.analysis import IssueCategory, IssuePriority


# --------------------------------------------------------------------------- #
# Validation constants
# --------------------------------------------------------------------------- #

# GitHub owner: 1-39 alphanumeric or hyphen, cannot start/end with hyphen
_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,37}[a-zA-Z0-9])?$")

# GitHub repo: 1-100 chars, alphanumeric, hyphen, underscore, dot
_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9._\-]{1,100}$")

# Maximum number of issues to analyze per request
MAX_ANALYSIS_LIMIT = 10


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/v1/analyze."""

    owner: str = Field(
        min_length=1,
        max_length=39,
        description="GitHub repository owner or organization identifier.",
        examples=["tiangolo"],
    )
    repo: str = Field(
        min_length=1,
        max_length=100,
        description="GitHub repository name.",
        examples=["fastapi"],
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=MAX_ANALYSIS_LIMIT,
        description=(
            f"Number of issues to analyze (1–{MAX_ANALYSIS_LIMIT}). "
            f"Defaults to {MAX_ANALYSIS_LIMIT}."
        ),
    )

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, v: str) -> str:
        if not _OWNER_PATTERN.match(v):
            raise ValueError(
                "Invalid GitHub owner. Must be 1-39 alphanumeric characters "
                "or hyphens, cannot start or end with a hyphen."
            )
        return v

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        if not _REPO_PATTERN.match(v):
            raise ValueError(
                "Invalid GitHub repository name. Must be 1-100 characters "
                "using alphanumeric, hyphens, underscores, or dots."
            )
        # Prevent path-traversal attempts
        if ".." in v:
            raise ValueError(
                "Repository name must not contain consecutive dots."
            )
        return v


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class AnalyzedIssue(BaseModel):
    """A single analyzed issue in the API response."""

    issue_number: int = Field(description="The issue number.")
    title: str = Field(description="The issue title.")
    html_url: str = Field(description="The GitHub URL for this issue.")
    category: IssueCategory = Field(description="The classified category.")
    priority_level: IssuePriority = Field(description="The assigned priority.")
    tldr_summary: str = Field(description="One-sentence technical summary.")


class IssueFailure(BaseModel):
    """A single issue that failed LLM analysis (partial-failure policy B)."""

    issue_number: int = Field(description="The issue number that failed.")
    title: str = Field(description="The issue title.")
    html_url: str = Field(description="The GitHub URL for this issue.")
    error: str = Field(description="Human-readable error description.")


class AnalyzeResponse(BaseModel):
    """Response body for POST /api/v1/analyze."""

    repository: str = Field(
        description="The full owner/repo identifier.",
        examples=["tiangolo/fastapi"],
    )
    requested_limit: int = Field(
        description="The limit that was requested.",
    )
    analyzed_count: int = Field(
        ge=0,
        description="Number of successfully analyzed issues.",
    )
    failed_count: int = Field(
        ge=0,
        description="Number of issues that failed analysis.",
    )
    issues: list[AnalyzedIssue] = Field(
        default_factory=list,
        description="Successfully analyzed issues.",
    )
    failures: list[IssueFailure] = Field(
        default_factory=list,
        description="Issues that failed LLM analysis. Empty on full success.",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(default="ok")
