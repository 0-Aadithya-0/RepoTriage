"""
GitHub data models for internal representation of fetched issues.

These models represent the subset of GitHub API response data that
RepoTriage needs. They are not API response models — those live in api.py.
"""

from pydantic import BaseModel, Field, field_validator


class GitHubIssue(BaseModel):
    """Internal representation of a GitHub issue after filtering.

    Created from the GitHub REST API response. Pull requests are
    excluded before this model is instantiated.
    """

    number: int = Field(
        gt=0,
        description="The issue number within the repository.",
    )
    title: str = Field(
        min_length=1,
        max_length=1024,
        description="The issue title.",
    )
    body: str = Field(
        default="",
        max_length=65536,
        description=(
            "The issue body text. Defaults to empty string when "
            "the GitHub API returns null."
        ),
    )
    html_url: str = Field(
        description="The full GitHub URL for this issue.",
    )

    @field_validator("body", mode="before")
    @classmethod
    def convert_none_to_empty_string(cls, v: str | None) -> str:
        return "" if v is None else v
