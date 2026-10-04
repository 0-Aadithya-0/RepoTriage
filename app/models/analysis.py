"""
Analysis models for LLM-generated issue triage results.

Defines the enums and Pydantic models that constrain LLM structured output
to valid category/priority combinations with a bounded summary.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class IssueCategory(StrEnum):
    """Classification categories for GitHub issues.

    Uses StrEnum so JSON serialization produces lowercase string values
    compatible with both OpenAI and Gemini structured output schemas.
    """

    BUG = "Bug"
    FEATURE_REQUEST = "Feature Request"
    DOCUMENTATION = "Documentation"
    SUPPORT_QUESTION = "Support Question"
    OTHER = "Other"


class IssuePriority(StrEnum):
    """Priority levels for triaged issues.

    Priority definitions:
    - Critical: security problem, severe data loss, complete outage,
      or a broadly blocking regression.
    - High: major functionality broken, important workflow blocked,
      or many users affected.
    - Medium: meaningful defect or enhancement without urgent
      system-wide impact.
    - Low: minor defect, documentation improvement, general question,
      cosmetic concern, or low-impact request.
    """

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class IssueAnalysis(BaseModel):
    """Structured LLM output for a single issue analysis.

    This model is passed directly to provider SDKs as the structured
    output schema. All fields are required and constrained.
    """

    category: IssueCategory = Field(
        description="The classification category for this issue."
    )
    priority_level: IssuePriority = Field(
        description="The priority level based on impact and severity."
    )
    tldr_summary: str = Field(
        description=(
            "A concise one-sentence technical summary of the core issue or request. "
            "Must be useful to a repository maintainer. No speculation or unsupported solutions."
        ),
    )
