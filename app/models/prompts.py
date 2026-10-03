"""
Prompt configuration models for RepoTriage.

These Pydantic models validate the structure of prompts.yaml.
They are used by the prompt loader to ensure all required prompt
fields are present and valid at startup time.
"""

from pydantic import BaseModel, Field

from app.models.analysis import IssueCategory, IssuePriority


class FewShotExample(BaseModel):
    """A single few-shot example for LLM prompt construction."""

    title: str = Field(
        min_length=1,
        description="Example issue title.",
    )
    body: str = Field(
        min_length=1,
        description="Example issue body text.",
    )
    category: IssueCategory = Field(
        description="Expected classification category.",
    )
    priority: IssuePriority = Field(
        description="Expected priority level.",
    )
    summary: str = Field(
        min_length=10,
        max_length=256,
        description="Expected one-sentence summary.",
    )


class PromptConfig(BaseModel):
    """Validated configuration loaded from prompts.yaml.

    All fields are required. The prompt loader fails fast if any
    field is missing or invalid.
    """

    version: str = Field(
        min_length=1,
        description="Prompt configuration version identifier.",
    )
    system_role: str = Field(
        min_length=1,
        description="System role description for the LLM.",
    )
    task: str = Field(
        min_length=1,
        description="Task description explaining what the LLM should do.",
    )
    classification_rules: str = Field(
        min_length=1,
        description="Rules for classifying issues into categories.",
    )
    priority_rules: str = Field(
        min_length=1,
        description="Rules for assigning priority levels.",
    )
    summary_rules: str = Field(
        min_length=1,
        description="Rules for generating the one-sentence summary.",
    )
    few_shot_examples: list[FewShotExample] = Field(
        min_length=1,
        description="At least one few-shot example for the LLM.",
    )
