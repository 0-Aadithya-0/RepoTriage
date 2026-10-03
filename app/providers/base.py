"""
LLM provider protocol for RepoTriage.

Defines the typed interface that all LLM provider adapters must satisfy.
Uses typing.Protocol for structural subtyping — adapters do not need to
inherit from this class, they just need to implement the method signature.

This keeps the service layer decoupled from any specific LLM SDK.
"""

from typing import Protocol, runtime_checkable

from app.models.analysis import IssueAnalysis
from app.models.github import GitHubIssue
from app.models.prompts import PromptConfig


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM provider adapters.

    Any class with an `analyze_issue` async method matching this signature
    satisfies this protocol — no inheritance required.

    The `runtime_checkable` decorator allows `isinstance()` checks,
    which is useful for validation in tests and the factory.
    """

    async def analyze_issue(
        self,
        issue: GitHubIssue,
        prompt_config: PromptConfig,
    ) -> IssueAnalysis:
        """Analyze a single GitHub issue using the LLM.

        Must use native structured output (schema-constrained generation)
        to produce a valid IssueAnalysis. Must not use regex parsing,
        json.loads on freeform text, or prompt-only JSON instructions.

        Args:
            issue: The GitHub issue to analyze.
            prompt_config: The validated prompt configuration from YAML.

        Returns:
            A validated IssueAnalysis instance.

        Raises:
            LLMAuthenticationError: API key is invalid or expired.
            LLMRateLimitError: Provider rate limit exceeded.
            LLMTimeoutError: Request timed out.
            LLMServerError: Provider returned a server error.
            LLMResponseParseError: Response could not be parsed into schema.
            LLMRefusalError: Model refused to generate content.
        """
        ...
