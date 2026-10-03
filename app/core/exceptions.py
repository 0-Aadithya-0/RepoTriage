"""
Domain exception hierarchy for RepoTriage.

All application-specific exceptions inherit from RepoTriageError.
These are translated into appropriate HTTP responses by the API layer's
exception handlers. They never expose API keys, tokens, or internal
request headers.
"""


class RepoTriageError(Exception):
    """Base exception for all RepoTriage domain errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Configuration errors
# --------------------------------------------------------------------------- #


class ConfigurationError(RepoTriageError):
    """Raised when application configuration is invalid or incomplete."""


# --------------------------------------------------------------------------- #
# GitHub client errors
# --------------------------------------------------------------------------- #


class GitHubError(RepoTriageError):
    """Base exception for GitHub API errors."""


class RepositoryNotFoundError(GitHubError):
    """The requested repository does not exist or is not accessible."""


class GitHubRateLimitError(GitHubError):
    """GitHub API rate limit has been exceeded."""

    def __init__(
        self,
        message: str = "GitHub API rate limit exceeded",
        *,
        retry_after: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, detail=detail)


class GitHubTimeoutError(GitHubError):
    """GitHub API request timed out."""


class GitHubServerError(GitHubError):
    """GitHub API returned a server error (5xx)."""


class GitHubAuthenticationError(GitHubError):
    """GitHub token is invalid or lacks required permissions."""


# --------------------------------------------------------------------------- #
# LLM provider errors
# --------------------------------------------------------------------------- #


class LLMError(RepoTriageError):
    """Base exception for LLM provider errors."""


class LLMAuthenticationError(LLMError):
    """LLM API key is invalid or expired."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        *,
        retry_after: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, detail=detail)


class LLMTimeoutError(LLMError):
    """LLM API request timed out."""


class LLMServerError(LLMError):
    """LLM provider returned a server error."""


class LLMResponseParseError(LLMError):
    """LLM response could not be parsed into the expected schema."""


class LLMRefusalError(LLMError):
    """LLM refused to generate content (safety filter or policy)."""
