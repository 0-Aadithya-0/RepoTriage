"""
Application settings using pydantic-settings.

All configuration is loaded from environment variables and/or a .env file.
Secrets are never logged or exposed in exception messages.
"""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RepoTriage application.

    Reads from environment variables (case-insensitive) and .env file.
    Provider-specific API keys are validated conditionally: only the
    key for the selected provider is required.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application --------------------------------------------------------
    app_name: str = Field(default="RepoTriage")
    app_environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # ---- GitHub -------------------------------------------------------------
    github_token: SecretStr | None = Field(
        default=None,
        description="Optional GitHub personal access token for higher rate limits.",
    )
    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub REST API base URL.",
    )
    github_api_version: str = Field(
        default="2026-03-10",
        description="GitHub API version header value.",
    )
    github_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description="HTTP timeout for GitHub API requests in seconds.",
    )

    # ---- LLM ----------------------------------------------------------------
    llm_provider: Literal["gemini", "cohere"] = Field(
        default="cohere",
        description="Active LLM provider: 'gemini' or 'cohere'.",
    )
    llm_model: str = Field(
        default="command-a-plus-05-2026",
        description="Model identifier for the selected LLM provider.",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="HTTP timeout for LLM API requests in seconds.",
    )
    llm_max_concurrency: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent LLM requests.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient LLM errors.",
    )

    # ---- Provider-specific API keys -----------------------------------------
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key. Required when llm_provider='gemini'.",
    )
    cohere_api_key: SecretStr | None = Field(
        default=None,
        description="Cohere API key. Required when llm_provider='cohere'.",
    )

    # ---- Prompt configuration -----------------------------------------------
    prompts_file: str = Field(
        default="prompts.yaml",
        description="Path to the YAML file containing prompt configuration.",
    )

    # ---- Content truncation -------------------------------------------------
    max_issue_body_length: int = Field(
        default=4096,
        ge=256,
        description=(
            "Maximum character length for issue body content sent to the LLM. "
            "Bodies exceeding this limit are truncated with a marker."
        ),
    )

    # ---- Validators ---------------------------------------------------------

    @model_validator(mode="after")
    def validate_provider_api_key(self) -> "Settings":
        """Ensure the API key for the selected provider is present."""
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'."
            )
        if self.llm_provider == "cohere" and not self.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is required when LLM_PROVIDER is 'cohere'."
            )
        return self
