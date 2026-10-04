"""
Tests for application Settings (pydantic-settings configuration).

Validates environment variable loading, provider API key conditional
validation, default values, and secret safety.
"""

import os

import pytest
from pydantic import ValidationError

from app.core.config import Settings


class TestSettingsDefaults:
    """Verify default values when minimal environment is set."""

    def test_defaults_with_cohere_key(self, clean_env: None) -> None:
        """With only COHERE_API_KEY set, all defaults should apply."""
        os.environ["COHERE_API_KEY"] = "cohere-test-key"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.app_name == "RepoTriage"
        assert settings.app_environment == "development"
        assert settings.log_level == "INFO"
        assert settings.github_api_base_url == "https://api.github.com"
        assert settings.github_api_version == "2026-03-10"
        assert settings.github_timeout_seconds == 15.0
        assert settings.llm_provider == "cohere"
        assert settings.llm_model == "command-a-plus-05-2026"
        assert settings.llm_timeout_seconds == 30.0
        assert settings.llm_max_concurrency == 3
        assert settings.llm_max_retries == 3
        assert settings.prompts_file == "prompts.yaml"
        assert settings.max_issue_body_length == 4096

    def test_github_token_defaults_to_none(self, clean_env: None) -> None:
        os.environ["COHERE_API_KEY"] = "cohere-test-key"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.github_token is None


class TestProviderKeyValidation:
    """Verify that only the selected provider's API key is required."""

    def test_gemini_requires_gemini_key(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        # No GEMINI_API_KEY set
        with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_gemini_key_accepted_when_present(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "gemini-valid-key"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.gemini_api_key is not None
        assert settings.gemini_api_key.get_secret_value() == "gemini-valid-key"


class TestUnsupportedProvider:
    """Verify that an unsupported provider is rejected clearly."""

    def test_invalid_provider_rejected(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "anthropic"
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]


class TestSecretSafety:
    """Verify that secrets are not leaked through string representations."""

    def test_gemini_key_hidden_in_repr(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-super-secret-key-12345"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        repr_str = repr(settings)
        assert "sk-super-secret-key-12345" not in repr_str

    def test_gemini_key_hidden_in_str(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-super-secret-key-12345"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        str_str = str(settings)
        assert "sk-super-secret-key-12345" not in str_str

    def test_github_token_hidden_when_set(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-test"
        os.environ["GITHUB_TOKEN"] = "ghp_super_secret_token"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.github_token is not None
        assert settings.github_token.get_secret_value() == "ghp_super_secret_token"
        assert "ghp_super_secret_token" not in repr(settings)


class TestGitHubTokenOptional:
    """Verify GitHub token is optional."""

    def test_no_token_is_valid(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-test"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.github_token is None

    def test_token_stored_as_secret(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-test"
        os.environ["GITHUB_TOKEN"] = "ghp_token123"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.github_token is not None
        assert settings.github_token.get_secret_value() == "ghp_token123"


class TestEnvironmentOverrides:
    """Verify that environment variables override defaults."""

    def test_custom_values(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LLM_MODEL"] = "gemini-2.5-flash"
        os.environ["LLM_MAX_CONCURRENCY"] = "5"
        os.environ["GITHUB_TIMEOUT_SECONDS"] = "20.0"
        os.environ["LOG_LEVEL"] = "DEBUG"

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.llm_provider == "gemini"
        assert settings.llm_model == "gemini-2.5-flash"
        assert settings.llm_max_concurrency == 5
        assert settings.github_timeout_seconds == 20.0
        assert settings.log_level == "DEBUG"

    def test_max_concurrency_bounds(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-test"
        os.environ["LLM_MAX_CONCURRENCY"] = "11"
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_timeout_must_be_positive(self, clean_env: None) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-test"
        os.environ["GITHUB_TIMEOUT_SECONDS"] = "0"
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]
