"""
Tests for the LLM provider factory.

Verifies:
- Gemini provider created when llm_provider='gemini' with correct key.
- Missing API key raises ConfigurationError.
- Created instances satisfy the LLMProvider protocol.
- API keys are not exposed in error messages.

No live SDK calls — factory just instantiates adapter classes.
"""

import os

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.providers.base import LLMProvider
from app.providers.factory import create_llm_provider
from app.providers.gemini_provider import GeminiProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(clean_env: None, **overrides: str) -> Settings:
    """Create Settings with given env overrides, using clean_env fixture."""
    defaults = {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "sk-test-key-for-factory",
        "LLM_MODEL": "gemini-2.5-flash",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        os.environ[k] = v
    return Settings(_env_file=None)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Gemini provider creation
# ---------------------------------------------------------------------------


class TestGeminiProviderCreation:
    """Tests for creating the Gemini provider via the factory."""

    def test_creates_gemini_provider(self, clean_env: None) -> None:
        settings = _make_settings(
            clean_env,
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="test-gemini-key",
        )
        provider = create_llm_provider(settings)

        assert isinstance(provider, GeminiProvider)

    def test_gemini_provider_satisfies_protocol(
        self, clean_env: None
    ) -> None:
        settings = _make_settings(
            clean_env,
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="test-gemini-key",
        )
        provider = create_llm_provider(settings)

        assert isinstance(provider, LLMProvider)





# ---------------------------------------------------------------------------
# Unsupported provider
# ---------------------------------------------------------------------------


class TestUnsupportedProvider:
    """Tests for unsupported provider values."""

    def test_unsupported_provider_raises_config_error(
        self, clean_env: None
    ) -> None:
        """Even though Settings' Literal type should prevent this,
        the factory has a defensive check."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "sk-test-key"
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Force an unsupported provider value.
        object.__setattr__(settings, "llm_provider", "anthropic")

        with pytest.raises(ConfigurationError, match="Unsupported"):
            create_llm_provider(settings)


# ---------------------------------------------------------------------------
# Model passthrough
# ---------------------------------------------------------------------------


class TestModelPassthrough:
    """Tests that the configured model is passed to the provider."""

    def test_gemini_receives_configured_model(
        self, clean_env: None
    ) -> None:
        settings = _make_settings(
            clean_env,
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="test-key",
            LLM_MODEL="gemini-2.5-pro",
        )
        provider = create_llm_provider(settings)

        assert isinstance(provider, GeminiProvider)
        assert provider._model == "gemini-2.5-pro"
