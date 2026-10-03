"""
LLM provider factory for RepoTriage.

Reads the selected provider from Settings and creates the appropriate
adapter. Only the selected provider's SDK is imported (lazy imports).
Validates that the required API key exists at creation time (fail-fast).
Never exposes API keys in logs or exceptions.
"""

import logging

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the LLM provider adapter based on application settings.

    This function is called once during FastAPI lifespan startup.
    It uses lazy imports so that the SDK is only loaded when needed.

    Args:
        settings: Application settings containing provider selection
            and API key configuration.

    Returns:
        An LLMProvider-satisfying adapter instance.

    Raises:
        ConfigurationError: If the provider is unsupported or the
            required API key is missing.
    """
    provider = settings.llm_provider

    logger.info(
        "Creating LLM provider",
        extra={
            "provider": provider,
            "model": settings.llm_model,
        },
    )

    match provider:
        case "gemini":
            return _create_gemini_provider(settings)
        case _:
            # This should be unreachable due to Literal["gemini"]
            # validation in Settings, but we handle it defensively.
            raise ConfigurationError(
                f"Unsupported LLM provider: '{provider}'. "
                "Supported providers: 'gemini'."
            )


def _create_gemini_provider(settings: Settings) -> LLMProvider:
    """Create and return a Gemini provider adapter.

    Lazily imports the Gemini adapter module. API key presence is
    guaranteed by the Settings model validation.
    """
    # Lazy import
    from app.providers.gemini_provider import GeminiProvider

    return GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
