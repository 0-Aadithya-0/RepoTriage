"""
Exception-to-HTTP response handlers for the FastAPI application.

Maps domain exceptions from the RepoTriage exception hierarchy
into structured JSON error responses with appropriate HTTP status codes.
Never exposes API keys, tokens, or internal stack traces.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.exceptions import (
    ConfigurationError,
    GitHubAuthenticationError,
    GitHubRateLimitError,
    GitHubServerError,
    GitHubTimeoutError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
    RepositoryNotFoundError,
    RepoTriageError,
)

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
    *,
    detail: str | None = None,
    retry_after: int | None = None,
) -> JSONResponse:
    """Build a consistent JSON error response."""
    body: dict[str, str | int | None] = {
        "error": error_type,
        "message": message,
    }
    if detail:
        body["detail"] = detail
    if retry_after is not None:
        body["retry_after_seconds"] = retry_after

    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)

    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=headers or None,
    )


# --------------------------------------------------------------------------- #
# Handler functions
# --------------------------------------------------------------------------- #


async def repository_not_found_handler(
    request: Request, exc: RepositoryNotFoundError
) -> JSONResponse:
    logger.info("Repository not found", extra={"path": str(request.url)})
    return _error_response(404, "repository_not_found", exc.message)


async def github_auth_handler(
    request: Request, exc: GitHubAuthenticationError
) -> JSONResponse:
    logger.warning("GitHub authentication error", extra={"path": str(request.url)})
    return _error_response(401, "github_authentication_error", exc.message)


async def github_rate_limit_handler(
    request: Request, exc: GitHubRateLimitError
) -> JSONResponse:
    logger.warning(
        "GitHub rate limit exceeded",
        extra={"path": str(request.url), "retry_after": exc.retry_after},
    )
    return _error_response(
        429, "github_rate_limit", exc.message, retry_after=exc.retry_after
    )


async def github_timeout_handler(
    request: Request, exc: GitHubTimeoutError
) -> JSONResponse:
    logger.warning("GitHub timeout", extra={"path": str(request.url)})
    return _error_response(504, "github_timeout", exc.message)


async def github_server_error_handler(
    request: Request, exc: GitHubServerError
) -> JSONResponse:
    logger.error("GitHub server error", extra={"path": str(request.url)})
    return _error_response(502, "github_server_error", exc.message)


async def llm_auth_handler(
    request: Request, exc: LLMAuthenticationError
) -> JSONResponse:
    logger.warning("LLM authentication error", extra={"path": str(request.url)})
    return _error_response(
        401, "llm_authentication_error",
        "LLM provider authentication failed. Check your API key configuration.",
    )


async def llm_rate_limit_handler(
    request: Request, exc: LLMRateLimitError
) -> JSONResponse:
    logger.warning("LLM rate limit exceeded", extra={"path": str(request.url)})
    return _error_response(
        429, "llm_rate_limit", exc.message, retry_after=exc.retry_after
    )


async def llm_timeout_handler(
    request: Request, exc: LLMTimeoutError
) -> JSONResponse:
    logger.warning("LLM timeout", extra={"path": str(request.url)})
    return _error_response(504, "llm_timeout", exc.message)


async def llm_server_error_handler(
    request: Request, exc: LLMServerError
) -> JSONResponse:
    logger.error("LLM server error", extra={"path": str(request.url)})
    return _error_response(502, "llm_server_error", exc.message)


async def llm_parse_error_handler(
    request: Request, exc: LLMResponseParseError
) -> JSONResponse:
    logger.error("LLM response parse error", extra={"path": str(request.url)})
    return _error_response(
        502, "llm_parse_error",
        "The LLM returned a response that could not be parsed into the expected schema.",
    )


async def llm_refusal_handler(
    request: Request, exc: LLMRefusalError
) -> JSONResponse:
    logger.warning("LLM refused generation", extra={"path": str(request.url)})
    return _error_response(
        422, "llm_refusal",
        "The LLM refused to generate content for one or more issues.",
    )


async def config_error_handler(
    request: Request, exc: ConfigurationError
) -> JSONResponse:
    logger.critical("Configuration error", extra={"path": str(request.url)})
    return _error_response(500, "configuration_error", exc.message)


async def repotriage_error_handler(
    request: Request, exc: RepoTriageError
) -> JSONResponse:
    """Catch-all for any RepoTriageError not handled by a more specific handler."""
    logger.error(
        "Unhandled RepoTriage error",
        extra={"path": str(request.url), "error_type": type(exc).__name__},
    )
    return _error_response(500, "internal_error", exc.message)


async def validation_error_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors for malformed request bodies."""
    logger.info(
        "Request validation error",
        extra={"path": str(request.url), "error_count": exc.error_count()},
    )
    return _error_response(
        422, "validation_error",
        "Request validation failed.",
        detail=str(exc),
    )


async def generic_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last-resort handler. Never leaks internal details."""
    logger.exception(
        "Unhandled exception",
        extra={"path": str(request.url), "error_type": type(exc).__name__},
    )
    return _error_response(
        500, "internal_error",
        "An unexpected internal error occurred. Please try again later.",
    )


# --------------------------------------------------------------------------- #
# Registration helper
# --------------------------------------------------------------------------- #


def register_error_handlers(app: "fastapi.FastAPI") -> None:
    """Register all exception handlers on the FastAPI application.

    Handlers are registered from most-specific to least-specific.
    FastAPI matches the most specific exception class first.
    """
    import fastapi

    # GitHub errors
    app.add_exception_handler(RepositoryNotFoundError, repository_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(GitHubAuthenticationError, github_auth_handler)  # type: ignore[arg-type]
    app.add_exception_handler(GitHubRateLimitError, github_rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(GitHubTimeoutError, github_timeout_handler)  # type: ignore[arg-type]
    app.add_exception_handler(GitHubServerError, github_server_error_handler)  # type: ignore[arg-type]

    # LLM errors
    app.add_exception_handler(LLMAuthenticationError, llm_auth_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LLMRateLimitError, llm_rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LLMTimeoutError, llm_timeout_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LLMServerError, llm_server_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LLMResponseParseError, llm_parse_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LLMRefusalError, llm_refusal_handler)  # type: ignore[arg-type]

    # Configuration and catch-all
    app.add_exception_handler(ConfigurationError, config_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RepoTriageError, repotriage_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_error_handler)
