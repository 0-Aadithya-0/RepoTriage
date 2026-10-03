"""
Tests for the Gemini LLM provider.

Uses unittest.mock to mock the google.genai async client without making real network calls.
Verifies system instruction formatting, successful schema parsing, refusal handling,
and API error mapping.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.errors import APIError
from google.genai import types

from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
)
from app.models.analysis import IssueAnalysis, IssueCategory, IssuePriority
from app.models.github import GitHubIssue
from app.models.prompts import FewShotExample, PromptConfig
from app.providers.gemini_provider import GeminiProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_issue() -> GitHubIssue:
    return GitHubIssue(
        number=42,
        title="Test Issue",
        body="This is a test issue body.",
        html_url="https://github.com/owner/repo/issues/42",
    )


@pytest.fixture
def dummy_prompt_config() -> PromptConfig:
    return PromptConfig(
        version="1.0",
        system_role="You are an AI.",
        task="Analyze this issue.",
        classification_rules="Use Bug for bugs.",
        priority_rules="High for urgent.",
        summary_rules="One sentence.",
        few_shot_examples=[
            FewShotExample(
                title="Bug report",
                body="It crashed.",
                category=IssueCategory.BUG,
                priority=IssuePriority.HIGH,
                summary="It is a bug.",
            )
        ],
    )


@pytest.fixture
def mock_genai_client():
    """Mock the genai.Client class and its async models client."""
    with patch("app.providers.gemini_provider.genai.Client") as mock_client_cls:
        # Create a mock instance for the client
        mock_instance = MagicMock()
        
        # Create a mock for the aio.models.generate_content method
        mock_generate_content = AsyncMock()
        mock_instance.aio.models.generate_content = mock_generate_content
        
        # When genai.Client() is called, return our mock instance
        mock_client_cls.return_value = mock_instance
        
        yield mock_client_cls


@pytest.fixture
def provider() -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        model="gemini-2.5-flash",
        timeout=10.0,
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    parsed_obj: IssueAnalysis | None = None,
    finish_reason: str = "STOP",
    empty_content: bool = False,
) -> MagicMock:
    """Helper to construct a mock GenerateContentResponse."""
    mock_response = MagicMock()
    
    if empty_content:
        mock_response.candidates = []
        return mock_response
        
    mock_candidate = MagicMock()
    mock_candidate.content = "mock content"
    
    # Mock the finish reason enum
    mock_reason = MagicMock()
    mock_reason.name = finish_reason
    mock_candidate.finish_reason = mock_reason
    
    mock_response.candidates = [mock_candidate]
    mock_response.parsed = parsed_obj
    return mock_response


def _make_api_error(code: int, message: str) -> APIError:
    """Helper to construct a real APIError."""
    return APIError(code, {"message": message})


# ---------------------------------------------------------------------------
# Successful Analysis
# ---------------------------------------------------------------------------


class TestSuccessfulAnalysis:
    
    async def test_successful_analysis(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """A valid response parsed into IssueAnalysis is returned directly."""
        expected_analysis = IssueAnalysis(
            category=IssueCategory.BUG,
            priority_level=IssuePriority.MEDIUM,
            tldr_summary="A test summary",
        )
        
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.return_value = _make_mock_response(
            parsed_obj=expected_analysis
        )
        
        result = await provider.analyze_issue(dummy_issue, dummy_prompt_config)
        
        assert result == expected_analysis
        
        # Verify the call arguments
        mock_client_instance.aio.models.generate_content.assert_called_once()
        kwargs = mock_client_instance.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        assert "Title: Test Issue" in kwargs["contents"]
        assert "Body: This is a test issue body." in kwargs["contents"]
        
        config = kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.response_mime_type == "application/json"
        assert config.response_schema == IssueAnalysis
        assert config.temperature == 0.0
        
        # Verify system instructions
        assert "You are an AI." in config.system_instruction
        assert "Analyze this issue." in config.system_instruction
        assert "Bug report" in config.system_instruction
        assert "Bug" in config.system_instruction

# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    
    async def test_timeout_error(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """asyncio.TimeoutError is mapped to LLMTimeoutError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(LLMTimeoutError, match="timed out after 10.0s"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)
            
    async def test_api_error_401(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """401 is mapped to LLMAuthenticationError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.side_effect = _make_api_error(
            code=401, message="Invalid API key"
        )
        
        with pytest.raises(LLMAuthenticationError, match="Authentication failed: Invalid API key"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)
            
    async def test_api_error_429(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """429 is mapped to LLMRateLimitError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.side_effect = _make_api_error(
            code=429, message="Too many requests"
        )
        
        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded: Too many requests"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)
            
    async def test_api_error_500(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """500 is mapped to LLMServerError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.side_effect = _make_api_error(
            code=500, message="Internal Server Error"
        )
        
        with pytest.raises(LLMServerError, match="Provider server error \\(500\\): Internal Server Error"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)
            
    async def test_unexpected_exception(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """Unexpected exceptions are wrapped in LLMServerError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.side_effect = ValueError("Some weird error")
        
        with pytest.raises(LLMServerError, match="Unexpected error: Some weird error"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)

    async def test_api_error_403(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """403 is mapped to LLMAuthenticationError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.side_effect = _make_api_error(
            code=403, message="Permission denied"
        )

        with pytest.raises(LLMAuthenticationError, match="Authentication failed: Permission denied"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)

# ---------------------------------------------------------------------------
# Refusals and Parsing
# ---------------------------------------------------------------------------

class TestRefusalsAndParsing:

    async def test_empty_candidates(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """Empty candidates trigger LLMRefusalError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.return_value = _make_mock_response(
            empty_content=True
        )
        
        with pytest.raises(LLMRefusalError, match="empty response"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)

    async def test_finish_reason_safety(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """Finish reason other than STOP or MAX_TOKENS triggers LLMRefusalError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.return_value = _make_mock_response(
            parsed_obj=None,
            finish_reason="SAFETY"
        )
        
        with pytest.raises(LLMRefusalError, match="Model refused generation. Reason: SAFETY"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)

    async def test_parsed_none(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """If parsed is None (but finish_reason is STOP), raises ParseError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.return_value = _make_mock_response(
            parsed_obj=None,
            finish_reason="STOP"
        )
        
        with pytest.raises(LLMResponseParseError, match="did not return valid JSON"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)

    async def test_parsed_wrong_type(
        self,
        mock_genai_client: MagicMock,
        provider: GeminiProvider,
        dummy_issue: GitHubIssue,
        dummy_prompt_config: PromptConfig,
    ) -> None:
        """If parsed is not an IssueAnalysis instance, raises ParseError."""
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.aio.models.generate_content.return_value = _make_mock_response(
            parsed_obj={"category": "Bug"},  # raw dict instead of Pydantic model
            finish_reason="STOP"
        )
        
        with pytest.raises(LLMResponseParseError, match="not of type IssueAnalysis"):
            await provider.analyze_issue(dummy_issue, dummy_prompt_config)
