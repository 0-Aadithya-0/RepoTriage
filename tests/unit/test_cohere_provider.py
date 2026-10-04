"""Unit tests for Cohere response parsing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMResponseParseError
from app.models.analysis import IssueCategory, IssuePriority
from app.models.github import GitHubIssue
from app.models.prompts import FewShotExample, PromptConfig
from app.providers.cohere_provider import CohereProvider


@pytest.fixture
def settings(clean_env: None) -> Settings:
    return Settings(
        llm_provider="cohere",
        llm_model="command-a-plus-05-2026",
        cohere_api_key="test-key",
        llm_max_retries=0,
        _env_file=None,
    )


@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(
        number=34,
        title="Cohere response parsing fails",
        body="The response begins with a thinking block.",
        html_url="https://github.com/owner/repo/issues/34",
    )


@pytest.fixture
def prompt_config() -> PromptConfig:
    return PromptConfig(
        version="1.0",
        system_role="You are an issue triage assistant.",
        task="Classify and summarize the issue.",
        classification_rules="Choose the closest category.",
        priority_rules="Choose a priority based on impact.",
        summary_rules="Write one concise sentence.",
        few_shot_examples=[
            FewShotExample(
                title="Application crashes",
                body="The application crashes during startup.",
                category=IssueCategory.BUG,
                priority=IssuePriority.HIGH,
                summary="The application crashes during the startup sequence.",
            )
        ],
    )


@pytest.fixture
def provider(settings: Settings) -> CohereProvider:
    with patch("app.providers.cohere_provider.cohere.AsyncClient"):
        return CohereProvider(api_key="test-key", settings=settings)


async def test_ignores_thinking_block_and_parses_text(
    provider: CohereProvider,
    issue: GitHubIssue,
    prompt_config: PromptConfig,
) -> None:
    payload = {
        "category": IssueCategory.BUG.value,
        "priority_level": IssuePriority.MEDIUM.value,
        "tldr_summary": "Cohere response parsing fails on thinking content blocks.",
    }
    response = SimpleNamespace(
        message=SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Internal reasoning"),
                SimpleNamespace(type="text", text=json.dumps(payload)),
            ]
        )
    )
    provider._client = MagicMock()
    provider._client.v2.chat = AsyncMock(return_value=response)

    result = await provider.analyze_issue(issue, prompt_config)

    assert result.category is IssueCategory.BUG
    assert result.priority_level is IssuePriority.MEDIUM
    assert result.tldr_summary == payload["tldr_summary"]


async def test_missing_text_block_raises_parse_error(
    provider: CohereProvider,
    issue: GitHubIssue,
    prompt_config: PromptConfig,
) -> None:
    response = SimpleNamespace(
        message=SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Internal reasoning"),
            ]
        )
    )
    provider._client = MagicMock()
    provider._client.v2.chat = AsyncMock(return_value=response)

    with pytest.raises(LLMResponseParseError, match="no text content"):
        await provider.analyze_issue(issue, prompt_config)
