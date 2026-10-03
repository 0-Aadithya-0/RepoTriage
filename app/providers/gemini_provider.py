"""
Google Gemini LLM provider adapter.

Implements the LLMProvider protocol using the google-genai SDK's
async client with native Pydantic structured output.
"""

import asyncio
import json
from typing import NoReturn

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
)
from app.models.analysis import IssueAnalysis
from app.models.github import GitHubIssue
from app.models.prompts import PromptConfig


class GeminiProvider:
    """Gemini adapter satisfying the LLMProvider protocol.

    Uses google.genai.Client.aio.models.generate_content() with
    response_schema for schema-constrained output.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        
        # Initialize the synchronous/asynchronous client structure
        self._client = genai.Client(api_key=self._api_key)

    async def analyze_issue(
        self,
        issue: GitHubIssue,
        prompt_config: PromptConfig,
    ) -> IssueAnalysis:
        """Analyze a GitHub issue using Gemini structured output.

        Constructs the system instruction from the prompt_config,
        applies the IssueAnalysis schema, and maps API errors to domain exceptions.
        """
        system_instruction = self._build_system_instruction(prompt_config)
        user_prompt = f"Title: {issue.title}\n\nBody: {issue.body}"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=IssueAnalysis,
            temperature=0.0,
        )

        try:
            # We use asyncio.wait_for to enforce the timeout, as http_options
            # isn't robustly documented for the new SDK's async client yet.
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=config,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"Request timed out after {self._timeout}s") from e
        except APIError as e:
            self._map_api_error(e)
        except Exception as e:
            # Catch unexpected errors to ensure we always raise domain exceptions
            raise LLMServerError(f"Unexpected error: {str(e)}") from e

        if not response.candidates or not response.candidates[0].content:
            raise LLMRefusalError("Model returned an empty response")
            
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and finish_reason.name not in ("STOP", "MAX_TOKENS"):
            raise LLMRefusalError(f"Model refused generation. Reason: {finish_reason.name}")

        if not response.parsed:
            raise LLMResponseParseError(
                "Model did not return valid JSON matching the requested schema."
            )

        if not isinstance(response.parsed, IssueAnalysis):
            raise LLMResponseParseError(
                "Parsed response is not of type IssueAnalysis."
            )

        return response.parsed

    def _build_system_instruction(self, config: PromptConfig) -> str:
        """Format the system role, task description, and few-shot examples."""
        parts = [
            config.system_role,
            "",
            config.task,
            "",
            config.classification_rules,
            "",
            config.priority_rules,
            "",
            config.summary_rules,
            "",
            "Examples:",
        ]
        
        for ex in config.few_shot_examples:
            expected_json = {
                "category": ex.category.value,
                "priority_level": ex.priority.value,
                "tldr_summary": ex.summary,
            }
            parts.extend([
                "---",
                f"Input Issue:\nTitle: {ex.title}\nBody: {ex.body}",
                "Expected Output JSON:",
                json.dumps(expected_json, indent=2),
            ])
            
        return "\n".join(parts)

    def _map_api_error(self, e: APIError) -> NoReturn:
        """Map google-genai APIError to RepoTriage domain exceptions."""
        code = e.code
        
        if code in (401, 403):
            raise LLMAuthenticationError(f"Authentication failed: {e.message}") from e
        elif code == 429:
            raise LLMRateLimitError(f"Rate limit exceeded: {e.message}") from e
        elif code and code >= 500:
            raise LLMServerError(f"Provider server error ({code}): {e.message}") from e
        else:
            raise LLMServerError(f"API Error ({code}): {e.message}") from e
