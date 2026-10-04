import logging

import cohere
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
)
from app.models.analysis import IssueAnalysis
from app.models.github import GitHubIssue
from app.models.prompts import PromptConfig
from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class CohereProvider(LLMProvider):
    """Cohere implementation of the LLMProvider protocol.

    Uses the official cohere SDK and relies on Cohere's native structured
    JSON output mode for guaranteed parsing.
    """

    def __init__(self, api_key: str, settings: Settings) -> None:
        self._settings = settings
        # Initialize the async client. We set max_retries to 0 because
        # we handle retries externally via Tenacity.
        self._client = cohere.AsyncClient(
            api_key=api_key,
            timeout=settings.llm_timeout_seconds,
            client_name="repotriage",
            max_retries=0,
        )

    async def analyze_issue(
        self,
        issue: GitHubIssue,
        prompt_config: PromptConfig,
    ) -> IssueAnalysis:
        system_prompt = self._build_system_prompt(prompt_config)
        user_prompt = self._build_user_prompt(issue, prompt_config)

        # Create a dynamic retry decorator using settings
        retry_decorator = retry(
            # llm_max_retries counts retries after the initial request.
            stop=stop_after_attempt(self._settings.llm_max_retries + 1),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (LLMRateLimitError, LLMTimeoutError, LLMServerError)
            ),
            reraise=True,
        )

        @retry_decorator
        async def _call_api() -> IssueAnalysis:
            try:
                response = await self._client.v2.chat(
                    model=self._settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=cohere.JsonObjectResponseFormatV2(
                        json_schema=IssueAnalysis.model_json_schema(),
                    ),
                )

                # Command A reasoning models can return a thinking block before
                # the final text block. Only text content contains the JSON.
                text_parts = [
                    item.text
                    for item in response.message.content
                    if isinstance(getattr(item, "text", None), str)
                ]
                if not text_parts:
                    raise LLMResponseParseError("Cohere returned no text content.")

                return IssueAnalysis.model_validate_json("".join(text_parts))

            except cohere.errors.UnauthorizedError as exc:
                raise LLMAuthenticationError(f"Authentication failed: {exc}") from exc
            except cohere.errors.TooManyRequestsError as exc:
                raise LLMRateLimitError(
                    "Cohere rate limit exceeded", retry_after=5
                ) from exc
            except (
                cohere.errors.ServiceUnavailableError,
                cohere.errors.InternalServerError,
            ) as exc:
                raise LLMServerError(f"Cohere server error: {exc}") from exc
            except cohere.errors.GatewayTimeoutError as exc:
                raise LLMTimeoutError("Connection to Cohere failed") from exc
            except ValidationError as exc:
                raise LLMResponseParseError(
                    f"Failed to parse Cohere response into IssueAnalysis: {exc}"
                ) from exc
            except (
                LLMAuthenticationError,
                LLMRateLimitError,
                LLMResponseParseError,
                LLMServerError,
                LLMTimeoutError,
            ):
                raise
            except Exception as exc:
                if "timeout" in str(exc).lower():
                    raise LLMTimeoutError("Cohere request timed out") from exc
                raise LLMServerError(f"Unexpected Cohere error: {exc}") from exc

        return await _call_api()

    def _build_system_prompt(self, config: PromptConfig) -> str:
        """Construct the system preamble from the prompt configuration."""
        lines = [config.system_role, "", "TASK:", config.task, ""]

        if config.few_shot_examples:
            lines.extend(["", "EXAMPLES:"])
            for i, example in enumerate(config.few_shot_examples, 1):
                lines.append(f"\n--- Example {i} ---")
                lines.append(f"Input Title: {example.title}")
                lines.append(f"Input Body: {example.body}")
                expected_output = IssueAnalysis(
                    category=example.category,
                    priority_level=example.priority,
                    tldr_summary=example.summary,
                ).model_dump_json(indent=2)
                lines.append(f"Output:\n{expected_output}")

        return "\n".join(lines)

    def _build_user_prompt(self, issue: GitHubIssue, config: PromptConfig) -> str:
        """Construct the user message containing the issue content."""
        body_text = issue.body or "No description provided."

        # Truncate body if it exceeds the configured limit
        if len(body_text) > self._settings.max_issue_body_length:
            truncated_len = self._settings.max_issue_body_length
            body_text = body_text[:truncated_len] + "\n\n... [CONTENT TRUNCATED]"

        return f"Issue Title: {issue.title}\n\nIssue Body:\n{body_text}"
