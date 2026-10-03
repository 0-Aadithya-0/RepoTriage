"""
Issue analysis orchestrator for RepoTriage.

Coordinates the full analysis pipeline:
1. Fetch issues from GitHub via GitHubClient.
2. Analyze each issue concurrently via LLMProvider (with bounded concurrency).
3. Collect successes and failures into a unified AnalyzeResponse.

Implements partial-failure semantics: individual LLM failures do not
abort the entire batch. Failed issues are reported in the response
alongside successful analyses.
"""

import asyncio
import logging

from app.clients.github_client import GitHubClient
from app.core.exceptions import LLMError
from app.models.analysis import IssueAnalysis
from app.models.api import AnalyzedIssue, AnalyzeResponse, IssueFailure
from app.models.github import GitHubIssue
from app.models.prompts import PromptConfig
from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Internal container for a single issue's analysis outcome.

    Holds either a successful IssueAnalysis or an error message,
    never both. Used only within the orchestrator before mapping
    into the public API response models.
    """

    __slots__ = ("issue", "analysis", "error")

    def __init__(
        self,
        issue: GitHubIssue,
        *,
        analysis: IssueAnalysis | None = None,
        error: str | None = None,
    ) -> None:
        self.issue = issue
        self.analysis = analysis
        self.error = error

    @property
    def is_success(self) -> bool:
        return self.analysis is not None


async def analyze_repository(
    *,
    owner: str,
    repo: str,
    limit: int,
    github_client: GitHubClient,
    llm_provider: LLMProvider,
    prompt_config: PromptConfig,
    max_concurrency: int = 3,
) -> AnalyzeResponse:
    """Orchestrate the full issue analysis pipeline.

    Fetches issues from GitHub, analyzes each one via the LLM provider
    with bounded concurrency, and returns an AnalyzeResponse containing
    both successes and failures.

    Args:
        owner: GitHub repository owner.
        repo: Repository name.
        limit: Maximum number of issues to fetch and analyze.
        github_client: Initialized GitHubClient instance.
        llm_provider: LLMProvider adapter (e.g. GeminiProvider).
        prompt_config: Validated prompt configuration from YAML.
        max_concurrency: Maximum concurrent LLM requests.

    Returns:
        AnalyzeResponse with analyzed issues and any failures.

    Raises:
        GitHubError subclasses: If the GitHub fetch itself fails.
            These are not caught here — they propagate to the API layer.
    """
    logger.info(
        "Starting analysis pipeline",
        extra={"owner": owner, "repo": repo, "limit": limit},
    )

    # Step 1: Fetch issues from GitHub
    issues = await github_client.fetch_issues(owner, repo, limit=limit)

    logger.info(
        "Fetched issues, starting LLM analysis",
        extra={"issue_count": len(issues), "max_concurrency": max_concurrency},
    )

    if not issues:
        return AnalyzeResponse(
            repository=f"{owner}/{repo}",
            requested_limit=limit,
            analyzed_count=0,
            failed_count=0,
            issues=[],
            failures=[],
        )

    # Step 2: Analyze each issue with bounded concurrency
    results = await _analyze_issues_concurrently(
        issues=issues,
        llm_provider=llm_provider,
        prompt_config=prompt_config,
        max_concurrency=max_concurrency,
    )

    # Step 3: Separate successes and failures
    analyzed: list[AnalyzedIssue] = []
    failures: list[IssueFailure] = []

    for result in results:
        if result.is_success:
            assert result.analysis is not None  # Type narrowing
            analyzed.append(
                AnalyzedIssue(
                    issue_number=result.issue.number,
                    title=result.issue.title,
                    html_url=result.issue.html_url,
                    category=result.analysis.category,
                    priority_level=result.analysis.priority_level,
                    tldr_summary=result.analysis.tldr_summary,
                )
            )
        else:
            failures.append(
                IssueFailure(
                    issue_number=result.issue.number,
                    title=result.issue.title,
                    html_url=result.issue.html_url,
                    error=result.error or "Unknown error",
                )
            )

    logger.info(
        "Analysis pipeline complete",
        extra={
            "owner": owner,
            "repo": repo,
            "analyzed_count": len(analyzed),
            "failed_count": len(failures),
        },
    )

    return AnalyzeResponse(
        repository=f"{owner}/{repo}",
        requested_limit=limit,
        analyzed_count=len(analyzed),
        failed_count=len(failures),
        issues=analyzed,
        failures=failures,
    )


async def _analyze_issues_concurrently(
    *,
    issues: list[GitHubIssue],
    llm_provider: LLMProvider,
    prompt_config: PromptConfig,
    max_concurrency: int,
) -> list[AnalysisResult]:
    """Analyze a batch of issues with bounded concurrency.

    Uses an asyncio.Semaphore to limit simultaneous LLM requests.
    Each issue is analyzed independently; failures do not affect
    other issues in the batch.

    Returns results in the same order as the input issues.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _analyze_one(issue: GitHubIssue) -> AnalysisResult:
        async with semaphore:
            try:
                analysis = await llm_provider.analyze_issue(issue, prompt_config)
                logger.debug(
                    "Issue analyzed successfully",
                    extra={"issue_number": issue.number},
                )
                return AnalysisResult(issue, analysis=analysis)
            except LLMError as e:
                logger.warning(
                    "LLM analysis failed for issue",
                    extra={
                        "issue_number": issue.number,
                        "error_type": type(e).__name__,
                        "error_message": e.message,
                    },
                )
                return AnalysisResult(issue, error=e.message)
            except Exception as e:
                # Defensive catch-all: should not happen if the provider
                # maps all errors to LLMError, but we never let an
                # unhandled exception crash the entire batch.
                logger.error(
                    "Unexpected error analyzing issue",
                    extra={
                        "issue_number": issue.number,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
                return AnalysisResult(issue, error=f"Unexpected error: {str(e)}")

    tasks = [asyncio.create_task(_analyze_one(issue)) for issue in issues]
    return list(await asyncio.gather(*tasks))
