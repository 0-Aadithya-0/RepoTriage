"""
Async GitHub REST API client for fetching repository issues.

Responsibilities:
- Build and send authenticated (or unauthenticated) requests to GitHub.
- Filter out pull requests from the issues endpoint response.
- Map GitHub HTTP errors to domain exceptions.
- Truncate issue bodies to a configurable limit before returning.
- Never log or expose the GitHub token.

The httpx.AsyncClient is injected via the constructor — it will be
created and closed by FastAPI's lifespan handler in production, and
by MockTransport fixtures in tests.
"""

import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import (
    GitHubAuthenticationError,
    GitHubError,
    GitHubRateLimitError,
    GitHubServerError,
    GitHubTimeoutError,
    RepositoryNotFoundError,
)
from app.models.github import GitHubIssue

logger = logging.getLogger(__name__)

# Truncation marker appended when body is shortened.
_TRUNCATION_MARKER = "\n\n[... body truncated for analysis]"


class GitHubClient:
    """Async client for the GitHub REST API issues endpoint.

    Args:
        http_client: A reusable httpx.AsyncClient instance.
        settings: Application settings (used for base URL, version header,
            timeout, optional token, and body truncation limit).
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self._http = http_client
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_issues(
        self,
        owner: str,
        repo: str,
        limit: int = 10,
    ) -> list[GitHubIssue]:
        """Fetch the newest open issues for a repository.

        Pull requests are excluded. Results are returned in newest-first
        order, up to `limit` items.

        Args:
            owner: GitHub owner or organization.
            repo: Repository name.
            limit: Maximum number of issues to return (after filtering).

        Returns:
            A list of GitHubIssue objects, newest first.

        Raises:
            RepositoryNotFoundError: Repository does not exist or is private.
            GitHubRateLimitError: GitHub API rate limit exceeded.
            GitHubAuthenticationError: Token is invalid or lacks permissions.
            GitHubTimeoutError: Request timed out.
            GitHubServerError: GitHub returned a 5xx status.
            GitHubError: Unexpected GitHub API error.
        """
        url = (
            f"{self._settings.github_api_base_url}"
            f"/repos/{owner}/{repo}/issues"
        )

        headers = self._build_headers()

        # Request more than `limit` to account for pull requests mixed in.
        # per_page=30 gives us enough headroom for most repositories.
        params: dict[str, str | int] = {
            "state": "open",
            "sort": "created",
            "direction": "desc",
            "per_page": 30,
        }

        logger.info(
            "Fetching GitHub issues",
            extra={
                "owner": owner,
                "repo": repo,
                "limit": limit,
                "per_page": params["per_page"],
            },
        )

        response = await self._send_request(url, headers=headers, params=params)

        raw_items = response.json()

        if not isinstance(raw_items, list):
            raise GitHubError(
                f"Unexpected GitHub response type: expected list, "
                f"got {type(raw_items).__name__}"
            )

        issues = self._parse_issues(raw_items, limit)

        logger.info(
            "Fetched GitHub issues",
            extra={
                "owner": owner,
                "repo": repo,
                "total_items_returned": len(raw_items),
                "pull_requests_filtered": len(raw_items) - len(
                    [i for i in raw_items if "pull_request" not in i]
                ),
                "issues_returned": len(issues),
            },
        )

        return issues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build request headers, including optional authentication."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self._settings.github_api_version,
            "User-Agent": "RepoTriage/1.0",
        }

        token = self._settings.github_token
        if token is not None:
            headers["Authorization"] = f"Bearer {token.get_secret_value()}"

        return headers

    async def _send_request(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, str | int],
    ) -> httpx.Response:
        """Send a GET request and translate HTTP errors to domain exceptions."""
        try:
            response = await self._http.get(
                url,
                headers=headers,
                params=params,
                timeout=self._settings.github_timeout_seconds,
                follow_redirects=True,
            )
        except httpx.TimeoutException as exc:
            raise GitHubTimeoutError(
                "GitHub API request timed out"
            ) from exc
        except httpx.HTTPError as exc:
            raise GitHubError(
                f"GitHub API request failed: {type(exc).__name__}"
            ) from exc

        self._check_response_status(response)
        return response

    def _check_response_status(self, response: httpx.Response) -> None:
        """Translate non-2xx HTTP status codes to domain exceptions."""
        status = response.status_code

        if 200 <= status < 300:
            return

        if status == 404:
            raise RepositoryNotFoundError(
                "Repository not found. Verify the owner and repository name, "
                "or check that the repository is not private."
            )

        if status == 401:
            raise GitHubAuthenticationError(
                "GitHub authentication failed. The token may be invalid or expired."
            )

        if status == 403:
            # Check for rate limiting vs. permission denied.
            remaining = response.headers.get("x-ratelimit-remaining")
            if remaining == "0":
                retry_after = self._parse_retry_after(response)
                raise GitHubRateLimitError(
                    "GitHub API rate limit exceeded. "
                    "Consider setting GITHUB_TOKEN for higher limits.",
                    retry_after=retry_after,
                )
            raise GitHubAuthenticationError(
                "GitHub access forbidden. The token may lack required permissions."
            )

        if status == 422:
            raise GitHubError(
                "GitHub rejected the request as unprocessable (422). "
                "Verify the owner and repository name."
            )

        if 500 <= status < 600:
            raise GitHubServerError(
                f"GitHub server error (HTTP {status}). Try again later."
            )

        # Catch-all for unexpected status codes.
        raise GitHubError(
            f"Unexpected GitHub API response (HTTP {status})."
        )

    def _parse_retry_after(self, response: httpx.Response) -> int | None:
        """Extract Retry-After from response headers if available."""
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return int(retry_after)
            except ValueError:
                pass
        return None

    def _parse_issues(
        self,
        raw_items: list[dict[str, Any]],
        limit: int,
    ) -> list[GitHubIssue]:
        """Filter PRs, parse into GitHubIssue models, and truncate bodies.

        Items are already in newest-first order from the API.
        """
        issues: list[GitHubIssue] = []
        max_body = self._settings.max_issue_body_length

        for item in raw_items:
            if len(issues) >= limit:
                break

            # Skip pull requests — they have a "pull_request" key.
            if "pull_request" in item:
                continue

            body = item.get("body") or ""
            if len(body) > max_body:
                body = body[:max_body] + _TRUNCATION_MARKER

            issue = GitHubIssue(
                number=item["number"],
                title=item.get("title", ""),
                body=body,
                html_url=item.get("html_url", ""),
            )
            issues.append(issue)

        return issues
