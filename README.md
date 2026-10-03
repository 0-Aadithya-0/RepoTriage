# RepoTriage — Live GitHub Issue Analyzer

An academic backend project that fetches open issues from any public GitHub repository and uses **Google Gemini 2.5 Flash** to classify, prioritize, and summarize each one — returning a structured JSON response.

---

## Architecture Overview

```
POST /api/v1/analyze
        │
        ▼
  FastAPI Route (app/api/routes.py)
        │
        ▼
  Orchestrator (app/services/analyzer.py)
  ┌─────┴─────────────────────────────┐
  │                                   │
  ▼                                   ▼
GitHubClient                  GeminiProvider
(app/clients/github_client.py) (app/providers/gemini_provider.py)
  │                                   │
  │  fetch open issues                │  structured output
  │  filter PRs                       │  (response_schema=IssueAnalysis)
  │  truncate bodies                  │  asyncio.Semaphore concurrency
  └──────────────┬────────────────────┘
                 │
                 ▼
          AnalyzeResponse
          ┌─────────────────────────────┐
          │ repository: "owner/repo"    │
          │ analyzed_count: N           │
          │ failed_count: M             │
          │ issues: [AnalyzedIssue...]  │
          │ failures: [IssueFailure...] │
          └─────────────────────────────┘
```

---

## Project Structure

```
repotriage/
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI endpoints
│   ├── clients/
│   │   └── github_client.py    # Async GitHub REST client
│   ├── core/
│   │   ├── config.py           # Pydantic-settings configuration
│   │   └── exceptions.py       # Domain exception hierarchy
│   ├── middleware/
│   │   └── error_handlers.py   # Exception → HTTP status mapping
│   ├── models/
│   │   ├── analysis.py         # IssueAnalysis, IssueCategory, IssuePriority
│   │   ├── api.py              # AnalyzeRequest, AnalyzeResponse
│   │   ├── github.py           # GitHubIssue
│   │   └── prompts.py          # PromptConfig, FewShotExample
│   ├── providers/
│   │   ├── base.py             # LLMProvider Protocol
│   │   ├── factory.py          # Provider factory
│   │   └── gemini_provider.py  # Gemini adapter (google-genai SDK)
│   ├── services/
│   │   ├── analyzer.py         # Orchestrator: fetch → analyze → respond
│   │   └── prompt_loader.py    # YAML prompt loader with lru_cache
│   └── main.py                 # FastAPI app factory + lifespan
├── tests/
│   ├── unit/                   # 167 unit tests (no real API calls)
│   └── e2e/                    # Live end-to-end tests (real API calls)
├── prompts.yaml                # LLM prompt configuration
├── .env.example                # Environment variable template
├── requirements.txt
└── requirements-dev.txt
```

---

## Setup

### 1. Prerequisites

- Python 3.12+
- A virtual environment (`.venv` already exists if you've been following along)
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- (Optional) A **GitHub Personal Access Token** for higher rate limits

### 2. Install Dependencies

```powershell
# From the repotriage/ directory
..\.venv\Scripts\pip install -r requirements.txt
..\.venv\Scripts\pip install -r requirements-dev.txt
```

### 3. Configure Environment

```powershell
# Copy the template
copy .env.example .env

# Edit .env and fill in your GEMINI_API_KEY
# Optionally uncomment GITHUB_TOKEN for higher rate limits
notepad .env
```

Your `.env` should look like:

```env
GEMINI_API_KEY=your_actual_api_key_here
# GITHUB_TOKEN=ghp_your_token_here  (optional, recommended)
```

---

## Running the Server

```powershell
# From the repotriage/ directory
..\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

The server starts at **http://127.0.0.1:8000**

- **Interactive API docs**: http://127.0.0.1:8000/docs
- **Health check**: http://127.0.0.1:8000/health

---

## API Usage

### `POST /api/v1/analyze`

Fetch and analyze open issues from a public GitHub repository.

**Request body:**

```json
{
  "owner": "tiangolo",
  "repo": "fastapi",
  "limit": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `owner` | string | required | GitHub username or organization |
| `repo` | string | required | Repository name |
| `limit` | integer | 10 | Issues to analyze (1–10) |

**Example with curl:**

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/analyze `
  -H "Content-Type: application/json" `
  -d '{"owner": "tiangolo", "repo": "fastapi", "limit": 3}'
```

**Example response:**

```json
{
  "repository": "tiangolo/fastapi",
  "requested_limit": 3,
  "analyzed_count": 3,
  "failed_count": 0,
  "issues": [
    {
      "issue_number": 13212,
      "title": "Docs: clarify dependency injection scope",
      "html_url": "https://github.com/tiangolo/fastapi/issues/13212",
      "category": "Documentation",
      "priority_level": "Low",
      "tldr_summary": "User requests clarification in the docs on how dependency scope differs between requests."
    }
  ],
  "failures": []
}
```

### `GET /health`

```powershell
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

---

## Running Tests

### Unit Tests (no API key required)

```powershell
$env:PYTHONPATH="."; ..\.venv\Scripts\python.exe -m pytest tests/unit/ -v
```

### End-to-End Tests (requires real GEMINI_API_KEY in .env)

```powershell
$env:PYTHONPATH="."; ..\.venv\Scripts\python.exe -m pytest tests/e2e/ -v -m e2e
```

### Full Suite (unit only, skip e2e)

```powershell
$env:PYTHONPATH="."; ..\.venv\Scripts\python.exe -m pytest tests/ -m "not e2e" --cov=app --cov-report=term-missing
```

### Coverage Summary

| File | Stmts | Missed | Cover |
|------|------:|-------:|------:|
| `github_client.py` | 80 | 0 | **100%** |
| `analyzer.py` | 51 | 0 | **100%** |
| `routes.py` | 16 | 0 | **100%** |
| `config.py` | 25 | 0 | **100%** |
| `exceptions.py` | 25 | 0 | **100%** |
| `factory.py` | 16 | 0 | **100%** |
| `gemini_provider.py` | 54 | 1 | **98%** |
| `error_handlers.py` | 78 | 3 | **96%** |
| `prompt_loader.py` | 35 | 3 | **91%** |
| `main.py` | 35 | 35 | 0%* |
| **TOTAL** | **510** | **42** | **92%** |

> *`main.py` at 0%: the production lifespan is intentionally excluded from unit tests
> (mocks are injected directly into `app.state`). It is exercised by the E2E suite.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `typing.Protocol` for `LLMProvider` | Structural subtyping — adapters don't inherit, they just match the signature |
| `response_schema=IssueAnalysis` | Native SDK structured output — no manual JSON parsing |
| `asyncio.Semaphore` for concurrency | Prevents LLM API rate limit violations |
| Partial-failure semantics | One failed issue never aborts the whole batch |
| `SecretStr` for API keys | Pydantic prevents accidental logging of secrets |
| `lru_cache` on prompt loader | Prompts YAML is read once at startup, not per-request |
| Domain exception hierarchy | Clean separation: GitHub errors vs. LLM errors vs. config errors |

---

## Academic Notes

This project demonstrates:

1. **Clean Architecture** — each layer has a single responsibility
2. **Dependency Inversion** — the orchestrator depends on the `LLMProvider` protocol, not the Gemini SDK
3. **Structured LLM Outputs** — using `response_schema` instead of prompt engineering for reliable parsing
4. **Async-first Design** — `asyncio` + `httpx` throughout; no blocking I/O
5. **Test Pyramid** — 167 unit tests with mocks, plus live E2E tests against real APIs
6. **Fail-Safe Error Handling** — partial failures, domain exceptions, zero secret leakage
