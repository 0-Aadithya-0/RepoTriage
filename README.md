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
- `pip`, included with a standard Python installation
- An API key for the configured LLM provider:
  - **Cohere** (the default provider), or
  - **Gemini** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- (Optional) A **GitHub Personal Access Token** for higher rate limits

### 2. Create and Activate the Virtual Environment

Run the following commands from the project root.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal prompt should begin with `(.venv)`. You can
confirm that the environment is active with:

```bash
python --version
python -m pip --version
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current
> terminal, then activate the environment again.

### 3. Install Dependencies

With `.venv` activated:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` already includes the production requirements, so for
development you may install only that file:

```bash
python -m pip install -r requirements-dev.txt
```

When returning to the project later, activate `.venv` again before running the
server, frontend, or tests. Run `deactivate` to leave the environment.

### 4. Configure Environment

```powershell
# Windows PowerShell
copy .env.example .env

# Edit .env and add the API key for your selected provider
# Optionally uncomment GITHUB_TOKEN for higher rate limits
notepad .env
```

On macOS/Linux, copy the template with:

```bash
cp .env.example .env
```

For the default Cohere provider, your `.env` should include:

```env
LLM_PROVIDER=cohere
LLM_MODEL=command-a-plus-05-2026
COHERE_API_KEY=your_actual_cohere_api_key
# GITHUB_TOKEN=github_token_for_public_repositories
```

To use Gemini instead:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.8-flash
GEMINI_API_KEY=your_actual_api_key_here
```

---

## Running the Server

```bash
python -m uvicorn app.main:app --reload --port 8000
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
| `gemini_provider.py` | 54 | 0 | **100%** |
| `main.py` | 35 | 0 | **100%** |
| `error_handlers.py` | 78 | 3 | **96%** |
| `prompt_loader.py` | 35 | 3 | **91%** |
| **TOTAL** | **510** | **6** | **99%** |

> Note: Running only unit tests (`-m "not e2e"`) yields 92% coverage because the production lifespan in `main.py` is excluded. Running the full suite exercises the live app and brings coverage to 99%.

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
