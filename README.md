# llm-toll

A lightweight, drop-in Python decorator to track costs, monitor token usage, and enforce budget and rate limits for LLM API calls.

## Overview

`llm_toll` is a developer tool designed for local prototyping and small-scale production scripts. By simply wrapping a function with `@track_costs`, developers can automatically log token usage, calculate the exact cost of the run in USD, and halt execution if a hard-coded budget or API rate limit is breached.

## Features

- **Drop-In Decorator** — Minimal code intrusion. Just add `@track_costs` above any function making an LLM call.
- **Multi-Provider Support** — Built-in pricing matrices for OpenAI, Anthropic, Gemini, and general OpenAI-compatible endpoints.
- **Hard Budget Caps** — Prevents functions from executing if the cumulative cost exceeds a defined threshold.
- **Rate Limiting** — Local enforcement of RPM and TPM to prevent HTTP 429 errors.
- **Local Persistence** — SQLite-backed usage tracking across multiple script runs and days.
- **Cost Reporting** — Clean, color-coded terminal summary of cost per call and total session cost.

## Quick Start

### Installation

```bash
pip install llm-toll
# or, with uv
uv add llm-toll
```

### Basic Usage (Auto-detect)

For users utilizing standard SDKs, the decorator infers the model and token count from the response object.

```python
from llm_toll import track_costs

@track_costs(project="my_scraper", max_budget=2.00, reset="monthly")
def generate_summary(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}]
    )
    return response  # Decorator parses the usage from this object
```

### Advanced Usage (Rate Limits & Explicit Models)

For custom setups or raw API requests, users can explicitly state the model and rate limits.

```python
from llm_toll import track_costs

@track_costs(
    model="claude-sonnet-4-20250514",
    rate_limit=50,       # max 50 requests per minute
    tpm_limit=40000,     # max 40k tokens per minute
    extract_usage=lambda res: (res['model'], res['in_tokens'], res['out_tokens'])
)
def custom_anthropic_call(prompt):
    # custom logic here
    pass
```

Rate limits use a sliding-window algorithm. When a limit is reached, `LocalRateLimitError` is raised with a `retry_after` attribute indicating how long to wait.

### Streaming Support

The decorator automatically detects streaming responses (generators). Cost is tracked after the stream is fully consumed.

```python
from llm_toll import track_costs

@track_costs(project="my_app", max_budget=5.00)
def stream_response(text):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}],
        stream=True,
        stream_options={"include_usage": True},  # recommended for accurate counts
    )

for chunk in stream_response("Hello"):
    print(chunk.choices[0].delta.content, end="")
# Cost is logged automatically after the stream completes
```

> **Note:** For accurate token counts with OpenAI streaming, pass `stream_options={"include_usage": True}`. Without it, output tokens are estimated using a character-based heuristic.

### Async Support

The decorator auto-detects async functions and async generators — no changes needed:

```python
from llm_toll import track_costs

@track_costs(project="my_app", max_budget=5.00)
async def async_chat(text):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}]
    )
    return response

@track_costs(project="my_app")
async def async_stream(text):
    stream = await client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": text}],
        stream=True, stream_options={"include_usage": True},
    )
    async for chunk in stream:
        yield chunk
```

SQLite operations run in a thread pool (`asyncio.to_thread`) so the event loop is never blocked.

## Supported Providers

| Provider | SDK Auto-Parsing | Streaming Support | Custom Model Overrides |
|----------|-----------------|-------------------|----------------------|
| OpenAI | Yes (`openai` client) | Yes (chunk calculation) | Yes |
| Anthropic | Yes (`anthropic` client) | Yes | Yes |
| Google Gemini | Yes (`google-genai` client) | Yes | Yes |
| Local/Ollama | Via OpenAI-compat API | N/A | Rate limiting only ($0 cost) |

### Local/Ollama Models

Local models (`ollama/`, `local/`, `llama.cpp/` prefixes) are tracked at $0 cost. Rate limiting still applies — useful for managing local GPU resources.

```python
from llm_toll import track_costs

@track_costs(
    model="ollama/llama3",
    rate_limit=10,       # limit local GPU to 10 RPM
    extract_usage=lambda r: ("ollama/llama3", r["prompt_tokens"], r["completion_tokens"])
)
def local_inference(prompt):
    # Ollama call here
    pass
```

> **Tip:** Ollama's API is OpenAI-compatible, so if you use the `openai` client pointed at `localhost:11434`, auto-parsing works automatically.

## Error Handling

```python
from llm_toll.exceptions import BudgetExceededError, LocalRateLimitError

try:
    result = generate_summary("some text")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
except LocalRateLimitError as e:
    print(f"Rate limit hit: {e}")
```

## CLI Dashboard

View costs and usage from the terminal:

```bash
# Show cost summary across all projects
llm-toll --stats

# Filter by project or model
llm-toll --stats --project my_scraper
llm-toll --stats --model gpt-4o

# Reset a project's budget counter
llm-toll --reset --project my_scraper

# Export usage logs to CSV
llm-toll --export csv > usage.csv
llm-toll --export csv --project my_scraper --output report.csv
```

## Development

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/llm_toll
```

## License

MIT License — see [LICENSE](LICENSE) for details.
