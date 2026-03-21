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
    rate_limit="50/min",
    tpm_limit="40000/min",
    extract_usage=lambda res: (res['in_tokens'], res['out_tokens'])
)
def custom_anthropic_call(prompt):
    # custom logic here
    pass
```

## Supported Providers

| Provider | SDK Auto-Parsing | Streaming Support | Custom Model Overrides |
|----------|-----------------|-------------------|----------------------|
| OpenAI | Yes (`openai` client) | Yes (chunk calculation) | Yes |
| Anthropic | Yes (`anthropic` client) | Yes | Yes |
| Google Gemini | Yes (`google-genai` client) | Yes | Yes |
| Local/Ollama | No (Cost is $0) | N/A | Rate limiting only |

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
