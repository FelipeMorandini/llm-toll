"""SDK auto-detection and parsing for extracting token usage from responses."""

from __future__ import annotations

from llm_toll.parsers.anthropic import parse_anthropic_response
from llm_toll.parsers.gemini import parse_gemini_response
from llm_toll.parsers.openai import parse_openai_response

UsageInfo = tuple[str, int, int]  # (model_name, input_tokens, output_tokens)


def auto_detect_usage(response: object) -> UsageInfo | None:
    """Auto-detect the SDK and extract token usage from a response object.

    Tries each parser in sequence and returns the first successful result.
    Returns None if no parser can handle the response.
    """
    for parser in (parse_openai_response, parse_anthropic_response, parse_gemini_response):
        result = parser(response)
        if result is not None:
            return result
    return None
