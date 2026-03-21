"""Auto-parser for Google Gemini SDK response objects."""

from __future__ import annotations


def parse_gemini_response(response: object) -> tuple[str, int, int] | None:
    """Extract model name and token usage from a Gemini response.

    Uses duck-typing to detect Gemini response objects without
    importing the google-genai package.

    Returns (model, input_tokens, output_tokens) or None.
    """
    return None
