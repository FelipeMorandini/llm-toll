"""Auto-parser for Google Gemini SDK response objects."""

from __future__ import annotations


def parse_gemini_response(response: object) -> tuple[str, int, int] | None:
    """Extract model name and token usage from a Gemini response.

    Uses duck-typing to detect ``GenerateContentResponse`` objects from
    the ``google-genai`` SDK without importing the package.  Checks for
    the ``candidates`` and ``usage_metadata`` attributes as the primary
    discriminators.

    The model name is read from ``model_version`` when available.
    Unlike OpenAI/Anthropic responses, Gemini may not include the model
    name — an empty string is returned in that case so that cost
    tracking still proceeds (with a pricing warning).

    Returns ``(model, input_tokens, output_tokens)`` or ``None``.
    """
    if not (hasattr(response, "candidates") and hasattr(response, "usage_metadata")):
        return None

    # Reject OpenAI-like objects that also have a choices attribute
    if hasattr(response, "choices"):
        return None

    usage_metadata = response.usage_metadata
    if usage_metadata is None:
        return None

    if not (
        hasattr(usage_metadata, "prompt_token_count")
        and hasattr(usage_metadata, "candidates_token_count")
    ):
        return None

    model = getattr(response, "model_version", "") or ""
    if not isinstance(model, str):
        model = ""

    raw_in = getattr(usage_metadata, "prompt_token_count", 0)
    raw_out = getattr(usage_metadata, "candidates_token_count", 0)
    if not isinstance(raw_in, int):
        raw_in = 0
    if not isinstance(raw_out, int):
        raw_out = 0
    return (model, raw_in or 0, raw_out or 0)
