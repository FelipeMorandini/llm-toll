"""Auto-parser for OpenAI SDK response objects."""

from __future__ import annotations


def parse_openai_response(response: object) -> tuple[str, int, int] | None:
    """Extract model name and token usage from an OpenAI response.

    Uses duck-typing to detect OpenAI ChatCompletion/Completion objects
    without importing the openai package. Checks for the ``choices``
    attribute as the primary discriminator from other SDK responses.

    Returns ``(model, input_tokens, output_tokens)`` or ``None``.
    """
    if not (
        hasattr(response, "choices") and hasattr(response, "usage") and hasattr(response, "model")
    ):
        return None

    usage = response.usage
    if usage is None:
        return None

    if not (hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens")):
        return None

    model = response.model
    if not isinstance(model, str) or not model:
        return None

    raw_in = getattr(usage, "prompt_tokens", 0)
    raw_out = getattr(usage, "completion_tokens", 0)
    if not isinstance(raw_in, int):
        raw_in = 0
    if not isinstance(raw_out, int):
        raw_out = 0
    return (model, raw_in or 0, raw_out or 0)
