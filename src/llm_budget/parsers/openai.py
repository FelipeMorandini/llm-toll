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

    model: str = response.model
    input_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens: int = getattr(usage, "completion_tokens", 0) or 0
    return (model, input_tokens, output_tokens)
