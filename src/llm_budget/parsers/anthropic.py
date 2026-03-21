"""Auto-parser for Anthropic SDK response objects."""

from __future__ import annotations


def parse_anthropic_response(response: object) -> tuple[str, int, int] | None:
    """Extract model name and token usage from an Anthropic response.

    Uses duck-typing to detect Anthropic Message objects without
    importing the anthropic package. Checks for the ``stop_reason``
    attribute as the primary discriminator from other SDK responses.

    Returns ``(model, input_tokens, output_tokens)`` or ``None``.
    """
    if not (
        hasattr(response, "stop_reason")
        and hasattr(response, "usage")
        and hasattr(response, "model")
        and hasattr(response, "content")
    ):
        return None

    usage = response.usage
    if usage is None:
        return None

    if not (hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens")):
        return None

    model: str = response.model
    input_tokens: int = getattr(usage, "input_tokens", 0) or 0
    output_tokens: int = getattr(usage, "output_tokens", 0) or 0
    return (model, input_tokens, output_tokens)
