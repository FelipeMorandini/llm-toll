"""Streaming response support for sync iterators/generators."""

from __future__ import annotations

import contextlib
import types
import warnings
from collections.abc import Generator, Iterator
from typing import Any

from llm_toll.exceptions import BudgetExceededError
from llm_toll.pricing import PricingRegistry
from llm_toll.rate_limiter import RateLimiter
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using a character-based heuristic.

    English text averages roughly 4 characters per token.  This is a
    coarse approximation used only when the API does not provide usage
    data (e.g. OpenAI streaming without ``stream_options``).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _is_sync_stream(response: object) -> bool:
    """Return ``True`` if *response* looks like a sync stream.

    Matches Python generators and SDK stream objects (which expose
    both ``__next__`` and ``close``).  Plain iterators such as
    ``map``, ``filter``, and ``zip`` are excluded because they lack
    a ``close`` method.  The object must also be iterable (support
    ``__iter__``) so that ``for chunk in stream`` works.
    """
    if isinstance(response, (str, bytes, dict, list)):
        return False
    if isinstance(response, types.GeneratorType):
        return True
    # SDK stream objects (e.g. openai.Stream) expose __next__ + close()
    if not (hasattr(response, "__next__") and hasattr(response, "close")):
        return False
    # Ensure the object is actually iterable
    return hasattr(response, "__iter__")


class StreamAccumulator:
    """Accumulates token usage from streaming chunks.

    Supports OpenAI ``ChatCompletionChunk`` objects and Anthropic
    streaming events via duck-typing.
    """

    def __init__(self) -> None:
        self._model: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._char_count: int = 0
        self._has_api_usage: bool = False

    def process_chunk(self, chunk: object) -> None:
        """Dispatch a single chunk to the appropriate handler."""
        if self._try_openai_chunk(chunk):
            return
        if self._try_anthropic_event(chunk):
            return
        if self._try_gemini_chunk(chunk):
            return

    def _try_openai_chunk(self, chunk: object) -> bool:
        """Process an OpenAI ``ChatCompletionChunk``-like object.

        Returns ``True`` if the chunk was recognised as OpenAI.
        """
        if not (hasattr(chunk, "choices") and hasattr(chunk, "model")):
            return False

        choices = chunk.choices
        if choices and hasattr(choices[0], "delta"):
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                self._char_count += len(content)

        model = getattr(chunk, "model", None)
        if isinstance(model, str) and model:
            self._model = model

        # Final chunk may carry usage when stream_options={"include_usage": True}
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            raw_in = getattr(usage, "prompt_tokens", 0)
            raw_out = getattr(usage, "completion_tokens", 0)
            if isinstance(raw_in, int) and isinstance(raw_out, int):
                self._input_tokens = raw_in
                self._output_tokens = raw_out
                self._has_api_usage = True

        return True

    def _try_anthropic_event(self, chunk: object) -> bool:
        """Process an Anthropic streaming event.

        Returns ``True`` if the chunk was recognised as an Anthropic event.
        """
        event_type = getattr(chunk, "type", None)
        if not isinstance(event_type, str):
            return False

        if event_type == "message_start":
            message = getattr(chunk, "message", None)
            if message is not None:
                model = getattr(message, "model", None)
                if isinstance(model, str) and model:
                    self._model = model
                usage = getattr(message, "usage", None)
                if usage is not None:
                    raw_in = getattr(usage, "input_tokens", None)
                    if isinstance(raw_in, int):
                        self._input_tokens = raw_in
                        self._has_api_usage = True

        elif event_type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if delta is not None:
                text = getattr(delta, "text", None)
                if text:
                    self._char_count += len(text)

        elif event_type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                raw_out = getattr(usage, "output_tokens", None)
                if isinstance(raw_out, int):
                    self._output_tokens = raw_out
                    self._has_api_usage = True

        return True

    def _try_gemini_chunk(self, chunk: object) -> bool:
        """Process a Gemini ``GenerateContentResponse`` streaming chunk.

        Returns ``True`` if the chunk was recognised as Gemini.
        """
        if not (hasattr(chunk, "candidates") and hasattr(chunk, "usage_metadata")):
            return False
        if hasattr(chunk, "choices"):
            return False

        # Extract text from candidates[0].content.parts[0].text
        candidates = getattr(chunk, "candidates", None)
        if candidates:
            try:
                content = getattr(candidates[0], "content", None)
                if content is not None:
                    parts = getattr(content, "parts", None)
                    if parts:
                        text = getattr(parts[0], "text", None)
                        if text:
                            self._char_count += len(text)
            except (IndexError, TypeError):
                pass

        # Extract model version
        model_version = getattr(chunk, "model_version", None)
        if isinstance(model_version, str) and model_version:
            self._model = model_version
        elif self._model is None:
            self._model = ""  # Gemini may not carry model in chunks

        # Extract usage from usage_metadata
        usage_metadata = getattr(chunk, "usage_metadata", None)
        if usage_metadata is not None:
            raw_in = getattr(usage_metadata, "prompt_token_count", None)
            raw_out = getattr(usage_metadata, "candidates_token_count", None)
            if isinstance(raw_in, int) and isinstance(raw_out, int):
                self._input_tokens = raw_in
                self._output_tokens = raw_out
                self._has_api_usage = True

        return True

    def get_usage(self) -> tuple[str, int, int] | None:
        """Return ``(model, input_tokens, output_tokens)`` or ``None``.

        Uses API-provided token counts when available.  Falls back to
        :func:`estimate_tokens` for output tokens derived from
        accumulated text, and ``0`` for input tokens (unknown from the
        stream alone).
        """
        model = self._model
        if model is None:
            return None

        if self._has_api_usage:
            return (model, self._input_tokens or 0, self._output_tokens or 0)

        # Fallback: estimate from accumulated character count
        if self._char_count == 0:
            return None

        estimated_out = max(1, self._char_count // 4)
        warnings.warn(
            "No API-provided token usage in stream; using character-based "
            f"estimate ({estimated_out} output tokens). Pass "
            "stream_options={'include_usage': True} for accurate counts.",
            stacklevel=3,
        )
        return (model, 0, estimated_out)


def _finalize_stream(
    accumulator: StreamAccumulator,
    *,
    project: str,
    model_override: str | None,
    max_budget: float | None,
    store: UsageStore,
    registry: PricingRegistry,
    reporter: CostReporter,
    rate_limiter: RateLimiter | None = None,
) -> None:
    """Extract usage from accumulator and log cost.

    Called after a stream is exhausted (or the consumer breaks out).
    Because the stream has already been consumed by the caller, the
    cost is real regardless of budget state.  If the budget is exceeded,
    the usage is still logged unconditionally (via :meth:`log_usage`) so
    the running total stays accurate for future pre-call checks, and a
    warning is emitted.
    """
    usage_info = accumulator.get_usage()
    if usage_info is None:
        # Record the request for RPM tracking even though we
        # could not extract usage (the stream was consumed).
        if rate_limiter is not None:
            rate_limiter.record(tokens=0)
        return

    detected_model, input_tokens, output_tokens = usage_info
    effective_model = model_override if model_override is not None else detected_model

    if rate_limiter is not None:
        rate_limiter.record(tokens=input_tokens + output_tokens)

    cost = registry.get_cost(effective_model, input_tokens, output_tokens)

    try:
        if max_budget is not None:
            store.log_usage_if_within_budget(
                project, effective_model, input_tokens, output_tokens, cost, max_budget
            )
        else:
            store.log_usage(project, effective_model, input_tokens, output_tokens, cost)
    except BudgetExceededError:
        # The stream was already consumed so the cost is real.  Log it
        # unconditionally to keep the budget total accurate, and warn.
        store.log_usage(project, effective_model, input_tokens, output_tokens, cost)
        warnings.warn(
            f"Streaming call for project {project!r} exceeded budget "
            f"(${cost:.4f} pushed total over ${max_budget:.4f}). "
            "The cost has been logged but the stream was already consumed.",
            stacklevel=2,
        )

    with contextlib.suppress(Exception):
        reporter.report_call(effective_model, input_tokens, output_tokens, cost)


def wrap_sync_stream(
    stream: Iterator[Any],
    *,
    project: str,
    model_override: str | None,
    max_budget: float | None,
    store: UsageStore,
    registry: PricingRegistry,
    reporter: CostReporter,
    rate_limiter: RateLimiter | None = None,
) -> Generator[Any, None, None]:
    """Wrap a sync streaming response to track cost after exhaustion.

    Yields every chunk through transparently.  After the stream is
    fully consumed (or the caller breaks out), extracts accumulated
    usage information, calculates cost, and logs it to the store.
    """
    accumulator = StreamAccumulator()
    try:
        for chunk in stream:
            accumulator.process_chunk(chunk)
            yield chunk
    finally:
        try:
            _finalize_stream(
                accumulator,
                project=project,
                model_override=model_override,
                max_budget=max_budget,
                store=store,
                registry=registry,
                reporter=reporter,
                rate_limiter=rate_limiter,
            )
        finally:
            # Close the underlying SDK stream if it supports it
            close = getattr(stream, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
