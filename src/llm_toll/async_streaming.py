"""Async streaming response support for async iterators/generators."""

from __future__ import annotations

import asyncio
import contextlib
import types
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from llm_toll.pricing import PricingRegistry
from llm_toll.rate_limiter import RateLimiter
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore
from llm_toll.streaming import StreamAccumulator, _finalize_stream


def _is_async_stream(response: object) -> bool:
    """Return ``True`` if *response* looks like an async stream.

    Matches async generators and SDK async stream objects (which
    expose ``__anext__`` and ``aclose``).
    """
    if isinstance(response, (str, bytes, dict, list)):
        return False
    if isinstance(response, types.AsyncGeneratorType):
        return True
    # SDK async stream objects expose __anext__ + aclose()
    if not (hasattr(response, "__anext__") and hasattr(response, "aclose")):
        return False
    return hasattr(response, "__aiter__")


async def wrap_async_stream(
    stream: AsyncIterator[Any],
    *,
    project: str,
    model_override: str | None,
    max_budget: float | None,
    store: UsageStore,
    registry: PricingRegistry,
    reporter: CostReporter,
    rate_limiter: RateLimiter | None = None,
) -> AsyncGenerator[Any, None]:
    """Wrap an async streaming response to track cost after exhaustion.

    Yields every chunk through transparently.  After the stream is
    fully consumed (or the caller breaks out), extracts accumulated
    usage information, calculates cost, and logs it to the store
    via :func:`asyncio.to_thread` to avoid blocking the event loop.
    """
    accumulator = StreamAccumulator()
    try:
        async for chunk in stream:
            accumulator.process_chunk(chunk)
            yield chunk
    finally:
        try:
            await asyncio.to_thread(
                _finalize_stream,
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
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()
