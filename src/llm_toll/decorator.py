"""Main entry point: the @track_costs decorator."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import os
import threading
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

from llm_toll.async_streaming import _is_async_stream, wrap_async_stream
from llm_toll.exceptions import BudgetExceededError
from llm_toll.parsers import auto_detect_usage
from llm_toll.pricing import default_registry
from llm_toll.rate_limiter import RateLimiter
from llm_toll.reporter import CostReporter
from llm_toll.store import BaseStore, create_store
from llm_toll.streaming import _is_sync_stream, wrap_sync_stream

F = TypeVar("F", bound=Callable[..., Any])

_default_store: BaseStore | None = None
_store_lock = threading.Lock()

_default_reporter: CostReporter | None = None
_reporter_lock = threading.Lock()


def _get_store() -> BaseStore:
    """Return the shared module-level store (lazily created)."""
    global _default_store
    if _default_store is not None:
        return _default_store
    with _store_lock:
        if _default_store is not None:
            return _default_store
        url = os.environ.get("LLM_TOLL_STORE_URL")
        try:
            _default_store = create_store(url=url)
        except Exception as exc:
            warnings.warn(
                f"Failed to create store from LLM_TOLL_STORE_URL={url!r}: {exc}. "
                "Falling back to local SQLite.",
                stacklevel=2,
            )
            _default_store = create_store(url=None)
        return _default_store


def _get_reporter() -> CostReporter:
    """Return the shared module-level CostReporter (lazily created)."""
    global _default_reporter
    if _default_reporter is not None:
        return _default_reporter
    with _reporter_lock:
        if _default_reporter is not None:
            return _default_reporter
        _default_reporter = CostReporter()
        return _default_reporter


def set_reporter(reporter: CostReporter | None) -> None:
    """Inject a custom CostReporter for the decorator to use.

    Pass ``None`` to reset to the default (lazily created) reporter.
    """
    global _default_reporter
    with _reporter_lock:
        _default_reporter = reporter


def set_store(store: BaseStore | None) -> None:
    """Inject a custom store for the decorator to use.

    Pass ``None`` to reset to the default (lazily created) store.
    Useful for testing or directing storage to a custom database path.
    """
    global _default_store
    with _store_lock:
        _default_store = store


@overload
def track_costs(fn: F) -> F: ...


@overload
def track_costs(
    *,
    project: str = "default",
    model: str | None = None,
    max_budget: float | None = None,
    reset: str | None = None,
    rate_limit: int | None = None,
    tpm_limit: int | None = None,
    extract_usage: Callable[..., tuple[str, int, int]] | None = None,
) -> Callable[[F], F]: ...


def track_costs(
    fn: F | None = None,
    *,
    project: str = "default",
    model: str | None = None,
    max_budget: float | None = None,
    reset: str | None = None,
    rate_limit: int | None = None,
    tpm_limit: int | None = None,
    extract_usage: Callable[..., tuple[str, int, int]] | None = None,
) -> F | Callable[[F], F]:
    """Decorator to track costs, enforce budgets, and rate-limit LLM API calls.

    Can be used with or without arguments on both sync and async functions::

        @track_costs
        def my_func(): ...

        @track_costs(project="my-project", max_budget=10.0)
        async def my_func(): ...

    Async functions are auto-detected at decoration time.  SQLite
    operations run in a thread pool via ``asyncio.to_thread`` so the
    event loop is never blocked.  Async generators (streaming) are
    wrapped transparently, just like sync generators.
    """

    def decorator(func: F) -> F:
        limiter = (
            RateLimiter(rpm=rate_limit, tpm=tpm_limit)
            if rate_limit is not None or tpm_limit is not None
            else None
        )

        # --- Async generator path ---
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                store = _get_store()

                if max_budget is not None:
                    current_cost = await asyncio.to_thread(store.get_total_cost, project)
                    if current_cost >= max_budget:
                        raise BudgetExceededError(
                            project=project,
                            current_cost=current_cost,
                            max_budget=max_budget,
                        )

                if limiter is not None:
                    limiter.check()

                async_stream = func(*args, **kwargs)
                wrapped = wrap_async_stream(
                    async_stream,
                    project=project,
                    model_override=model,
                    max_budget=max_budget,
                    store=store,
                    registry=default_registry,
                    reporter=_get_reporter(),
                    rate_limiter=limiter,
                )
                try:
                    async for chunk in wrapped:
                        yield chunk
                finally:
                    with contextlib.suppress(Exception):
                        await wrapped.aclose()

            return async_gen_wrapper  # type: ignore[return-value]

        # --- Async coroutine path ---
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                store = _get_store()

                if max_budget is not None:
                    current_cost = await asyncio.to_thread(store.get_total_cost, project)
                    if current_cost >= max_budget:
                        raise BudgetExceededError(
                            project=project,
                            current_cost=current_cost,
                            max_budget=max_budget,
                        )

                if limiter is not None:
                    limiter.check()

                response = await func(*args, **kwargs)

                if _is_async_stream(response):
                    return wrap_async_stream(
                        response,
                        project=project,
                        model_override=model,
                        max_budget=max_budget,
                        store=store,
                        registry=default_registry,
                        reporter=_get_reporter(),
                        rate_limiter=limiter,
                    )

                usage_info: tuple[str, int, int] | None = None

                if response is not None:
                    usage_info = auto_detect_usage(response)

                if usage_info is None and extract_usage is not None:
                    try:
                        usage_info = extract_usage(response)
                    except Exception:
                        warnings.warn(
                            "extract_usage callback raised an exception; "
                            "skipping cost tracking for this call.",
                            stacklevel=2,
                        )
                        if limiter is not None:
                            limiter.record(tokens=0)
                        return response

                if usage_info is None:
                    if limiter is not None:
                        limiter.record(tokens=0)
                    return response

                detected_model, input_tokens, output_tokens = usage_info
                effective_model = model if model is not None else detected_model

                if limiter is not None:
                    limiter.record(tokens=input_tokens + output_tokens)

                cost = default_registry.get_cost(effective_model, input_tokens, output_tokens)

                if max_budget is not None:
                    await asyncio.to_thread(
                        store.log_usage_if_within_budget,
                        project,
                        effective_model,
                        input_tokens,
                        output_tokens,
                        cost,
                        max_budget,
                    )
                else:
                    await asyncio.to_thread(
                        store.log_usage,
                        project,
                        effective_model,
                        input_tokens,
                        output_tokens,
                        cost,
                    )

                with contextlib.suppress(Exception):
                    _get_reporter().report_call(effective_model, input_tokens, output_tokens, cost)

                return response

            return async_wrapper  # type: ignore[return-value]

        # --- Sync path (unchanged) ---
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            store = _get_store()

            # 1. Pre-call budget check
            if max_budget is not None:
                current_cost = store.get_total_cost(project)
                if current_cost >= max_budget:
                    raise BudgetExceededError(
                        project=project,
                        current_cost=current_cost,
                        max_budget=max_budget,
                    )

            # 2. Pre-call rate limit check
            if limiter is not None:
                limiter.check()

            # 3. Execute the wrapped function
            response = func(*args, **kwargs)

            # 4. If the response is a sync stream, wrap it for deferred tracking
            if _is_sync_stream(response):
                return wrap_sync_stream(
                    response,
                    project=project,
                    model_override=model,
                    max_budget=max_budget,
                    store=store,
                    registry=default_registry,
                    reporter=_get_reporter(),
                    rate_limiter=limiter,
                )

            # 5. Extract usage from response
            usage_info: tuple[str, int, int] | None = None

            if response is not None:
                usage_info = auto_detect_usage(response)

            if usage_info is None and extract_usage is not None:
                try:
                    usage_info = extract_usage(response)
                except Exception:
                    warnings.warn(
                        "extract_usage callback raised an exception; "
                        "skipping cost tracking for this call.",
                        stacklevel=2,
                    )
                    # Record the request for RPM tracking even though
                    # we could not extract usage (the API call happened).
                    if limiter is not None:
                        limiter.record(tokens=0)
                    return response

            if usage_info is None:
                # Record the request for RPM tracking even when no
                # usage info is available (the API call still happened).
                if limiter is not None:
                    limiter.record(tokens=0)
                return response

            detected_model, input_tokens, output_tokens = usage_info
            effective_model = model if model is not None else detected_model

            # 6. Record tokens for rate limiting
            if limiter is not None:
                limiter.record(tokens=input_tokens + output_tokens)

            # 7. Calculate cost and log usage
            cost = default_registry.get_cost(effective_model, input_tokens, output_tokens)

            if max_budget is not None:
                store.log_usage_if_within_budget(
                    project, effective_model, input_tokens, output_tokens, cost, max_budget
                )
            else:
                store.log_usage(project, effective_model, input_tokens, output_tokens, cost)

            # 8. Report cost
            with contextlib.suppress(Exception):
                _get_reporter().report_call(effective_model, input_tokens, output_tokens, cost)

            return response

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator
