"""Main entry point: the @track_costs decorator."""

from __future__ import annotations

import contextlib
import functools
import threading
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

from llm_toll.exceptions import BudgetExceededError
from llm_toll.parsers import auto_detect_usage
from llm_toll.pricing import default_registry
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore
from llm_toll.streaming import _is_sync_stream, wrap_sync_stream

F = TypeVar("F", bound=Callable[..., Any])

_default_store: UsageStore | None = None
_store_lock = threading.Lock()

_default_reporter: CostReporter | None = None
_reporter_lock = threading.Lock()


def _get_store() -> UsageStore:
    """Return the shared module-level UsageStore (lazily created)."""
    global _default_store
    if _default_store is not None:
        return _default_store
    with _store_lock:
        if _default_store is not None:
            return _default_store
        _default_store = UsageStore()
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


def set_store(store: UsageStore | None) -> None:
    """Inject a custom UsageStore for the decorator to use.

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

    Can be used with or without arguments::

        @track_costs
        def my_func(): ...

        @track_costs(project="my-project", max_budget=10.0)
        def my_func(): ...

    Workflow on each call:
    1. Check budget (if *max_budget* is set).
    2. Execute the wrapped function.
    3. If the response is a sync generator/stream, wrap it so cost is
       tracked after the stream is consumed (the wrapper yields chunks
       through transparently).
    4. Otherwise, extract token usage from the response object.
    5. Calculate cost via the pricing registry.
    6. Log usage to the local SQLite store.
    7. Return the original response (or wrapped stream) unchanged.
    """

    def decorator(func: F) -> F:
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

            # 2. Execute the wrapped function
            response = func(*args, **kwargs)

            # 2b. If the response is a sync stream, wrap it for deferred tracking
            if _is_sync_stream(response):
                return wrap_sync_stream(
                    response,
                    project=project,
                    model_override=model,
                    max_budget=max_budget,
                    store=store,
                    registry=default_registry,
                    reporter=_get_reporter(),
                )

            # 3. Extract usage from response
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
                    return response

            if usage_info is None:
                return response

            detected_model, input_tokens, output_tokens = usage_info
            effective_model = model if model is not None else detected_model

            # 4. Calculate cost
            cost = default_registry.get_cost(effective_model, input_tokens, output_tokens)

            # 5. Log usage (atomic budget check when max_budget is set)
            if max_budget is not None:
                store.log_usage_if_within_budget(
                    project, effective_model, input_tokens, output_tokens, cost, max_budget
                )
            else:
                store.log_usage(project, effective_model, input_tokens, output_tokens, cost)

            # 5b. Report cost
            with contextlib.suppress(Exception):
                _get_reporter().report_call(effective_model, input_tokens, output_tokens, cost)

            # 6. Return original response
            return response

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator
