"""LiteLLM callback integration for automatic cost tracking."""

from __future__ import annotations

import contextlib
from typing import Any

from llm_toll.parsers import auto_detect_usage
from llm_toll.pricing import default_registry
from llm_toll.reporter import CostReporter
from llm_toll.store import BaseStore


def _normalize_model(model: str) -> str:
    """Strip LiteLLM provider prefix when the suffix is a known model.

    LiteLLM uses model strings like ``"openai/gpt-4o"`` or
    ``"anthropic/claude-sonnet-4-20250514"``.  The prefix is stripped
    only when the suffix (e.g. ``"gpt-4o"``) is already registered in
    the pricing registry.  This preserves namespace-prefixed models
    like ``"ollama/llama3"`` that rely on the ``"ollama/"`` pricing
    prefix.
    """
    if "/" not in model:
        return model
    _, _, suffix = model.partition("/")
    if suffix and default_registry.has_model(suffix):
        return suffix
    return model


class LiteLLMCallback:
    """LiteLLM custom callback that tracks costs via llm_toll.

    Register as a LiteLLM callback for zero-decorator cost tracking::

        import litellm
        from llm_toll.integrations.litellm import LiteLLMCallback

        litellm.callbacks = [LiteLLMCallback(project="my-app", max_budget=10.0)]

    All LiteLLM completions will be automatically logged to the local
    SQLite store with cost calculation and optional budget enforcement.

    Parameters
    ----------
    project:
        Project name for grouping usage in the store.
    max_budget:
        Optional hard budget cap in USD.  Raises
        :class:`BudgetExceededError` on the *next* callback if exceeded.
    store:
        Optional :class:`UsageStore` instance.  Defaults to the shared
        module-level store (same one used by ``@track_costs``).
    reporter:
        Optional :class:`CostReporter` instance.  Defaults to the
        shared module-level reporter.
    """

    def __init__(
        self,
        *,
        project: str = "default",
        max_budget: float | None = None,
        store: BaseStore | None = None,
        reporter: CostReporter | None = None,
    ) -> None:
        self._project = project
        self._max_budget = max_budget
        self._store = store
        self._reporter = reporter

    def _get_store(self) -> BaseStore:
        if self._store is not None:
            return self._store
        # Lazy import to avoid circular dependency at module level
        from llm_toll.decorator import _get_store

        return _get_store()

    def _get_reporter(self) -> CostReporter:
        if self._reporter is not None:
            return self._reporter
        from llm_toll.decorator import _get_reporter

        return _get_reporter()

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called by LiteLLM after a successful completion.

        Extracts token usage from *response_obj*, calculates cost, and
        logs it to the store.
        """
        usage_info = auto_detect_usage(response_obj)
        if usage_info is None:
            return

        detected_model, input_tokens, output_tokens = usage_info
        model = _normalize_model(detected_model)

        cost = default_registry.get_cost(model, input_tokens, output_tokens)

        store = self._get_store()
        if self._max_budget is not None:
            store.log_usage_if_within_budget(
                self._project, model, input_tokens, output_tokens, cost, self._max_budget
            )
        else:
            store.log_usage(self._project, model, input_tokens, output_tokens, cost)

        with contextlib.suppress(Exception):
            self._get_reporter().report_call(model, input_tokens, output_tokens, cost)

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called by LiteLLM after a failed completion. No-op."""
