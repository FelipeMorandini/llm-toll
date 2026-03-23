"""LangChain callback handler for automatic cost tracking."""

from __future__ import annotations

import contextlib
from typing import Any

from llm_toll.exceptions import BudgetExceededError
from llm_toll.pricing import default_registry
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore


class LangChainCallback:
    """LangChain callback handler that tracks costs via llm_toll.

    Register as a LangChain callback for automatic cost tracking::

        from langchain_openai import ChatOpenAI
        from llm_toll.integrations.langchain import LangChainCallback

        handler = LangChainCallback(project="my-chain", max_budget=10.0)
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

    Budget is checked *before* each LLM call (in ``on_llm_start``),
    and usage is logged *after* each call (in ``on_llm_end``).

    Parameters
    ----------
    project:
        Project name for grouping usage in the store.
    max_budget:
        Optional hard budget cap in USD.  Raises
        :class:`BudgetExceededError` before the LLM call if exceeded.
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
        store: UsageStore | None = None,
        reporter: CostReporter | None = None,
    ) -> None:
        self._project = project
        self._max_budget = max_budget
        self._store = store
        self._reporter = reporter

    def _get_store(self) -> UsageStore:
        if self._store is not None:
            return self._store
        from llm_toll.decorator import _get_store

        return _get_store()

    def _get_reporter(self) -> CostReporter:
        if self._reporter is not None:
            return self._reporter
        from llm_toll.decorator import _get_reporter

        return _get_reporter()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Called before the LLM executes. Checks budget if configured."""
        if self._max_budget is None:
            return
        current_cost = self._get_store().get_total_cost(self._project)
        if current_cost >= self._max_budget:
            raise BudgetExceededError(
                project=self._project,
                current_cost=current_cost,
                max_budget=self._max_budget,
            )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called after a successful LLM completion. Logs cost."""
        llm_output = getattr(response, "llm_output", None)
        if not isinstance(llm_output, dict):
            return

        token_usage = llm_output.get("token_usage")
        if not isinstance(token_usage, dict):
            token_usage = {}
        model = llm_output.get("model_name")
        if not isinstance(model, str) or not model:
            model = "unknown"

        input_tokens = token_usage.get("prompt_tokens", 0) or 0
        output_tokens = token_usage.get("completion_tokens", 0) or 0

        if not isinstance(input_tokens, int):
            input_tokens = 0
        if not isinstance(output_tokens, int):
            output_tokens = 0

        if input_tokens == 0 and output_tokens == 0:
            return

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

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called after a failed LLM call. No-op."""
