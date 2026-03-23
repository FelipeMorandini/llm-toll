"""Unit tests for the LangChain callback integration."""

from __future__ import annotations

import pytest

from llm_toll.exceptions import BudgetExceededError
from llm_toll.integrations.langchain import LangChainCallback
from llm_toll.store import UsageStore

# ---------------------------------------------------------------------------
# Mock helpers (replicating the LangChain LLMResult response object)
# ---------------------------------------------------------------------------


class _MockLLMResult:
    def __init__(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": model_name,
        }
        self.generations: list = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLangChainCallback:
    """Tests for :class:`LangChainCallback`."""

    @staticmethod
    def _make_callback(
        tmp_db_path: str,
        *,
        project: str = "test-project",
        max_budget: float | None = None,
    ) -> tuple[LangChainCallback, UsageStore]:
        store = UsageStore(db_path=tmp_db_path)
        cb = LangChainCallback(
            project=project,
            max_budget=max_budget,
            store=store,
        )
        return cb, store

    def test_on_llm_end_logs_usage(self, tmp_db_path: str) -> None:
        """on_llm_end should persist cost to the store."""
        cb, store = self._make_callback(tmp_db_path)
        resp = _MockLLMResult("gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.on_llm_end(resp)

        total = store.get_total_cost("test-project")
        assert total > 0

    def test_on_llm_end_with_budget(self, tmp_db_path: str) -> None:
        """When max_budget is set, usage goes through log_usage_if_within_budget."""
        cb, store = self._make_callback(tmp_db_path, max_budget=10.0)
        resp = _MockLLMResult("gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.on_llm_end(resp)

        total = store.get_total_cost("test-project")
        assert total > 0

    def test_on_llm_end_budget_exceeded_raises(self, tmp_db_path: str) -> None:
        """Should raise BudgetExceededError when the budget is already exhausted."""
        cb, store = self._make_callback(tmp_db_path, max_budget=0.0001)
        # Pre-seed the store so the budget is already exceeded
        store.log_usage("test-project", "gpt-4o", 100_000, 100_000, 1.0)

        resp = _MockLLMResult("gpt-4o", prompt_tokens=500, completion_tokens=200)

        with pytest.raises(BudgetExceededError):
            cb.on_llm_end(resp)

    def test_on_llm_start_budget_check(self, tmp_db_path: str) -> None:
        """on_llm_start should raise BudgetExceededError when budget is exceeded."""
        cb, store = self._make_callback(tmp_db_path, max_budget=0.0001)
        store.log_usage("test-project", "gpt-4o", 100_000, 100_000, 1.0)

        with pytest.raises(BudgetExceededError):
            cb.on_llm_start({}, ["Hello"])

    def test_on_llm_start_within_budget(self, tmp_db_path: str) -> None:
        """on_llm_start should not raise when usage is within budget."""
        cb, _store = self._make_callback(tmp_db_path, max_budget=100.0)

        # Should not raise
        cb.on_llm_start({}, ["Hello"])

    def test_on_llm_start_no_budget(self, tmp_db_path: str) -> None:
        """on_llm_start should be a no-op when max_budget is None."""
        cb, _store = self._make_callback(tmp_db_path)

        # Should not raise
        cb.on_llm_start({}, ["Hello"])

    def test_on_llm_end_missing_llm_output(self, tmp_db_path: str) -> None:
        """When response.llm_output is None, on_llm_end should be a safe no-op."""
        cb, store = self._make_callback(tmp_db_path)

        class _NullOutput:
            llm_output = None

            def __init__(self) -> None:
                self.generations: list = []

        cb.on_llm_end(_NullOutput())

        total = store.get_total_cost("test-project")
        assert total == 0.0

    def test_on_llm_error_is_noop(self, tmp_db_path: str) -> None:
        """on_llm_error should not crash or record anything."""
        cb, store = self._make_callback(tmp_db_path)

        cb.on_llm_error(RuntimeError("boom"))

        total = store.get_total_cost("test-project")
        assert total == 0.0

    def test_custom_project(self, tmp_db_path: str) -> None:
        """The project name should flow through to the store."""
        cb, store = self._make_callback(tmp_db_path, project="custom-proj")
        resp = _MockLLMResult("gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.on_llm_end(resp)

        assert store.get_total_cost("custom-proj") > 0
        assert store.get_total_cost("test-project") == 0.0

    def test_end_to_end(self, tmp_db_path: str) -> None:
        """Full flow: multiple calls accumulate cost in a real UsageStore."""
        cb, store = self._make_callback(tmp_db_path, project="e2e")

        for _ in range(3):
            resp = _MockLLMResult("gpt-4o", prompt_tokens=1000, completion_tokens=500)
            cb.on_llm_end(resp)

        total = store.get_total_cost("e2e")
        # 3 identical calls should yield 3x a single call's cost
        cb2, store2 = self._make_callback(tmp_db_path + "_ref", project="e2e")
        single_resp = _MockLLMResult("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        cb2.on_llm_end(single_resp)
        single_cost = store2.get_total_cost("e2e")

        assert single_cost > 0
        assert total == pytest.approx(3 * single_cost)
