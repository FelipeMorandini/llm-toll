"""Unit tests for the LiteLLM callback integration."""

from __future__ import annotations

import pytest

from llm_toll.exceptions import BudgetExceededError
from llm_toll.integrations.litellm import LiteLLMCallback
from llm_toll.store import UsageStore

# ---------------------------------------------------------------------------
# Mock helpers (replicating the OpenAI-shaped response objects LiteLLM uses)
# ---------------------------------------------------------------------------


class _MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockResponse:
    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.model = model
        self.usage = _MockUsage(prompt_tokens, completion_tokens)
        self.choices: list = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiteLLMCallback:
    """Tests for :class:`LiteLLMCallback`."""

    @staticmethod
    def _make_callback(
        tmp_db_path: str,
        *,
        project: str = "test-project",
        max_budget: float | None = None,
    ) -> tuple[LiteLLMCallback, UsageStore]:
        store = UsageStore(db_path=tmp_db_path)
        cb = LiteLLMCallback(
            project=project,
            max_budget=max_budget,
            store=store,
        )
        return cb, store

    def test_log_success_logs_usage(self, tmp_db_path: str) -> None:
        """log_success_event should persist cost to the store."""
        cb, store = self._make_callback(tmp_db_path)
        resp = _MockResponse("gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.log_success_event({}, resp, None, None)

        total = store.get_total_cost("test-project")
        assert total > 0

    def test_log_success_with_budget(self, tmp_db_path: str) -> None:
        """When max_budget is set, usage goes through log_usage_if_within_budget."""
        cb, store = self._make_callback(tmp_db_path, max_budget=10.0)
        resp = _MockResponse("gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.log_success_event({}, resp, None, None)

        total = store.get_total_cost("test-project")
        assert total > 0

    def test_log_success_budget_exceeded_raises(self, tmp_db_path: str) -> None:
        """Should raise BudgetExceededError when the budget is already exhausted."""
        cb, store = self._make_callback(tmp_db_path, max_budget=0.0001)
        # Pre-seed the store so the budget is already exceeded
        store.log_usage("test-project", "gpt-4o", 100_000, 100_000, 1.0)

        resp = _MockResponse("gpt-4o", prompt_tokens=500, completion_tokens=200)

        with pytest.raises(BudgetExceededError):
            cb.log_success_event({}, resp, None, None)

    def test_log_success_unrecognized_response(self, tmp_db_path: str) -> None:
        """A plain dict should not crash; auto_detect_usage returns None -> no-op."""
        cb, _store = self._make_callback(tmp_db_path)

        # Should not raise
        cb.log_success_event({}, {"plain": "dict"}, None, None)

    def test_log_failure_is_noop(self, tmp_db_path: str) -> None:
        """log_failure_event should not crash or record anything."""
        cb, store = self._make_callback(tmp_db_path)

        cb.log_failure_event({}, None, None, None)

        total = store.get_total_cost("test-project")
        assert total == 0.0

    def test_model_normalization(self, tmp_db_path: str) -> None:
        """A provider-prefixed model like 'openai/gpt-4o' should be logged as 'gpt-4o'."""
        cb, store = self._make_callback(tmp_db_path)
        resp = _MockResponse("openai/gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.log_success_event({}, resp, None, None)

        # The pricing registry does not have "openai/gpt-4o", so _normalize_model
        # should strip the prefix.  Verify cost was calculated (non-zero means
        # the normalized model was found in the registry).
        total = store.get_total_cost("test-project")
        assert total > 0

    def test_custom_project(self, tmp_db_path: str) -> None:
        """The project name should flow through to the store."""
        cb, store = self._make_callback(tmp_db_path, project="custom-proj")
        resp = _MockResponse("gpt-4o", prompt_tokens=100, completion_tokens=50)

        cb.log_success_event({}, resp, None, None)

        assert store.get_total_cost("custom-proj") > 0
        assert store.get_total_cost("test-project") == 0.0

    def test_end_to_end(self, tmp_db_path: str) -> None:
        """Full flow: multiple calls accumulate cost in a real UsageStore."""
        cb, store = self._make_callback(tmp_db_path, project="e2e")

        for _ in range(3):
            resp = _MockResponse("gpt-4o", prompt_tokens=1000, completion_tokens=500)
            cb.log_success_event({}, resp, None, None)

        total = store.get_total_cost("e2e")
        # 3 identical calls should yield 3x a single call's cost
        single_resp = _MockResponse("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        cb2, store2 = self._make_callback(tmp_db_path + "_ref", project="e2e")
        cb2.log_success_event({}, single_resp, None, None)
        single_cost = store2.get_total_cost("e2e")

        assert single_cost > 0
        assert total == pytest.approx(3 * single_cost)
