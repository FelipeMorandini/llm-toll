"""Unit tests for the track_costs decorator."""

from __future__ import annotations

import pytest

from llm_budget.decorator import set_store, track_costs
from llm_budget.exceptions import BudgetExceededError
from llm_budget.pricing import default_registry
from llm_budget.store import UsageStore


class TestTrackCostsDecorator:
    """Tests for the track_costs decorator's wrapping behavior."""

    def test_bare_decorator_preserves_return_value(self) -> None:
        @track_costs
        def greet() -> str:
            return "hello"

        assert greet() == "hello"

    def test_decorator_with_arguments_preserves_return_value(self) -> None:
        @track_costs(project="test", max_budget=10.0)
        def greet() -> str:
            return "hello"

        assert greet() == "hello"

    def test_decorated_function_preserves_name(self) -> None:
        @track_costs
        def my_function() -> None:
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_decorator_with_all_parameters(self) -> None:
        def custom_extractor(resp: object) -> tuple[str, int, int]:
            return ("model", 0, 0)

        @track_costs(
            project="proj",
            model="gpt-4o",
            max_budget=5.0,
            reset="monthly",
            rate_limit=60,
            tpm_limit=100_000,
            extract_usage=custom_extractor,
        )
        def call_llm() -> str:
            return "response"

        assert call_llm() == "response"

    def test_decorated_function_passes_through_args_and_kwargs(self) -> None:
        @track_costs
        def add(a: int, b: int, extra: int = 0) -> int:
            return a + b + extra

        assert add(1, 2) == 3
        assert add(1, 2, extra=10) == 13


class TestTrackCostsBehavior:
    """Tests for the track_costs decorator's tracking, budget, and logging behavior."""

    def teardown_method(self) -> None:
        """Reset the shared store after each test."""
        store = getattr(self, "_store", None)
        if store is not None:
            store.close()
            self._store = None  # type: ignore[assignment]
        set_store(None)

    def _make_store(self, tmp_db_path: str) -> UsageStore:
        """Create a UsageStore at tmp_db_path and inject it via set_store."""
        store = UsageStore(db_path=tmp_db_path)
        set_store(store)
        self._store = store  # type: ignore[assignment]
        return store

    def test_budget_enforcement_raises_on_exceeded(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)
        store.log_usage("proj", "gpt-4o", 1000, 500, 5.0)

        @track_costs(project="proj", max_budget=3.0)
        def call_llm() -> str:
            return "response"

        with pytest.raises(BudgetExceededError):
            call_llm()

    def test_budget_enforcement_allows_within_budget(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)
        store.log_usage("proj", "gpt-4o", 100, 50, 1.0)

        @track_costs(project="proj", max_budget=10.0)
        def call_llm() -> str:
            return "ok"

        assert call_llm() == "ok"

    def test_budget_exactly_at_limit_raises(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)
        store.log_usage("proj", "gpt-4o", 100, 50, 5.0)

        @track_costs(project="proj", max_budget=5.0)
        def call_llm() -> str:
            return "response"

        with pytest.raises(BudgetExceededError):
            call_llm()

    def test_usage_logged_to_store(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        def extractor(resp: object) -> tuple[str, int, int]:
            return ("gpt-4o", 100, 50)

        @track_costs(project="proj", extract_usage=extractor)
        def call_llm() -> dict:
            return {"text": "hello"}

        call_llm()

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("proj") == pytest.approx(expected_cost)

    def test_extract_usage_exception_caught(self, tmp_db_path: str) -> None:
        self._make_store(tmp_db_path)

        def bad_extractor(resp: object) -> tuple[str, int, int]:
            raise ValueError("extraction failed")

        @track_costs(project="proj", extract_usage=bad_extractor)
        def call_llm() -> dict:
            return {"text": "hello"}

        with pytest.warns(match="extract_usage callback raised an exception"):
            result = call_llm()

        assert result == {"text": "hello"}

    def test_response_none_no_error(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        @track_costs(project="proj")
        def call_llm() -> None:
            return None

        call_llm()

        assert store.get_total_cost("proj") == 0.0
        assert store.get_usage_logs("proj") == []

    def test_no_usage_info_no_logging(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        @track_costs(project="proj")
        def call_llm() -> dict:
            return {"choices": [{"text": "hi"}]}

        call_llm()

        assert store.get_total_cost("proj") == 0.0
        assert store.get_usage_logs("proj") == []

    def test_model_override_takes_precedence(self, tmp_db_path: str) -> None:
        import warnings

        store = self._make_store(tmp_db_path)

        def extractor(resp: object) -> tuple[str, int, int]:
            return ("detected-model", 100, 50)

        @track_costs(project="proj", model="override-model", extract_usage=extractor)
        def call_llm() -> dict:
            return {"text": "hello"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call_llm()

        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "override-model"

    def test_set_store_injection(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        def extractor(resp: object) -> tuple[str, int, int]:
            return ("gpt-4o", 200, 100)

        @track_costs(project="inject-test", extract_usage=extractor)
        def call_llm() -> dict:
            return {"data": "value"}

        call_llm()

        expected_cost = default_registry.get_cost("gpt-4o", 200, 100)
        assert store.get_total_cost("inject-test") == pytest.approx(expected_cost)

    def test_function_exception_not_logged(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        @track_costs(project="proj")
        def call_llm() -> str:
            raise RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            call_llm()

        assert store.get_total_cost("proj") == 0.0
        assert store.get_usage_logs("proj") == []

    def test_budget_check_before_execution(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)
        store.log_usage("proj", "gpt-4o", 100, 50, 1.0)

        call_count = 0

        @track_costs(project="proj", max_budget=0.01)
        def call_llm() -> str:
            nonlocal call_count
            call_count += 1
            return "response"

        with pytest.raises(BudgetExceededError):
            call_llm()

        assert call_count == 0

    def test_multiple_calls_accumulate_cost(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        def extractor(resp: object) -> tuple[str, int, int]:
            return ("gpt-4o", 100, 50)

        @track_costs(project="proj", extract_usage=extractor)
        def call_llm() -> dict:
            return {"text": "hello"}

        call_llm()
        call_llm()
        call_llm()

        single_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("proj") == pytest.approx(single_cost * 3)
