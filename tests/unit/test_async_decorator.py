"""Unit tests for the async branches of the track_costs decorator."""

from __future__ import annotations

import pytest

from llm_toll.decorator import set_store, track_costs
from llm_toll.exceptions import BudgetExceededError, LocalRateLimitError
from llm_toll.pricing import default_registry
from llm_toll.store import UsageStore


class TestAsyncTrackCostsDecorator:
    """Tests for the track_costs decorator on async functions."""

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

    @pytest.mark.asyncio
    async def test_async_function_preserves_return_value(self, tmp_db_path: str) -> None:
        """@track_costs on async def should preserve the return value."""
        self._make_store(tmp_db_path)

        @track_costs(project="proj")
        async def greet() -> str:
            return "hello async"

        result = await greet()
        assert result == "hello async"

    @pytest.mark.asyncio
    async def test_async_function_preserves_name(self) -> None:
        """functools.wraps should preserve __name__ and __doc__ on async wrappers."""

        @track_costs
        async def my_async_function() -> None:
            """My async docstring."""

        assert my_async_function.__name__ == "my_async_function"
        assert my_async_function.__doc__ == "My async docstring."

    @pytest.mark.asyncio
    async def test_async_budget_enforcement(self, tmp_db_path: str) -> None:
        """Pre-seeded cost > budget should raise BudgetExceededError before execution."""
        store = self._make_store(tmp_db_path)
        store.log_usage("proj", "gpt-4o", 1000, 500, 5.0)

        call_count = 0

        @track_costs(project="proj", max_budget=3.0)
        async def call_llm() -> str:
            nonlocal call_count
            call_count += 1
            return "response"

        with pytest.raises(BudgetExceededError):
            await call_llm()

        # The function body should never have executed
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_async_usage_logged_to_store(self, tmp_db_path: str) -> None:
        """With extract_usage, cost should be logged after await."""
        store = self._make_store(tmp_db_path)

        def extractor(resp: object) -> tuple[str, int, int]:
            return ("gpt-4o", 100, 50)

        @track_costs(project="proj", extract_usage=extractor)
        async def call_llm() -> dict:
            return {"text": "hello"}

        await call_llm()

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("proj") == pytest.approx(expected_cost)
        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_async_function_exception_not_logged(self, tmp_db_path: str) -> None:
        """If the async function raises, no cost should be logged."""
        store = self._make_store(tmp_db_path)

        @track_costs(project="proj")
        async def call_llm() -> str:
            raise RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            await call_llm()

        assert store.get_total_cost("proj") == 0.0
        assert store.get_usage_logs("proj") == []

    @pytest.mark.asyncio
    async def test_async_rate_limit_enforcement(self, tmp_db_path: str) -> None:
        """rate_limit=2 should allow 2 calls but reject the 3rd."""
        self._make_store(tmp_db_path)

        @track_costs(project="proj", rate_limit=2)
        async def call_llm() -> str:
            return "ok"

        # First two calls should succeed
        assert await call_llm() == "ok"
        assert await call_llm() == "ok"

        # Third call should raise LocalRateLimitError
        with pytest.raises(LocalRateLimitError):
            await call_llm()

    def test_sync_function_still_works(self, tmp_db_path: str) -> None:
        """Sync functions should be unaffected by async additions."""
        self._make_store(tmp_db_path)

        @track_costs(project="proj")
        def greet() -> str:
            return "hello sync"

        assert greet() == "hello sync"
