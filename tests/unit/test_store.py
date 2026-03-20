"""Unit tests for UsageStore."""

from __future__ import annotations

from llm_budget.store import UsageStore


class TestUsageStore:
    """Tests for UsageStore instantiation and stub behavior."""

    def test_instantiate_with_default_none_path(self) -> None:
        store = UsageStore()
        assert store._db_path is None

    def test_instantiate_with_custom_path(self, tmp_path: object) -> None:
        from pathlib import Path

        db_path = str(Path(str(tmp_path)) / "test.db")
        store = UsageStore(db_path=db_path)
        assert store._db_path == db_path

    def test_get_total_cost_returns_zero(self) -> None:
        store = UsageStore()
        assert store.get_total_cost("my-project") == 0.0

    def test_log_usage_does_not_raise(self) -> None:
        store = UsageStore()
        store.log_usage(
            project="test",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.5,
        )
