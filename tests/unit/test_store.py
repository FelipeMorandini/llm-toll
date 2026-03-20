"""Unit tests for UsageStore."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from llm_budget.store import UsageStore


class TestUsageStore:
    """Tests for UsageStore instantiation and persistence."""

    def test_instantiate_with_default_none_path(self) -> None:
        store = UsageStore()
        expected = str(Path.home() / ".llm_budget.db")
        assert store._db_path == expected

    def test_instantiate_with_custom_path(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        store = UsageStore(db_path=db_path)
        assert store._db_path == str(Path(db_path).expanduser().resolve())

    def test_get_total_cost_returns_zero_for_unknown_project(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        assert store.get_total_cost("my-project") == 0.0
        store.close()

    def test_log_usage_persists_cost(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(
            project="test",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.5,
        )
        assert store.get_total_cost("test") == 0.5
        store.close()


class TestUsageStorePersistence:
    """Tests for UsageStore persistence, isolation, and edge cases."""

    def test_log_usage_accumulates_cost(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(project="p", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.10)
        store.log_usage(project="p", model="gpt-4o", input_tokens=20, output_tokens=10, cost=0.25)
        store.log_usage(project="p", model="gpt-4o", input_tokens=30, output_tokens=15, cost=0.65)
        assert store.get_total_cost("p") == pytest.approx(1.0)
        store.close()

    def test_projects_are_isolated(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(
            project="proj-a", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.50
        )
        store.log_usage(
            project="proj-b", model="gpt-4o", input_tokens=10, output_tokens=5, cost=1.50
        )
        store.log_usage(
            project="proj-a", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.25
        )
        assert store.get_total_cost("proj-a") == pytest.approx(0.75)
        assert store.get_total_cost("proj-b") == pytest.approx(1.50)
        store.close()

    def test_get_total_cost_unknown_project_returns_zero(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        # Force schema creation by logging to a different project
        store.log_usage(project="existing", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        assert store.get_total_cost("nonexistent") == 0.0
        store.close()

    def test_get_usage_logs_returns_records(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        for i in range(3):
            store.log_usage(
                project="p",
                model=f"model-{i}",
                input_tokens=10 * (i + 1),
                output_tokens=5 * (i + 1),
                cost=0.1 * (i + 1),
            )
        logs = store.get_usage_logs("p")
        assert len(logs) == 3
        required_keys = {"project", "model", "input_tokens", "output_tokens", "cost", "created_at"}
        for log in logs:
            assert required_keys.issubset(log.keys())
            assert log["project"] == "p"
        store.close()

    def test_get_usage_logs_respects_limit(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        for _ in range(5):
            store.log_usage(project="p", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        logs = store.get_usage_logs("p", limit=2)
        assert len(logs) == 2
        store.close()

    def test_get_usage_logs_ordered_desc(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(project="p", model="first", input_tokens=1, output_tokens=1, cost=0.01)
        store.log_usage(project="p", model="second", input_tokens=1, output_tokens=1, cost=0.01)
        store.log_usage(project="p", model="third", input_tokens=1, output_tokens=1, cost=0.01)
        logs = store.get_usage_logs("p")
        assert logs[0]["model"] == "third"
        assert logs[-1]["model"] == "first"
        store.close()

    def test_get_usage_logs_empty_project(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        # Ensure DB is initialized
        store.log_usage(project="other", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        logs = store.get_usage_logs("nonexistent")
        assert logs == []
        store.close()

    def test_reset_budget(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(project="p", model="m", input_tokens=1, output_tokens=1, cost=1.00)
        store.log_usage(project="p", model="m", input_tokens=1, output_tokens=1, cost=2.00)
        store.reset_budget("p")
        assert store.get_total_cost("p") == pytest.approx(0.0)
        # Logs should still exist
        logs = store.get_usage_logs("p")
        assert len(logs) == 2
        store.close()

    def test_reset_budget_nonexistent_project(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        # Ensure DB is initialized
        store.log_usage(project="other", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        # Should not raise
        store.reset_budget("nonexistent")
        store.close()

    def test_sql_injection_project_name(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        evil_project = "'; DROP TABLE usage_logs; --"
        store.log_usage(
            project=evil_project, model="m", input_tokens=1, output_tokens=1, cost=0.42
        )
        # Tables should still exist and data should be correct
        assert store.get_total_cost(evil_project) == pytest.approx(0.42)
        logs = store.get_usage_logs(evil_project)
        assert len(logs) == 1
        assert logs[0]["project"] == evil_project
        # Verify the usage_logs table still exists by querying it
        conn = store._get_conn()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}
        assert "usage_logs" in table_names
        assert "budgets" in table_names
        store.close()

    def test_db_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not contain"):
            UsageStore(db_path="/tmp/../etc/passwd")

    def test_db_path_traversal_rejected_relative(self) -> None:
        with pytest.raises(ValueError, match="must not contain"):
            UsageStore(db_path="../../etc/passwd")

    def test_close_and_reopen(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(project="p", model="m", input_tokens=10, output_tokens=5, cost=0.99)
        store.close()

        store2 = UsageStore(db_path=tmp_db_path)
        assert store2.get_total_cost("p") == pytest.approx(0.99)
        logs = store2.get_usage_logs("p")
        assert len(logs) == 1
        store2.close()

    def test_concurrent_writes(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for _ in range(100):
                    store.log_usage(
                        project="p", model="m", input_tokens=1, output_tokens=1, cost=0.01
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert store.get_total_cost("p") == pytest.approx(10.0)
        store.close()

    def test_tables_created_on_first_access(self, tmp_db_path: str) -> None:
        db_file = Path(tmp_db_path)
        assert not db_file.exists()
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(project="p", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        assert db_file.exists()
        store.close()

    def test_zero_cost_log(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(project="p", model="local", input_tokens=100, output_tokens=50, cost=0.0)
        assert store.get_total_cost("p") == pytest.approx(0.0)
        store.log_usage(project="p", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.50)
        assert store.get_total_cost("p") == pytest.approx(0.50)
        logs = store.get_usage_logs("p")
        assert len(logs) == 2
        store.close()
