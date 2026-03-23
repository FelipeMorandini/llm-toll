"""Unit tests for create_store factory function and BaseStore ABC."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_toll.store import BaseStore, SQLiteStore, UsageStore, create_store


class TestCreateStoreFactory:
    """Tests for the create_store() factory function."""

    def test_create_store_default_returns_sqlite(self) -> None:
        store = create_store()
        assert isinstance(store, SQLiteStore)

    def test_create_store_with_path_returns_sqlite(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        store = create_store(url=db_path)
        assert isinstance(store, SQLiteStore)

    def test_create_store_postgres_url(self) -> None:
        mock_psycopg2 = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = MagicMock()

        # Clear cached module so the patched imports take effect
        saved = sys.modules.pop("llm_toll._postgres_store", None)
        try:
            with patch.dict(
                "sys.modules",
                {"psycopg2": mock_psycopg2, "psycopg2.pool": mock_psycopg2.pool},
            ):
                store = create_store(url="postgresql://user:pass@localhost:5432/mydb")

                # Check type by class name (avoids identity issues from module reimport)
                assert type(store).__name__ == "PostgresStore"
                assert any(cls.__name__ == "BaseStore" for cls in type(store).__mro__)
                mock_psycopg2.pool.ThreadedConnectionPool.assert_called_once_with(
                    1, 10, "postgresql://user:pass@localhost:5432/mydb"
                )
        finally:
            # Restore or remove to avoid polluting other tests
            sys.modules.pop("llm_toll._postgres_store", None)
            if saved is not None:
                sys.modules["llm_toll._postgres_store"] = saved

    def test_create_store_postgres_scheme(self) -> None:
        mock_psycopg2 = MagicMock()
        mock_psycopg2.pool.ThreadedConnectionPool.return_value = MagicMock()

        saved = sys.modules.pop("llm_toll._postgres_store", None)
        try:
            with patch.dict(
                "sys.modules",
                {"psycopg2": mock_psycopg2, "psycopg2.pool": mock_psycopg2.pool},
            ):
                store = create_store(url="postgres://user:pass@localhost:5432/mydb")

                assert type(store).__name__ == "PostgresStore"
                assert any(cls.__name__ == "BaseStore" for cls in type(store).__mro__)
                mock_psycopg2.pool.ThreadedConnectionPool.assert_called_once_with(
                    1, 10, "postgres://user:pass@localhost:5432/mydb"
                )
        finally:
            sys.modules.pop("llm_toll._postgres_store", None)
            if saved is not None:
                sys.modules["llm_toll._postgres_store"] = saved


class TestUsageStoreAlias:
    """Tests for the UsageStore backward-compatible alias."""

    def test_usage_store_is_sqlite_store(self) -> None:
        assert UsageStore is SQLiteStore

    def test_sqlite_store_is_base_store(self, tmp_db_path: str) -> None:
        store = SQLiteStore(db_path=tmp_db_path)
        assert isinstance(store, BaseStore)
        store.close()


class TestBaseStoreContextManager:
    """Tests for BaseStore __enter__/__exit__ protocol."""

    def test_base_store_context_manager(self, tmp_db_path: str) -> None:
        store = SQLiteStore(db_path=tmp_db_path)
        with store as s:
            assert s is store
            s.log_usage(project="p", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        # After exiting, connection should be closed
        assert store._conn is None

    def test_base_store_exit_calls_close(self) -> None:
        """Verify __exit__ delegates to close() on any BaseStore subclass."""
        mock_close = MagicMock()

        class FakeStore(BaseStore):
            def log_usage(self, *a, **kw) -> None: ...
            def log_usage_if_within_budget(self, *a, **kw) -> float:
                return 0.0

            def get_total_cost(self, project: str) -> float:
                return 0.0

            def get_usage_logs(self, *a, **kw) -> list:
                return []

            def get_all_project_summaries(self) -> list:
                return []

            def get_model_summaries(self, *a, **kw) -> list:
                return []

            def get_project_summaries_for_model(self, *a, **kw) -> list:
                return []

            def get_usage_logs_filtered(self, *a, **kw) -> list:
                return []

            def get_daily_cost_trends(self, *a, **kw) -> list:
                return []

            def get_budget_utilization(self) -> list:
                return []

            def reset_budget(self, project: str) -> None: ...
            def close(self) -> None:
                mock_close()

        fake = FakeStore()
        with fake:
            pass
        mock_close.assert_called_once()
