"""Unit tests for PostgresStore with fully mocked psycopg2."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_pg_mocks():
    """Create a full set of psycopg2 mocks wired together."""
    mock_psycopg2 = MagicMock()

    mock_pool_instance = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Wire: pool.getconn() -> conn
    mock_pool_instance.getconn.return_value = mock_conn

    # Wire: conn.cursor() used as context manager yields mock_cursor
    cursor_cm = MagicMock()
    cursor_cm.__enter__ = MagicMock(return_value=mock_cursor)
    cursor_cm.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = cursor_cm

    # Wire: conn used as context manager yields itself
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    # psycopg2.pool.ThreadedConnectionPool() -> pool_instance
    mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_pool_instance

    return {
        "psycopg2": mock_psycopg2,
        "pool_instance": mock_pool_instance,
        "conn": mock_conn,
        "cursor": mock_cursor,
    }


@pytest.fixture
def pg_mocks():
    """Fixture that patches sys.modules for psycopg2 and returns mock objects."""
    mocks = _make_pg_mocks()
    # Clear cached _postgres_store so the patched modules are picked up fresh
    sys.modules.pop("llm_toll._postgres_store", None)
    with patch.dict(
        "sys.modules",
        {
            "psycopg2": mocks["psycopg2"],
            "psycopg2.pool": mocks["psycopg2"].pool,
        },
    ):
        yield mocks
    # Clean up so other tests are not affected
    sys.modules.pop("llm_toll._postgres_store", None)


def _create_store(pg_mocks):
    """Import and instantiate PostgresStore under mocked psycopg2."""
    mod = importlib.import_module("llm_toll._postgres_store")
    return mod.PostgresStore(dsn="postgresql://user:pass@localhost:5432/testdb")


class TestPostgresStoreInstantiation:
    """Tests for PostgresStore creation and schema initialization."""

    def test_postgres_store_is_base_store(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        # Use MRO name check to avoid class identity issues from module reimport
        base_names = [cls.__name__ for cls in type(store).__mro__]
        assert "BaseStore" in base_names
        assert "PostgresStore" in base_names

    def test_schema_initialization(self, pg_mocks) -> None:
        _create_store(pg_mocks)
        cursor = pg_mocks["cursor"]

        # _init_schema should have called execute with CREATE TABLE statements
        assert cursor.execute.called
        schema_sql = cursor.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS usage_logs" in schema_sql
        assert "CREATE TABLE IF NOT EXISTS budgets" in schema_sql
        assert "CREATE INDEX IF NOT EXISTS idx_usage_logs_project_created" in schema_sql

    def test_pool_created_with_dsn(self, pg_mocks) -> None:
        _create_store(pg_mocks)
        pg_mocks["psycopg2"].pool.ThreadedConnectionPool.assert_called_once_with(
            1, 10, "postgresql://user:pass@localhost:5432/testdb"
        )


class TestPostgresStoreLogUsage:
    """Tests for PostgresStore.log_usage."""

    def test_log_usage_executes_insert(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        cursor = pg_mocks["cursor"]

        # Reset call tracking after schema init
        cursor.execute.reset_mock()

        store.log_usage(
            project="my-project",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.5,
        )

        # Should have two execute calls: INSERT into usage_logs + UPSERT into budgets
        assert cursor.execute.call_count == 2

        insert_sql = cursor.execute.call_args_list[0][0][0]
        assert "INSERT INTO usage_logs" in insert_sql

        upsert_sql = cursor.execute.call_args_list[1][0][0]
        assert "INSERT INTO budgets" in upsert_sql
        assert "ON CONFLICT" in upsert_sql

        # Verify params include the project and model
        insert_params = cursor.execute.call_args_list[0][0][1]
        assert insert_params[0] == "my-project"
        assert insert_params[1] == "gpt-4o"
        assert insert_params[2] == 100
        assert insert_params[3] == 50
        assert insert_params[4] == 0.5

    def test_log_usage_returns_conn_to_pool(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        pool = pg_mocks["pool_instance"]
        conn = pg_mocks["conn"]

        pool.putconn.reset_mock()
        store.log_usage(project="p", model="m", input_tokens=1, output_tokens=1, cost=0.01)
        pool.putconn.assert_called_with(conn)


class TestPostgresStoreGetTotalCost:
    """Tests for PostgresStore.get_total_cost."""

    def test_get_total_cost_returns_float(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        cursor = pg_mocks["cursor"]

        cursor.fetchone.return_value = (1.23,)

        result = store.get_total_cost("my-project")
        assert result == 1.23
        assert isinstance(result, float)

    def test_get_total_cost_no_data_returns_zero(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        cursor = pg_mocks["cursor"]

        cursor.fetchone.return_value = None

        result = store.get_total_cost("nonexistent")
        assert result == 0.0

    def test_get_total_cost_queries_correct_project(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        cursor = pg_mocks["cursor"]
        cursor.fetchone.return_value = (0.5,)
        cursor.execute.reset_mock()

        store.get_total_cost("specific-project")

        assert cursor.execute.called
        sql = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]
        assert "SELECT total_cost FROM budgets" in sql
        assert params == ("specific-project",)


class TestPostgresStoreClose:
    """Tests for PostgresStore.close."""

    def test_close_closes_pool(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        pool = pg_mocks["pool_instance"]

        store.close()
        pool.closeall.assert_called_once()

    def test_close_via_context_manager(self, pg_mocks) -> None:
        store = _create_store(pg_mocks)
        pool = pg_mocks["pool_instance"]

        with store:
            pass
        pool.closeall.assert_called_once()


class TestPostgresStoreImportError:
    """Tests for clear error message when psycopg2 is not installed."""

    def test_import_error_message(self) -> None:
        # Ensure _postgres_store is not cached
        sys.modules.pop("llm_toll._postgres_store", None)

        # Setting a module to None in sys.modules causes ImportError on import
        with patch.dict(
            "sys.modules",
            {"psycopg2": None, "psycopg2.pool": None},
        ):
            mod = importlib.import_module("llm_toll._postgres_store")
            with pytest.raises(ImportError, match="psycopg2 is required"):
                mod.PostgresStore(dsn="postgresql://localhost/test")

        # Clean up
        sys.modules.pop("llm_toll._postgres_store", None)
