"""Tests for the web dashboard feature."""

from __future__ import annotations

import http.server
import json
import threading
from urllib.request import urlopen

import pytest

from llm_toll.dashboard import DashboardHandler
from llm_toll.store import SQLiteStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_dashboard(store: SQLiteStore, port: int = 0):
    """Start a dashboard server on an OS-assigned port in a background thread."""
    handler = type("H", (DashboardHandler,), {"store": store})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    actual_port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, actual_port


def _seed_store(store: SQLiteStore) -> None:
    """Insert sample usage data into the store."""
    store.log_usage("my-project", "gpt-4o", 100, 50, 0.005)
    store.log_usage("my-project", "gpt-4o", 200, 100, 0.010)
    store.log_usage("other-project", "claude-sonnet-4-20250514", 300, 150, 0.020)


# ---------------------------------------------------------------------------
# Store method tests
# ---------------------------------------------------------------------------


class TestGetDailyCostTrends:
    def test_get_daily_cost_trends_empty(self, tmp_db_path: str) -> None:
        """Empty database returns an empty list."""
        store = SQLiteStore(db_path=tmp_db_path)
        try:
            result = store.get_daily_cost_trends()
            assert result == []
        finally:
            store.close()

    def test_get_daily_cost_trends_with_data(self, tmp_db_path: str) -> None:
        """Seeded data is aggregated by date with correct format."""
        store = SQLiteStore(db_path=tmp_db_path)
        try:
            _seed_store(store)
            result = store.get_daily_cost_trends(days=30)

            assert len(result) >= 1
            row = result[0]
            # Verify expected keys
            assert "date" in row
            assert "daily_cost" in row
            assert "daily_input_tokens" in row
            assert "daily_output_tokens" in row
            assert "call_count" in row
            # Date should be YYYY-MM-DD format
            assert len(row["date"]) == 10
            assert row["date"].count("-") == 2
            # All three calls land on today
            assert row["call_count"] == 3
            assert row["daily_cost"] == pytest.approx(0.035)
            assert row["daily_input_tokens"] == 600
            assert row["daily_output_tokens"] == 300
        finally:
            store.close()


class TestGetBudgetUtilization:
    def test_get_budget_utilization_empty(self, tmp_db_path: str) -> None:
        """Empty database returns an empty list."""
        store = SQLiteStore(db_path=tmp_db_path)
        try:
            result = store.get_budget_utilization()
            assert result == []
        finally:
            store.close()

    def test_get_budget_utilization_with_data(self, tmp_db_path: str) -> None:
        """Seeded usage creates budget rows with correct totals."""
        store = SQLiteStore(db_path=tmp_db_path)
        try:
            _seed_store(store)
            result = store.get_budget_utilization()

            assert len(result) == 2
            # Ordered by total_cost DESC
            by_project = {r["project"]: r for r in result}

            assert "my-project" in by_project
            assert "other-project" in by_project
            assert by_project["my-project"]["total_cost"] == pytest.approx(0.015)
            assert by_project["other-project"]["total_cost"] == pytest.approx(0.020)
            # Each row should have updated_at
            for row in result:
                assert "updated_at" in row
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Dashboard API tests
# ---------------------------------------------------------------------------


class TestDashboardAPI:
    @pytest.fixture(autouse=True)
    def _setup_server(self, tmp_db_path: str) -> None:
        """Set up a seeded store and start the dashboard server."""
        self.store = SQLiteStore(db_path=tmp_db_path)
        _seed_store(self.store)
        self.server, self.port = _start_dashboard(self.store)
        self.base_url = f"http://127.0.0.1:{self.port}"
        yield
        self.server.shutdown()
        self.server.server_close()
        self.store.close()

    def test_index_returns_html(self) -> None:
        """GET / returns 200 with HTML containing 'llm-toll'."""
        with urlopen(f"{self.base_url}/") as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "llm-toll" in body
            assert resp.headers.get("Content-Type", "").startswith("text/html")

    def test_api_summary(self) -> None:
        """GET /api/summary returns JSON with expected keys and values."""
        with urlopen(f"{self.base_url}/api/summary") as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "total_cost" in data
            assert "total_calls" in data
            assert "project_count" in data
            assert "model_count" in data
            assert data["total_calls"] == 3
            assert data["project_count"] == 2
            assert data["model_count"] == 2
            assert data["total_cost"] == pytest.approx(0.035)

    def test_api_trends(self) -> None:
        """GET /api/trends returns a JSON array."""
        with urlopen(f"{self.base_url}/api/trends") as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert isinstance(data, list)
            assert len(data) >= 1
            assert "date" in data[0]
            assert "daily_cost" in data[0]

    def test_api_projects(self) -> None:
        """GET /api/projects returns a JSON array with project summaries."""
        with urlopen(f"{self.base_url}/api/projects") as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert isinstance(data, list)
            assert len(data) == 2
            projects = {row["project"] for row in data}
            assert projects == {"my-project", "other-project"}

    def test_api_models(self) -> None:
        """GET /api/models returns a JSON array with model summaries."""
        with urlopen(f"{self.base_url}/api/models") as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert isinstance(data, list)
            assert len(data) == 2
            models = {row["model"] for row in data}
            assert models == {"gpt-4o", "claude-sonnet-4-20250514"}

    def test_api_unknown_path_returns_404(self) -> None:
        """GET /unknown returns 404."""
        from urllib.error import HTTPError

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{self.base_url}/unknown")
        assert exc_info.value.code == 404
