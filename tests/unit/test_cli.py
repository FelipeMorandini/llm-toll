"""Unit tests for the CLI dashboard and new store query methods."""

from __future__ import annotations

import csv
import io

import pytest

from llm_toll import __version__
from llm_toll.cli import main
from llm_toll.store import UsageStore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, capsys, *args, db_path=None):
    """Run the CLI with the given arguments and return captured output."""
    argv = ["llm-toll", *args]
    if db_path:
        argv.extend(["--db", db_path])
    monkeypatch.setattr("sys.argv", argv)
    main()
    return capsys.readouterr()


def _seed_db(store: UsageStore) -> None:
    """Seed the database with representative multi-project, multi-model data."""
    store.log_usage(project="alpha", model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.05)
    store.log_usage(
        project="alpha", model="gpt-4o", input_tokens=200, output_tokens=100, cost=0.10
    )
    store.log_usage(
        project="alpha",
        model="claude-sonnet-4-20250514",
        input_tokens=150,
        output_tokens=75,
        cost=0.08,
    )
    store.log_usage(project="beta", model="gpt-4o", input_tokens=300, output_tokens=150, cost=0.20)
    store.log_usage(
        project="beta", model="gemini-1.5-pro", input_tokens=500, output_tokens=200, cost=0.12
    )


# ===========================================================================
# Store query method tests
# ===========================================================================


class TestStoreQueryMethods:
    """Tests for the new store query methods used by the CLI."""

    def test_get_all_project_summaries_empty(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        result = store.get_all_project_summaries()
        assert result == []
        store.close()

    def test_get_all_project_summaries(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        summaries = store.get_all_project_summaries()

        assert len(summaries) == 2
        # Ordered by total_cost DESC: beta=0.32, alpha=0.23
        by_project = {s["project"]: s for s in summaries}

        alpha = by_project["alpha"]
        assert alpha["call_count"] == 3
        assert alpha["total_input_tokens"] == 450
        assert alpha["total_output_tokens"] == 225
        assert alpha["total_cost"] == pytest.approx(0.23)

        beta = by_project["beta"]
        assert beta["call_count"] == 2
        assert beta["total_input_tokens"] == 800
        assert beta["total_output_tokens"] == 350
        assert beta["total_cost"] == pytest.approx(0.32)

        store.close()

    def test_get_model_summaries_no_filter(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        summaries = store.get_model_summaries()

        by_model = {s["model"]: s for s in summaries}
        assert len(by_model) == 3

        gpt4o = by_model["gpt-4o"]
        assert gpt4o["call_count"] == 3
        assert gpt4o["total_input_tokens"] == 600
        assert gpt4o["total_output_tokens"] == 300
        assert gpt4o["total_cost"] == pytest.approx(0.35)

        claude = by_model["claude-sonnet-4-20250514"]
        assert claude["call_count"] == 1
        assert claude["total_cost"] == pytest.approx(0.08)

        gemini = by_model["gemini-1.5-pro"]
        assert gemini["call_count"] == 1
        assert gemini["total_cost"] == pytest.approx(0.12)

        store.close()

    def test_get_model_summaries_with_project(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        summaries = store.get_model_summaries(project="alpha")

        by_model = {s["model"]: s for s in summaries}
        assert len(by_model) == 2
        assert "gpt-4o" in by_model
        assert "claude-sonnet-4-20250514" in by_model
        # gemini-1.5-pro belongs to beta only
        assert "gemini-1.5-pro" not in by_model

        gpt4o = by_model["gpt-4o"]
        assert gpt4o["call_count"] == 2
        assert gpt4o["total_cost"] == pytest.approx(0.15)

        store.close()

    def test_get_usage_logs_filtered_by_project(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        logs = store.get_usage_logs_filtered(project="beta")

        assert len(logs) == 2
        assert all(log["project"] == "beta" for log in logs)
        store.close()

    def test_get_usage_logs_filtered_by_model(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        logs = store.get_usage_logs_filtered(model="gpt-4o")

        assert len(logs) == 3
        assert all(log["model"] == "gpt-4o" for log in logs)
        store.close()

    def test_get_usage_logs_filtered_by_both(self, tmp_db_path: str) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        logs = store.get_usage_logs_filtered(project="alpha", model="gpt-4o")

        assert len(logs) == 2
        assert all(log["project"] == "alpha" and log["model"] == "gpt-4o" for log in logs)
        store.close()


# ===========================================================================
# CLI --stats tests
# ===========================================================================


class TestCliStats:
    """Tests for the --stats CLI command."""

    def test_stats_all_projects(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        store.close()

        monkeypatch.setenv("NO_COLOR", "1")
        out = _run_cli(monkeypatch, capsys, "--stats", db_path=tmp_db_path)

        assert "alpha" in out.out
        assert "beta" in out.out
        assert "TOTAL" in out.out
        assert "llm-toll Usage Summary" in out.out

    def test_stats_with_project(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        store.close()

        monkeypatch.setenv("NO_COLOR", "1")
        out = _run_cli(monkeypatch, capsys, "--stats", "--project", "alpha", db_path=tmp_db_path)

        # Per-model breakdown
        assert "gpt-4o" in out.out
        assert "claude-sonnet-4-20250514" in out.out
        assert "TOTAL" in out.out
        assert "Usage for project: alpha" in out.out

    def test_stats_empty_db(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        # Ensure the DB exists but is empty
        store = UsageStore(db_path=tmp_db_path)
        store.close()

        monkeypatch.setenv("NO_COLOR", "1")
        out = _run_cli(monkeypatch, capsys, "--stats", db_path=tmp_db_path)

        assert "No usage data found." in out.out

    def test_stats_with_model(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        store.close()

        monkeypatch.setenv("NO_COLOR", "1")
        out = _run_cli(monkeypatch, capsys, "--stats", "--model", "gpt-4o", db_path=tmp_db_path)

        # Per-project breakdown for this model
        assert "alpha" in out.out
        assert "beta" in out.out
        assert "TOTAL" in out.out
        assert "Usage for model: gpt-4o" in out.out


# ===========================================================================
# CLI --reset tests
# ===========================================================================


class TestCliReset:
    """Tests for the --reset CLI command."""

    def test_reset_success(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        store = UsageStore(db_path=tmp_db_path)
        store.log_usage(
            project="proj", model="gpt-4o", input_tokens=10, output_tokens=5, cost=1.50
        )
        store.close()

        monkeypatch.setenv("NO_COLOR", "1")
        out = _run_cli(monkeypatch, capsys, "--reset", "--project", "proj", db_path=tmp_db_path)

        assert "Budget reset" in out.out
        assert "$1.5000" in out.out

        # Verify budget is actually 0
        store2 = UsageStore(db_path=tmp_db_path)
        assert store2.get_total_cost("proj") == pytest.approx(0.0)
        store2.close()

    def test_reset_no_project(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        with pytest.raises(SystemExit) as exc_info:
            _run_cli(monkeypatch, capsys, "--reset", db_path=tmp_db_path)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "--project" in captured.err


# ===========================================================================
# CLI --export tests
# ===========================================================================


class TestCliExport:
    """Tests for the --export csv CLI command."""

    def test_export_csv_stdout(self, tmp_db_path: str, monkeypatch, capsys) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        store.close()

        monkeypatch.setenv("NO_COLOR", "1")
        out = _run_cli(monkeypatch, capsys, "--export", "csv", db_path=tmp_db_path)

        reader = csv.reader(io.StringIO(out.out))
        rows = list(reader)

        # Header + 5 data rows
        assert len(rows) == 6
        assert rows[0] == [
            "project",
            "model",
            "input_tokens",
            "output_tokens",
            "cost",
            "created_at",
        ]

        # Verify all data rows have the right number of columns
        for row in rows[1:]:
            assert len(row) == 6

    def test_export_csv_to_file(self, tmp_db_path: str, tmp_path, monkeypatch, capsys) -> None:
        store = UsageStore(db_path=tmp_db_path)
        _seed_db(store)
        store.close()

        output_file = str(tmp_path / "export.csv")
        monkeypatch.setenv("NO_COLOR", "1")
        _run_cli(
            monkeypatch,
            capsys,
            "--export",
            "csv",
            "--output",
            output_file,
            db_path=tmp_db_path,
        )

        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 6
        assert rows[0][0] == "project"


# ===========================================================================
# CLI --version tests
# ===========================================================================


class TestCliVersion:
    """Tests for the --version CLI flag."""

    def test_version(self, monkeypatch, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run_cli(monkeypatch, capsys, "--version")

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out
