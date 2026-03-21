"""Unit tests for CostReporter."""

from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

from llm_toll.reporter import CostReporter


class TestCostReporterOutput:
    """Tests for per-call output formatting and color codes."""

    def test_report_call_format(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=1_200, output_tokens=350, cost=0.0042)
        output = buf.getvalue()
        assert "gpt-4o" in output
        assert "1,200" in output
        assert "350" in output
        assert "$0.0042" in output

    def test_report_call_green_cost(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.005)
        output = buf.getvalue()
        assert "\033[32m" in output

    def test_report_call_yellow_cost(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.05)
        output = buf.getvalue()
        assert "\033[33m" in output

    def test_report_call_red_cost(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.20)
        output = buf.getvalue()
        assert "\033[31m" in output

    def test_report_call_cyan_model(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)
        output = buf.getvalue()
        assert "\033[36m" in output
        assert "\033[36mgpt-4o\033[0m" in output

    def test_no_color_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)
        output = buf.getvalue()
        assert "\033" not in output
        assert "gpt-4o" in output

    def test_enabled_false_no_output(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(enabled=False, file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)
        assert buf.getvalue() == ""

    def test_enabled_false_still_accumulates(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(enabled=False, file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)
        assert reporter._call_count == 1


class TestCostReporterSession:
    """Tests for session summary output."""

    def test_session_summary_format(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        for _ in range(3):
            reporter.report_call(model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.01)
        buf.truncate(0)
        buf.seek(0)
        reporter.report_session()
        output = buf.getvalue()
        assert "3 calls" in output
        assert "300" in output
        assert "150" in output
        assert "$0.0300" in output

    def test_session_empty_no_output(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_session()
        assert buf.getvalue() == ""

    def test_session_bold_cost(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)
        buf.truncate(0)
        buf.seek(0)
        reporter.report_session()
        output = buf.getvalue()
        assert "\033[1;" in output

    def test_reset_zeroes_accumulators(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        reporter.report_call(model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.05)
        reporter.reset()
        assert reporter._session_cost == 0.0
        assert reporter._session_input_tokens == 0
        assert reporter._session_output_tokens == 0
        assert reporter._call_count == 0


class TestCostReporterThreadSafety:
    """Tests for concurrent access to the reporter."""

    def test_concurrent_report_calls(self) -> None:
        buf = io.StringIO()
        reporter = CostReporter(enabled=False, file=buf)

        def worker() -> None:
            for _ in range(100):
                reporter.report_call(model="gpt-4o", input_tokens=1, output_tokens=1, cost=0.001)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert reporter._call_count == 1000


class TestCostReporterIntegration:
    """Integration test: decorator -> reporter pipeline."""

    def test_decorator_calls_reporter(self, tmp_path: Path) -> None:
        from llm_toll.decorator import set_reporter, set_store, track_costs
        from llm_toll.store import UsageStore

        buf = io.StringIO()
        reporter = CostReporter(file=buf)
        db_path = str(tmp_path / "test.db")
        store = UsageStore(db_path=db_path)

        set_reporter(reporter)
        set_store(store)
        try:

            @track_costs(extract_usage=lambda resp: ("gpt-4o", 10, 5))
            def my_llm_call() -> dict:
                return {"result": "hello"}

            my_llm_call()
            output = buf.getvalue()
            assert "gpt-4o" in output
            assert output != ""
        finally:
            set_reporter(None)
            set_store(None)
