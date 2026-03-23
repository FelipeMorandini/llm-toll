"""End-to-end integration tests for the llm_toll scaffolding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from pytest import approx


def test_full_public_api_importable() -> None:
    """Verify every name in __all__ is importable from llm_toll."""
    import llm_toll

    expected_names = [
        "BaseStore",
        "BudgetExceededError",
        "CostReporter",
        "LangChainCallback",
        "LiteLLMCallback",
        "LocalRateLimitError",
        "PricingMatrixOutdatedWarning",
        "PricingRegistry",
        "RateLimiter",
        "SQLiteStore",
        "UsageStore",
        "__version__",
        "create_store",
        "default_registry",
        "set_reporter",
        "set_store",
        "track_costs",
        "update_pricing",
    ]
    for name in expected_names:
        assert hasattr(llm_toll, name), f"{name} missing from llm_toll"
    assert set(expected_names) == set(llm_toll.__all__)


def test_track_costs_bare_decorator_with_dict_response() -> None:
    """@track_costs bare on a function returning a dict (simulated LLM response)."""
    from llm_toll import track_costs

    @track_costs
    def call_llm() -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    result = call_llm()
    assert isinstance(result, dict)
    assert result["model"] == "gpt-4o"
    assert result["usage"]["prompt_tokens"] == 10


def test_track_costs_with_project_kwarg() -> None:
    """@track_costs(project="test") on a function returning a dict."""
    from llm_toll import track_costs

    @track_costs(project="test")
    def call_llm() -> dict[str, Any]:
        return {"model": "claude-sonnet-4-20250514", "content": "Hi"}

    result = call_llm()
    assert isinstance(result, dict)
    assert result["model"] == "claude-sonnet-4-20250514"


def test_track_costs_preserves_function_metadata() -> None:
    """Decorated function should preserve __name__ and __doc__."""
    from llm_toll import track_costs

    @track_costs
    def my_special_func() -> str:
        """My docstring."""
        return "ok"

    assert my_special_func.__name__ == "my_special_func"
    assert my_special_func.__doc__ == "My docstring."


def test_track_costs_with_all_kwargs(tmp_db_path: str) -> None:
    """Decorator accepts all documented keyword arguments without error."""
    from llm_toll import UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(
            project="integration-test",
            model="gpt-4o",
            max_budget=50.0,
            reset="monthly",
            rate_limit=60,
            tpm_limit=100000,
            extract_usage=lambda resp: ("gpt-4o", 10, 5),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        assert call_llm() == {"result": "ok"}
    finally:
        store.close()
        set_store(None)


def test_track_costs_passes_args_and_kwargs() -> None:
    """Decorated function correctly receives positional and keyword args."""
    from llm_toll import track_costs

    @track_costs
    def call_llm(prompt: str, temperature: float = 0.7) -> dict[str, Any]:
        return {"prompt": prompt, "temperature": temperature}

    result = call_llm("hello", temperature=0.9)
    assert result["prompt"] == "hello"
    assert result["temperature"] == 0.9


def test_pricing_registry_register_and_compute() -> None:
    """Register a model and verify cost calculation math."""
    from llm_toll import PricingRegistry

    registry = PricingRegistry()
    # $0.01 per input token, $0.03 per output token
    registry.register_model("test-model", 0.01, 0.03)

    cost = registry.get_cost("test-model", input_tokens=100, output_tokens=50)
    expected = 100 * 0.01 + 50 * 0.03  # 1.0 + 1.5 = 2.5
    assert cost == approx(expected)


def test_pricing_registry_unknown_model_returns_zero() -> None:
    """Unknown model should return 0.0 cost, not crash."""
    from llm_toll import PricingMatrixOutdatedWarning, PricingRegistry

    registry = PricingRegistry()
    with pytest.warns(PricingMatrixOutdatedWarning):
        cost = registry.get_cost("unknown-model", input_tokens=100, output_tokens=50)
    assert cost == 0.0


def test_pricing_registry_override_model() -> None:
    """Registering the same model twice should override pricing."""
    from llm_toll import PricingRegistry

    registry = PricingRegistry()
    registry.register_model("gpt-4o", 0.01, 0.03)
    registry.register_model("gpt-4o", 0.005, 0.015)

    cost = registry.get_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
    expected = 1000 * 0.005 + 1000 * 0.015
    assert cost == approx(expected)


def test_usage_store_creation_with_tmp_path(tmp_db_path: str) -> None:
    """Creating a UsageStore with a tmp path should not crash."""
    from llm_toll import UsageStore

    store = UsageStore(db_path=tmp_db_path)
    assert store is not None


def test_usage_store_creation_with_default_path() -> None:
    """Creating a UsageStore with no path should not crash."""
    from llm_toll import UsageStore

    store = UsageStore()
    assert store is not None


def test_usage_store_log_and_get_cost(tmp_db_path: str) -> None:
    """log_usage persists cost and get_total_cost returns the accumulated value."""
    from llm_toll import UsageStore

    store = UsageStore(db_path=tmp_db_path)
    store.log_usage(
        project="test",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        cost=0.05,
    )
    total = store.get_total_cost("test")
    assert total == approx(0.05)
    store.close()


def test_rate_limiter_creation_and_check() -> None:
    """Creating a RateLimiter and calling check() should not crash."""
    from llm_toll import RateLimiter

    limiter = RateLimiter(rpm=60, tpm=100000)
    assert limiter is not None
    limiter.check()
    limiter.record(tokens=500)


def test_rate_limiter_with_no_limits() -> None:
    """RateLimiter with no limits should accept any check."""
    from llm_toll import RateLimiter

    limiter = RateLimiter()
    limiter.check()
    limiter.record(tokens=999999)


def test_cost_reporter_report_call() -> None:
    """CostReporter.report_call should not crash."""
    from llm_toll import CostReporter

    reporter = CostReporter()
    reporter.report_call(
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        cost=0.05,
    )


def test_cost_reporter_report_session() -> None:
    """CostReporter.report_session should not crash."""
    from llm_toll import CostReporter

    reporter = CostReporter()
    reporter.report_session()


def test_cost_reporter_session_cost_starts_at_zero() -> None:
    """CostReporter._session_cost should start at 0.0."""
    from llm_toll import CostReporter

    reporter = CostReporter()
    assert reporter._session_cost == 0.0


def test_auto_detect_usage_returns_none_for_unknown() -> None:
    """auto_detect_usage should return None for an unrecognized object."""
    from llm_toll.parsers import auto_detect_usage

    result = auto_detect_usage({"random": "dict"})
    assert result is None


def test_auto_detect_usage_returns_none_for_string() -> None:
    """auto_detect_usage should return None for a plain string."""
    from llm_toll.parsers import auto_detect_usage

    result = auto_detect_usage("just a string")
    assert result is None


def test_auto_detect_usage_returns_none_for_none() -> None:
    """auto_detect_usage should return None for None input."""
    from llm_toll.parsers import auto_detect_usage

    result = auto_detect_usage(None)
    assert result is None


def test_individual_parsers_return_none() -> None:
    """Each individual parser stub should return None for any input."""
    from llm_toll.parsers.anthropic import parse_anthropic_response
    from llm_toll.parsers.gemini import parse_gemini_response
    from llm_toll.parsers.openai import parse_openai_response

    mock_response = {"model": "test", "usage": {"tokens": 100}}
    assert parse_openai_response(mock_response) is None
    assert parse_anthropic_response(mock_response) is None
    assert parse_gemini_response(mock_response) is None


def test_no_import_cycles() -> None:
    """Verify the full decorator + registry + parsers wiring has no import cycles.

    Force-reimport all modules to detect circular dependencies.
    """
    modules_to_check = [
        "llm_toll",
        "llm_toll.decorator",
        "llm_toll.exceptions",
        "llm_toll.pricing",
        "llm_toll.store",
        "llm_toll.rate_limiter",
        "llm_toll.reporter",
        "llm_toll.parsers",
        "llm_toll.parsers.openai",
        "llm_toll.parsers.anthropic",
        "llm_toll.parsers.gemini",
    ]

    # Remove all llm_toll modules from cache
    to_remove = [key for key in sys.modules if key.startswith("llm_toll")]
    for key in to_remove:
        del sys.modules[key]

    # Re-import everything — will raise ImportError on circular deps
    for mod in modules_to_check:
        importlib.import_module(mod)


def test_full_wiring_decorator_with_registry_and_store(tmp_db_path: str) -> None:
    """End-to-end: decorator + registry + store + reporter all instantiate together."""
    from llm_toll import CostReporter, PricingRegistry, UsageStore, track_costs

    registry = PricingRegistry()
    registry.register_model("gpt-4o", 0.000005, 0.000015)

    store = UsageStore(db_path=tmp_db_path)
    reporter = CostReporter()

    @track_costs(project="e2e-test", model="gpt-4o")
    def call_llm(prompt: str) -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25},
        }

    result = call_llm("Test prompt")
    assert result["model"] == "gpt-4o"

    cost = registry.get_cost("gpt-4o", input_tokens=50, output_tokens=25)
    assert cost > 0

    store.log_usage("e2e-test", "gpt-4o", 50, 25, cost)
    assert store.get_total_cost("e2e-test") == approx(cost)
    reporter.report_call("gpt-4o", 50, 25, cost)
    reporter.report_session()
    store.close()


def test_default_registry_importable() -> None:
    """``from llm_toll import default_registry`` works and has models loaded."""
    from llm_toll import default_registry

    assert default_registry is not None
    assert len(default_registry.list_models()) > 0


def test_default_registry_has_builtin_models() -> None:
    """default_registry ships with pricing for well-known models."""
    from llm_toll import default_registry

    assert default_registry.has_model("gpt-4o") is True


def test_default_registry_cost_calculation() -> None:
    """default_registry.get_cost returns the expected value for builtin pricing."""
    from llm_toll import default_registry

    # gpt-4o builtin: input 2.5e-06, output 10.0e-06
    cost = default_registry.get_cost("gpt-4o", 1000, 500)
    expected = 1000 * 2.5e-06 + 500 * 10.0e-06  # 0.0025 + 0.005 = 0.0075
    assert cost == approx(expected)


def test_pricing_registry_with_decorator() -> None:
    """A separate PricingRegistry can compute cost for the model the decorator tracks."""
    from llm_toll import PricingRegistry, track_costs

    registry = PricingRegistry()
    registry.register_model("gpt-4o", 2.5e-06, 10.0e-06)

    @track_costs(project="reg-test", model="gpt-4o")
    def call_llm() -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hi"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100},
        }

    result = call_llm()
    assert result["model"] == "gpt-4o"

    cost = registry.get_cost("gpt-4o", input_tokens=200, output_tokens=100)
    expected = 200 * 2.5e-06 + 100 * 10.0e-06
    assert cost == approx(expected)
    assert cost > 0


def test_prefix_matching_end_to_end() -> None:
    """Versioned model name resolves via prefix matching and returns non-zero cost."""
    from llm_toll import PricingRegistry

    registry = PricingRegistry()
    cost = registry.get_cost("gpt-4o-2024-08-06", 1000, 500)
    assert cost > 0


def test_unknown_model_warning_integration() -> None:
    """Querying a totally unknown model emits PricingMatrixOutdatedWarning."""
    from llm_toll import PricingMatrixOutdatedWarning, PricingRegistry

    registry = PricingRegistry()

    with pytest.warns(PricingMatrixOutdatedWarning):
        cost = registry.get_cost("totally-unknown-model-xyz", 100, 50)

    assert cost == 0.0


def test_fallback_pricing_end_to_end() -> None:
    """Registry with fallback pricing uses it for unknown models."""
    from llm_toll import PricingRegistry

    registry = PricingRegistry()
    registry.set_fallback_pricing(1e-05, 2e-05)

    cost = registry.get_cost("my-custom-model", 1000, 500)
    expected = 1000 * 1e-05 + 500 * 2e-05  # 0.01 + 0.01 = 0.02
    assert cost == approx(expected)
    assert cost > 0


# ---------------------------------------------------------------------------
# UsageStore integration tests
# ---------------------------------------------------------------------------


def test_store_persists_across_instances(tmp_db_path: str) -> None:
    """Data written by one UsageStore instance is visible to a new instance at the same path."""
    from llm_toll import UsageStore

    store1 = UsageStore(db_path=tmp_db_path)
    store1.log_usage(
        project="persist", model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.123
    )
    store1.close()

    store2 = UsageStore(db_path=tmp_db_path)
    total = store2.get_total_cost("persist")
    assert total == approx(0.123)
    store2.close()


def test_store_with_pricing_registry(tmp_db_path: str) -> None:
    """Cost computed by PricingRegistry matches what UsageStore reports after logging."""
    from llm_toll import PricingRegistry, UsageStore

    registry = PricingRegistry()
    registry.register_model("gpt-4o", 2.5e-06, 10.0e-06)

    cost = registry.get_cost("gpt-4o", input_tokens=500, output_tokens=200)

    store = UsageStore(db_path=tmp_db_path)
    store.log_usage(
        project="pricing", model="gpt-4o", input_tokens=500, output_tokens=200, cost=cost
    )

    assert store.get_total_cost("pricing") == approx(cost)
    assert cost == approx(500 * 2.5e-06 + 200 * 10.0e-06)
    store.close()


def test_store_multiple_models_same_project(tmp_db_path: str) -> None:
    """Logging calls with different models to the same project accumulates total cost."""
    from llm_toll import UsageStore

    store = UsageStore(db_path=tmp_db_path)
    store.log_usage(project="multi", model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.01)
    store.log_usage(
        project="multi",
        model="claude-sonnet-4-20250514",
        input_tokens=200,
        output_tokens=100,
        cost=0.02,
    )
    store.log_usage(
        project="multi", model="gemini-1.5-pro", input_tokens=300, output_tokens=150, cost=0.03
    )

    total = store.get_total_cost("multi")
    assert total == approx(0.01 + 0.02 + 0.03)
    store.close()


def test_store_get_usage_logs_end_to_end(tmp_db_path: str) -> None:
    """get_usage_logs returns entries with all expected fields populated."""
    from llm_toll import UsageStore

    store = UsageStore(db_path=tmp_db_path)
    store.log_usage(project="logs", model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.005)
    store.log_usage(
        project="logs",
        model="claude-sonnet-4-20250514",
        input_tokens=200,
        output_tokens=100,
        cost=0.010,
    )

    logs = store.get_usage_logs("logs")
    assert len(logs) == 2

    expected_keys = {
        "id",
        "project",
        "model",
        "input_tokens",
        "output_tokens",
        "cost",
        "created_at",
    }
    for entry in logs:
        assert set(entry.keys()) == expected_keys
        assert entry["project"] == "logs"
        assert isinstance(entry["id"], int)
        assert isinstance(entry["input_tokens"], int)
        assert isinstance(entry["output_tokens"], int)
        assert isinstance(entry["cost"], float)
        assert isinstance(entry["created_at"], str)
        assert len(entry["created_at"]) > 0

    # Most recent first
    models = [e["model"] for e in logs]
    assert models[0] == "claude-sonnet-4-20250514"
    assert models[1] == "gpt-4o"
    store.close()


def test_store_reset_preserves_logs(tmp_db_path: str) -> None:
    """Resetting a budget zeroes total_cost but does not delete usage log entries."""
    from llm_toll import UsageStore

    store = UsageStore(db_path=tmp_db_path)
    store.log_usage(project="reset", model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.05)
    store.log_usage(
        project="reset", model="gpt-4o", input_tokens=200, output_tokens=100, cost=0.10
    )

    assert store.get_total_cost("reset") == approx(0.15)

    store.reset_budget("reset")

    assert store.get_total_cost("reset") == approx(0.0)

    logs = store.get_usage_logs("reset")
    assert len(logs) == 2
    store.close()


def test_store_default_path_resolves() -> None:
    """UsageStore() without arguments resolves _db_path to ~/.llm_toll.db."""
    from pathlib import Path

    from llm_toll import UsageStore

    store = UsageStore()
    expected = str(Path.home() / ".llm_toll.db")
    assert store._db_path == expected


# ---------------------------------------------------------------------------
# Integration tests: decorator + store + pricing pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_extract_usage(tmp_db_path: str) -> None:
    """Decorator with extract_usage logs cost correctly and returns response unchanged."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        original_response = {"answer": "Hello, world!"}

        @track_costs(
            project="pipeline-test",
            extract_usage=lambda resp: ("gpt-4o", 1000, 500),
        )
        def call_llm() -> dict[str, Any]:
            return original_response

        result = call_llm()

        # Response returned unchanged
        assert result is original_response
        assert result == {"answer": "Hello, world!"}

        # Cost matches registry calculation
        expected_cost = default_registry.get_cost("gpt-4o", 1000, 500)
        assert store.get_total_cost("pipeline-test") == approx(expected_cost)

        # Exactly 1 log entry with correct fields
        logs = store.get_usage_logs("pipeline-test")
        assert len(logs) == 1
        entry = logs[0]
        assert entry["project"] == "pipeline-test"
        assert entry["model"] == "gpt-4o"
        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 500
        assert entry["cost"] == approx(expected_cost)
    finally:
        store.close()
        set_store(None)


def test_budget_enforcement_across_calls(tmp_db_path: str) -> None:
    """BudgetExceededError is raised when accumulated cost reaches max_budget."""
    from llm_toll import (
        BudgetExceededError,
        UsageStore,
        default_registry,
        set_store,
        track_costs,
    )

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        per_call_cost = default_registry.get_cost("gpt-4o", 1000, 500)
        max_budget = 0.05

        @track_costs(
            project="budget-test",
            max_budget=max_budget,
            extract_usage=lambda resp: ("gpt-4o", 1000, 500),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        call_count = 0
        with pytest.raises(BudgetExceededError):
            for _ in range(1000):  # upper bound to prevent infinite loop
                call_llm()
                call_count += 1

        # At least one call succeeded before the budget was exceeded
        assert call_count >= 1

        # Total cost should be under the budget (the exceeding call was blocked)
        total_cost = store.get_total_cost("budget-test")
        assert total_cost <= max_budget + per_call_cost  # last successful call may push near/over
        assert total_cost == approx(call_count * per_call_cost)
    finally:
        store.close()
        set_store(None)


def test_decorator_with_pricing_registry(tmp_db_path: str) -> None:
    """Cost logged via decorator matches default_registry.get_cost exactly."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        model = "gpt-4o"
        input_tokens = 2000
        output_tokens = 750

        @track_costs(
            project="pricing-test",
            extract_usage=lambda resp: (model, input_tokens, output_tokens),
        )
        def call_llm() -> dict[str, Any]:
            return {"content": "response"}

        call_llm()

        expected_cost = default_registry.get_cost(model, input_tokens, output_tokens)
        logs = store.get_usage_logs("pricing-test")
        assert len(logs) == 1
        assert logs[0]["cost"] == approx(expected_cost)
        assert store.get_total_cost("pricing-test") == approx(expected_cost)
    finally:
        store.close()
        set_store(None)


def test_decorator_bare_no_tracking_on_plain_response(tmp_db_path: str) -> None:
    """Bare @track_costs on a function returning a plain string logs nothing."""
    from llm_toll import UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs
        def call_llm() -> str:
            return "just a plain string"

        result = call_llm()
        assert result == "just a plain string"

        # auto_detect returns None for a plain string, no extract_usage provided
        logs = store.get_usage_logs("default")
        assert len(logs) == 0
        assert store.get_total_cost("default") == approx(0.0)
    finally:
        store.close()
        set_store(None)


def test_decorator_model_override_with_store(tmp_db_path: str) -> None:
    """Decorator model= parameter overrides the model returned by extract_usage."""
    from llm_toll import UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(
            project="override-test",
            model="gpt-4o",
            extract_usage=lambda resp: ("detected", 500, 200),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        call_llm()

        logs = store.get_usage_logs("override-test")
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4o"
        # Confirm it is NOT the detected model
        assert logs[0]["model"] != "detected"
    finally:
        store.close()
        set_store(None)


def test_budget_exceeded_message_contains_project(tmp_db_path: str) -> None:
    """BudgetExceededError message includes the project name."""
    from llm_toll import BudgetExceededError, UsageStore, default_registry, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        project_name = "my-important-project"
        per_call_cost = default_registry.get_cost("gpt-4o", 1000, 500)

        @track_costs(
            project=project_name,
            max_budget=per_call_cost * 1.5,
            extract_usage=lambda resp: ("gpt-4o", 1000, 500),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        # First call succeeds (cost fits within 1.5x budget)
        call_llm()

        # Second call raises: atomic post-call check rejects (2x > 1.5x)
        with pytest.raises(BudgetExceededError, match=project_name):
            call_llm()
    finally:
        store.close()
        set_store(None)


# ---------------------------------------------------------------------------
# Integration tests: auto-detection of SDK responses (no extract_usage)
# ---------------------------------------------------------------------------


class _MockOpenAIUsage:
    """Minimal mock of openai.types.CompletionUsage."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockOpenAIChatCompletion:
    """Minimal mock of openai.types.chat.ChatCompletion."""

    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.id = "chatcmpl-mock"
        self.model = model
        self.choices = [{"message": {"content": "Hello!"}}]
        self.usage = _MockOpenAIUsage(prompt_tokens, completion_tokens)


class _MockAnthropicUsage:
    """Minimal mock of anthropic Usage."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockAnthropicMessage:
    """Minimal mock of anthropic.types.Message."""

    def __init__(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.id = "msg-mock"
        self.model = model
        self.content = [{"type": "text", "text": "Hello!"}]
        self.stop_reason = "end_turn"
        self.usage = _MockAnthropicUsage(input_tokens, output_tokens)


def test_decorator_auto_detects_openai_response(tmp_db_path: str) -> None:
    """Decorator auto-detects an OpenAI ChatCompletion and logs model/cost correctly."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(project="openai-auto")
        def call_llm() -> _MockOpenAIChatCompletion:
            return _MockOpenAIChatCompletion("gpt-4o", prompt_tokens=100, completion_tokens=50)

        result = call_llm()

        # Response returned unchanged
        assert isinstance(result, _MockOpenAIChatCompletion)
        assert result.model == "gpt-4o"

        # Store has exactly one entry with correct model and cost
        logs = store.get_usage_logs("openai-auto")
        assert len(logs) == 1
        entry = logs[0]
        assert entry["model"] == "gpt-4o"
        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 50

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert entry["cost"] == approx(expected_cost)
        assert store.get_total_cost("openai-auto") == approx(expected_cost)
    finally:
        store.close()
        set_store(None)


def test_decorator_auto_detects_anthropic_response(tmp_db_path: str) -> None:
    """Decorator auto-detects an Anthropic Message and logs model/cost correctly."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(project="anthropic-auto")
        def call_llm() -> _MockAnthropicMessage:
            return _MockAnthropicMessage(
                "claude-sonnet-4-20250514", input_tokens=200, output_tokens=100
            )

        result = call_llm()

        assert isinstance(result, _MockAnthropicMessage)
        assert result.model == "claude-sonnet-4-20250514"

        logs = store.get_usage_logs("anthropic-auto")
        assert len(logs) == 1
        entry = logs[0]
        assert entry["model"] == "claude-sonnet-4-20250514"
        assert entry["input_tokens"] == 200
        assert entry["output_tokens"] == 100

        expected_cost = default_registry.get_cost("claude-sonnet-4-20250514", 200, 100)
        assert entry["cost"] == approx(expected_cost)
        assert store.get_total_cost("anthropic-auto") == approx(expected_cost)
    finally:
        store.close()
        set_store(None)


def test_decorator_openai_model_override(tmp_db_path: str) -> None:
    """Decorator model= overrides the model detected from an OpenAI response."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(project="model-override", model="gpt-4o")
        def call_llm() -> _MockOpenAIChatCompletion:
            return _MockOpenAIChatCompletion(
                "gpt-4o-2024-08-06", prompt_tokens=300, completion_tokens=150
            )

        result = call_llm()

        assert result.model == "gpt-4o-2024-08-06"

        logs = store.get_usage_logs("model-override")
        assert len(logs) == 1
        entry = logs[0]
        # Logged model should be the override, not the detected one
        assert entry["model"] == "gpt-4o"
        assert entry["input_tokens"] == 300
        assert entry["output_tokens"] == 150

        expected_cost = default_registry.get_cost("gpt-4o", 300, 150)
        assert entry["cost"] == approx(expected_cost)
    finally:
        store.close()
        set_store(None)


def test_decorator_auto_detect_with_budget(tmp_db_path: str) -> None:
    """Auto-detected OpenAI response works with max_budget enforcement."""
    from llm_toll import (
        BudgetExceededError,
        UsageStore,
        default_registry,
        set_store,
        track_costs,
    )

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        per_call_cost = default_registry.get_cost("gpt-4o", 1000, 500)
        # Budget allows exactly 2 calls: 2x fits within 2.5x, but the 3rd
        # call's post-call atomic check rejects it (2x + 1x = 3x > 2.5x).
        max_budget = per_call_cost * 2.5

        @track_costs(project="budget-auto", max_budget=max_budget)
        def call_llm() -> _MockOpenAIChatCompletion:
            return _MockOpenAIChatCompletion("gpt-4o", prompt_tokens=1000, completion_tokens=500)

        call_count = 0
        with pytest.raises(BudgetExceededError):
            for _ in range(100):
                call_llm()
                call_count += 1

        # 2 calls succeed; the 3rd call's atomic post-call check rejects it
        assert call_count == 2

        total_cost = store.get_total_cost("budget-auto")
        assert total_cost == approx(call_count * per_call_cost)
    finally:
        store.close()
        set_store(None)


# ---------------------------------------------------------------------------
# Integration tests: CostReporter + decorator pipeline
# ---------------------------------------------------------------------------


def test_decorator_reports_to_custom_reporter(tmp_db_path: str) -> None:
    """Decorator reports per-call cost to a custom CostReporter writing to StringIO."""
    from io import StringIO

    from llm_toll import CostReporter, UsageStore, set_reporter, set_store, track_costs

    buf = StringIO()
    reporter = CostReporter(file=buf)
    store = UsageStore(db_path=tmp_db_path)
    set_store(store)
    set_reporter(reporter)

    try:

        @track_costs(
            project="reporter-test",
            extract_usage=lambda resp: ("gpt-4o", 1000, 500),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        call_llm()

        output = buf.getvalue()
        assert "gpt-4o" in output
        assert "$" in output
    finally:
        store.close()
        set_store(None)
        set_reporter(None)


def test_reporter_session_after_multiple_decorated_calls(tmp_db_path: str) -> None:
    """After 3 decorated calls, report_session output contains '3 calls' and total cost."""
    from io import StringIO

    from llm_toll import CostReporter, UsageStore, set_reporter, set_store, track_costs

    buf = StringIO()
    reporter = CostReporter(file=buf)
    store = UsageStore(db_path=tmp_db_path)
    set_store(store)
    set_reporter(reporter)

    try:

        @track_costs(
            project="session-test",
            extract_usage=lambda resp: ("gpt-4o", 1000, 500),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        call_llm()
        call_llm()
        call_llm()

        reporter.report_session()

        output = buf.getvalue()
        assert "3 calls" in output
        assert "$" in output
    finally:
        store.close()
        set_store(None)
        set_reporter(None)


def test_reporter_disabled_no_output(tmp_db_path: str) -> None:
    """CostReporter(enabled=False) produces no output even after decorated calls."""
    from io import StringIO

    from llm_toll import CostReporter, UsageStore, set_reporter, set_store, track_costs

    buf = StringIO()
    reporter = CostReporter(enabled=False, file=buf)
    store = UsageStore(db_path=tmp_db_path)
    set_store(store)
    set_reporter(reporter)

    try:

        @track_costs(
            project="disabled-reporter-test",
            extract_usage=lambda resp: ("gpt-4o", 1000, 500),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        call_llm()
        call_llm()
        reporter.report_session()

        output = buf.getvalue()
        assert output == ""
    finally:
        store.close()
        set_store(None)
        set_reporter(None)


def test_concurrent_decorated_calls_respect_budget(tmp_db_path: str) -> None:
    """Concurrent threads calling a budgeted decorated function never exceed max_budget."""
    import threading

    from llm_toll import (
        BudgetExceededError,
        UsageStore,
        default_registry,
        set_store,
        track_costs,
    )

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        model = "gpt-4o"
        input_tokens = 1000
        output_tokens = 500
        per_call_cost = default_registry.get_cost(model, input_tokens, output_tokens)
        # Allow several calls but not unlimited
        max_budget = per_call_cost * 10

        @track_costs(
            project="concurrent-test",
            max_budget=max_budget,
            extract_usage=lambda resp: (model, input_tokens, output_tokens),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        def worker() -> None:
            for _ in range(50):
                try:
                    call_llm()
                except BudgetExceededError:
                    break

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_cost = store.get_total_cost("concurrent-test")
        # Allow a tiny float tolerance for accumulated rounding
        assert total_cost <= max_budget + 1e-9
    finally:
        store.close()
        set_store(None)


# ---------------------------------------------------------------------------
# Streaming integration tests
# ---------------------------------------------------------------------------


def test_streaming_openai_end_to_end(tmp_db_path: str) -> None:
    """OpenAI streaming call through the full decorator pipeline logs cost."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    # -- Mock OpenAI chunk objects (duck-typed) --

    class _Usage:
        def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Delta:
        def __init__(self, content: str | None = None) -> None:
            self.content = content

    class _Choice:
        def __init__(self, delta: _Delta) -> None:
            self.delta = delta

    class _OpenAIChunk:
        def __init__(
            self,
            model: str,
            choices: list[_Choice],
            usage: _Usage | None = None,
        ) -> None:
            self.model = model
            self.choices = choices
            self.usage = usage

    input_tokens = 15
    output_tokens = 10
    chunks = [
        _OpenAIChunk("gpt-4o", [_Choice(_Delta("Hello"))]),
        _OpenAIChunk("gpt-4o", [_Choice(_Delta(" world"))]),
        # Final chunk carries usage (stream_options={"include_usage": True})
        _OpenAIChunk("gpt-4o", [], _Usage(input_tokens, output_tokens)),
    ]

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(project="stream-openai-e2e")
        def call_openai_stream() -> Any:
            yield from chunks

        result = call_openai_stream()
        # Consume the stream
        consumed = list(result)
        assert len(consumed) == 3

        expected_cost = default_registry.get_cost("gpt-4o", input_tokens, output_tokens)
        total_cost = store.get_total_cost("stream-openai-e2e")
        assert total_cost == approx(expected_cost)
        assert total_cost > 0.0
    finally:
        store.close()
        set_store(None)


def test_streaming_anthropic_end_to_end(tmp_db_path: str) -> None:
    """Anthropic streaming events through the full decorator pipeline log cost."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    # -- Mock Anthropic streaming event objects (duck-typed) --

    class _AnthropicUsageStart:
        def __init__(self, input_tokens: int) -> None:
            self.input_tokens = input_tokens

    class _AnthropicUsageEnd:
        def __init__(self, output_tokens: int) -> None:
            self.output_tokens = output_tokens

    class _AnthropicMessage:
        def __init__(self, model: str, usage: _AnthropicUsageStart) -> None:
            self.model = model
            self.usage = usage

    class _AnthropicEvent:
        def __init__(self, type: str, **kwargs: Any) -> None:
            self.type = type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _ContentDelta:
        def __init__(self, text: str) -> None:
            self.text = text

    input_tokens = 20
    output_tokens = 12
    events = [
        _AnthropicEvent(
            "message_start",
            message=_AnthropicMessage(
                "claude-sonnet-4-20250514", _AnthropicUsageStart(input_tokens)
            ),
        ),
        _AnthropicEvent("content_block_delta", delta=_ContentDelta("Hello")),
        _AnthropicEvent("content_block_delta", delta=_ContentDelta(" there")),
        _AnthropicEvent("message_delta", usage=_AnthropicUsageEnd(output_tokens)),
        _AnthropicEvent("message_stop"),
    ]

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(project="stream-anthropic-e2e")
        def call_anthropic_stream() -> Any:
            yield from events

        result = call_anthropic_stream()
        consumed = list(result)
        assert len(consumed) == 5

        expected_cost = default_registry.get_cost(
            "claude-sonnet-4-20250514", input_tokens, output_tokens
        )
        total_cost = store.get_total_cost("stream-anthropic-e2e")
        assert total_cost == approx(expected_cost)
        assert total_cost > 0.0
    finally:
        store.close()
        set_store(None)


def test_streaming_with_budget_enforcement(tmp_db_path: str) -> None:
    """Streaming call with max_budget enforces budget after stream is consumed."""
    from llm_toll import UsageStore, default_registry, set_store, track_costs

    # -- Mock OpenAI chunks (reuse duck-typing pattern) --

    class _Usage:
        def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Delta:
        def __init__(self, content: str | None = None) -> None:
            self.content = content

    class _Choice:
        def __init__(self, delta: _Delta) -> None:
            self.delta = delta

    class _OpenAIChunk:
        def __init__(
            self,
            model: str,
            choices: list[_Choice],
            usage: _Usage | None = None,
        ) -> None:
            self.model = model
            self.choices = choices
            self.usage = usage

    input_tokens = 500
    output_tokens = 300
    per_call_cost = default_registry.get_cost("gpt-4o", input_tokens, output_tokens)
    # Budget allows exactly 2 calls (with a tiny margin)
    max_budget = per_call_cost * 2 + 1e-12

    def make_chunks() -> list[_OpenAIChunk]:
        return [
            _OpenAIChunk("gpt-4o", [_Choice(_Delta("response"))]),
            _OpenAIChunk("gpt-4o", [], _Usage(input_tokens, output_tokens)),
        ]

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(project="stream-budget-e2e", max_budget=max_budget)
        def call_llm_stream() -> Any:
            yield from make_chunks()

        # First call — should succeed and log cost
        list(call_llm_stream())
        assert store.get_total_cost("stream-budget-e2e") == approx(per_call_cost)

        # Second call — should also succeed
        list(call_llm_stream())
        assert store.get_total_cost("stream-budget-e2e") == approx(per_call_cost * 2)

        # Third call — budget exceeded; streaming still works (chunks yielded)
        # and cost IS logged (to keep totals accurate), with a warning
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            list(call_llm_stream())
        total_after_third = store.get_total_cost("stream-budget-e2e")
        # Cost is logged even when over budget (stream was already consumed)
        assert total_after_third == approx(per_call_cost * 3)
    finally:
        store.close()
        set_store(None)


# ---------------------------------------------------------------------------
# Rate limiter enforcement through the decorator
# ---------------------------------------------------------------------------


def test_decorator_rpm_enforcement(tmp_db_path: str) -> None:
    """rate_limit=3 allows 3 calls, then the 4th raises LocalRateLimitError(rpm)."""
    from llm_toll import LocalRateLimitError, UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(
            project="rpm-test",
            rate_limit=3,
            extract_usage=lambda _resp: ("gpt-4o", 10, 5),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        # First 3 calls should succeed
        for _ in range(3):
            assert call_llm() == {"result": "ok"}

        # 4th call should be blocked by the RPM limiter
        with pytest.raises(LocalRateLimitError) as exc_info:
            call_llm()

        err = exc_info.value
        assert err.limit_type == "rpm"
        assert err.retry_after is not None
        assert err.retry_after > 0

    finally:
        store.close()
        set_store(None)


def test_decorator_tpm_enforcement(tmp_db_path: str) -> None:
    """tpm_limit=500 allows 2 calls of 200 tokens, then the 3rd raises LocalRateLimitError(tpm)."""
    from llm_toll import LocalRateLimitError, UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:
        # Each call produces 200 total tokens (100 in + 100 out).
        # The rate limiter checks *before* the call, so after 2 successful
        # calls 400 tokens are recorded.  With tpm_limit=400 the 3rd
        # pre-call check sees 400 >= 400 and blocks it.
        @track_costs(
            project="tpm-test",
            tpm_limit=400,
            extract_usage=lambda _resp: ("gpt-4o", 100, 100),  # 200 total per call
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        # First 2 calls: 200 + 200 = 400 tokens recorded
        for _ in range(2):
            assert call_llm() == {"result": "ok"}

        # 3rd call — pre-call check sees 400 >= 400, raises LocalRateLimitError
        with pytest.raises(LocalRateLimitError) as exc_info:
            call_llm()

        err = exc_info.value
        assert err.limit_type == "tpm"
        assert err.retry_after is not None
        assert err.retry_after > 0

    finally:
        store.close()
        set_store(None)


def test_decorator_no_rate_limit_unlimited(tmp_db_path: str) -> None:
    """Without rate_limit/tpm_limit, no LocalRateLimitError is raised even after many calls."""
    from llm_toll import LocalRateLimitError, UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(
            project="unlimited-test",
            extract_usage=lambda _resp: ("gpt-4o", 500, 500),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        # Many calls with high token counts — should never raise LocalRateLimitError
        for _ in range(50):
            try:
                result = call_llm()
                assert result == {"result": "ok"}
            except LocalRateLimitError:
                pytest.fail("LocalRateLimitError raised without rate limits configured")

    finally:
        store.close()
        set_store(None)


def test_local_ollama_decorator_full_flow(tmp_db_path: str) -> None:
    """Local/Ollama models: zero cost, usage logged, rate limiting enforced."""
    from llm_toll import LocalRateLimitError, UsageStore, set_store, track_costs

    store = UsageStore(db_path=tmp_db_path)
    set_store(store)

    try:

        @track_costs(
            project="ollama-test",
            model="ollama/llama3",
            rate_limit=5,
            extract_usage=lambda _resp: ("ollama/llama3", 100, 50),
        )
        def call_llm() -> dict[str, Any]:
            return {"result": "ok"}

        # First call succeeds and is logged
        result = call_llm()
        assert result == {"result": "ok"}

        # Verify cost is $0.0 in store
        total_cost = store.get_total_cost("ollama-test")
        assert total_cost == 0.0

        # Verify usage was logged with correct tokens
        logs = store.get_usage_logs(project="ollama-test")
        assert len(logs) == 1
        assert logs[0]["input_tokens"] == 100
        assert logs[0]["output_tokens"] == 50
        assert logs[0]["cost"] == 0.0

        # Make 4 more calls to reach the rate limit of 5
        for _ in range(4):
            call_llm()

        # 6th call should raise LocalRateLimitError
        with pytest.raises(LocalRateLimitError) as exc_info:
            call_llm()

        err = exc_info.value
        assert err.limit_type == "rpm"
        assert err.retry_after is not None
        assert err.retry_after > 0

    finally:
        store.close()
        set_store(None)
