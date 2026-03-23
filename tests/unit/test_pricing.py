"""Unit tests for PricingRegistry."""

from __future__ import annotations

import threading
import warnings

import pytest
from pytest import approx

from llm_toll.exceptions import PricingMatrixOutdatedWarning
from llm_toll.pricing import PricingRegistry


class TestPricingRegistry:
    """Tests for PricingRegistry cost calculation and model registration."""

    def test_register_model_then_get_cost(self) -> None:
        registry = PricingRegistry()
        registry.register_model("gpt-4o", 0.005, 0.015)
        cost = registry.get_cost("gpt-4o", input_tokens=100, output_tokens=50)
        assert cost == approx(100 * 0.005 + 50 * 0.015)

    def test_get_cost_unknown_model_returns_zero(self) -> None:
        registry = PricingRegistry()
        with pytest.warns(PricingMatrixOutdatedWarning):
            assert registry.get_cost("unknown-model", input_tokens=500, output_tokens=200) == 0.0

    def test_multiple_models_each_returns_correct_cost(self) -> None:
        registry = PricingRegistry()
        registry.register_model("model-a", 0.001, 0.002)
        registry.register_model("model-b", 0.010, 0.020)

        cost_a = registry.get_cost("model-a", input_tokens=100, output_tokens=100)
        cost_b = registry.get_cost("model-b", input_tokens=100, output_tokens=100)

        assert cost_a == approx(100 * 0.001 + 100 * 0.002)
        assert cost_b == approx(100 * 0.010 + 100 * 0.020)
        assert cost_a != approx(cost_b)

    def test_zero_tokens_returns_zero_cost(self) -> None:
        registry = PricingRegistry()
        registry.register_model("gpt-4o", 0.005, 0.015)
        assert registry.get_cost("gpt-4o", input_tokens=0, output_tokens=0) == 0.0


class TestPricingRegistryBuiltins:
    """Tests for built-in pricing, prefix matching, fallback, and thread safety."""

    def test_builtin_pricing_loaded_on_init(self) -> None:
        registry = PricingRegistry()
        cost = registry.get_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        assert cost > 0.0

    def test_all_openai_models_present(self) -> None:
        registry = PricingRegistry()
        openai_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o4-mini",
        ]
        for model in openai_models:
            assert registry.has_model(model), f"{model} missing from registry"

    def test_all_anthropic_models_present(self) -> None:
        registry = PricingRegistry()
        anthropic_models = [
            "claude-sonnet-4-20250514",
            "claude-3.5-sonnet",
            "claude-3-haiku",
            "claude-3-opus",
            "claude-3.5-haiku",
        ]
        for model in anthropic_models:
            assert registry.has_model(model), f"{model} missing from registry"

    def test_all_gemini_models_present(self) -> None:
        registry = PricingRegistry()
        gemini_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]
        for model in gemini_models:
            assert registry.has_model(model), f"{model} missing from registry"

    def test_register_model_overrides_builtin(self) -> None:
        registry = PricingRegistry()
        # Confirm builtin cost first
        original = registry.get_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        # Override with custom pricing
        registry.register_model("gpt-4o", 0.001, 0.002)
        cost = registry.get_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        assert cost == approx(1000 * 0.001 + 1000 * 0.002)
        assert cost != approx(original)

    def test_prefix_matching(self) -> None:
        registry = PricingRegistry()
        base_cost = registry.get_cost("gpt-4o", input_tokens=500, output_tokens=500)
        versioned_cost = registry.get_cost(
            "gpt-4o-2024-08-06", input_tokens=500, output_tokens=500
        )
        assert versioned_cost == approx(base_cost)

        base_claude = registry.get_cost("claude-3.5-sonnet", input_tokens=500, output_tokens=500)
        versioned_claude = registry.get_cost(
            "claude-3.5-sonnet-20241022", input_tokens=500, output_tokens=500
        )
        assert versioned_claude == approx(base_claude)

    def test_prefix_matching_longest_wins(self) -> None:
        registry = PricingRegistry()
        mini_cost = registry.get_cost("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
        versioned_mini_cost = registry.get_cost(
            "gpt-4o-mini-2024-07-18", input_tokens=1000, output_tokens=1000
        )
        assert versioned_mini_cost == approx(mini_cost)
        # Ensure it did NOT match the shorter "gpt-4o" prefix
        gpt4o_cost = registry.get_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        assert versioned_mini_cost != approx(gpt4o_cost)

    def test_prefix_match_cached(self) -> None:
        registry = PricingRegistry()
        versioned = "gpt-4o-2024-08-06"
        cost1 = registry.get_cost(versioned, input_tokens=100, output_tokens=100)
        cost2 = registry.get_cost(versioned, input_tokens=100, output_tokens=100)
        assert cost1 == approx(cost2)
        # After first lookup, the versioned model should be cached in the registry
        assert registry.has_model(versioned)

    def test_unknown_model_emits_warning(self) -> None:
        registry = PricingRegistry()
        with pytest.warns(PricingMatrixOutdatedWarning):
            cost = registry.get_cost("totally-unknown-model", input_tokens=100, output_tokens=100)
        assert cost == 0.0

    def test_unknown_model_warning_only_once(self) -> None:
        registry = PricingRegistry()
        with pytest.warns(PricingMatrixOutdatedWarning):
            registry.get_cost("totally-unknown-model", input_tokens=100, output_tokens=100)
        # Second call should NOT emit a warning (model is cached as 0,0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry.get_cost("totally-unknown-model", input_tokens=100, output_tokens=100)
        pricing_warnings = [
            w for w in caught if issubclass(w.category, PricingMatrixOutdatedWarning)
        ]
        assert len(pricing_warnings) == 0

    def test_fallback_pricing(self) -> None:
        registry = PricingRegistry()
        registry.set_fallback_pricing(0.01, 0.02)
        # Query an unknown model — should use fallback, no warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cost = registry.get_cost("some-future-model", input_tokens=100, output_tokens=50)
        pricing_warnings = [
            w for w in caught if issubclass(w.category, PricingMatrixOutdatedWarning)
        ]
        assert len(pricing_warnings) == 0
        assert cost == approx(100 * 0.01 + 50 * 0.02)

    def test_fallback_not_used_when_model_known(self) -> None:
        registry = PricingRegistry()
        registry.set_fallback_pricing(0.99, 0.99)
        # Known model should use its builtin pricing, not the fallback
        cost = registry.get_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        fallback_cost = 1000 * 0.99 + 1000 * 0.99
        assert cost != approx(fallback_cost)
        assert cost == approx(1000 * 2.5e-06 + 1000 * 10.0e-06)

    def test_list_models(self) -> None:
        registry = PricingRegistry()
        models = registry.list_models()
        assert models == sorted(models), "list_models should return a sorted list"
        # Should contain all builtins
        for name in ["gpt-4o", "claude-3-opus", "gemini-2.5-pro"]:
            assert name in models

    def test_has_model_true_for_builtin(self) -> None:
        registry = PricingRegistry()
        assert registry.has_model("gpt-4o") is True

    def test_has_model_false_for_unknown(self) -> None:
        registry = PricingRegistry()
        assert registry.has_model("unknown") is False

    def test_thread_safety(self) -> None:
        registry = PricingRegistry()
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                model_name = f"thread-model-{idx}"
                registry.register_model(model_name, 0.001 * idx, 0.002 * idx)
                registry.get_cost("gpt-4o", input_tokens=100, output_tokens=100)
                registry.get_cost(model_name, input_tokens=50, output_tokens=50)
                registry.get_cost(f"gpt-4o-thread-{idx}", input_tokens=10, output_tokens=10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread safety violated: {errors}"


class TestPricingValidation:
    """Tests for input validation in register_model and set_fallback_pricing."""

    def test_register_model_negative_input_cost_raises(self) -> None:
        registry = PricingRegistry()
        with pytest.raises(ValueError, match="input_cost_per_token must be non-negative"):
            registry.register_model("test-model", -0.001, 0.002)

    def test_register_model_negative_output_cost_raises(self) -> None:
        registry = PricingRegistry()
        with pytest.raises(ValueError, match="output_cost_per_token must be non-negative"):
            registry.register_model("test-model", 0.001, -0.002)

    def test_register_model_zero_cost_allowed(self) -> None:
        registry = PricingRegistry()
        registry.register_model("free-model", 0.0, 0.0)
        cost = registry.get_cost("free-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_set_fallback_negative_input_raises(self) -> None:
        registry = PricingRegistry()
        with pytest.raises(ValueError, match="input_cost_per_token must be non-negative"):
            registry.set_fallback_pricing(-0.01, 0.02)

    def test_set_fallback_negative_output_raises(self) -> None:
        registry = PricingRegistry()
        with pytest.raises(ValueError, match="output_cost_per_token must be non-negative"):
            registry.set_fallback_pricing(0.01, -0.02)

    def test_set_fallback_zero_allowed(self) -> None:
        registry = PricingRegistry()
        registry.set_fallback_pricing(0.0, 0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cost = registry.get_cost("any-unknown-model", input_tokens=100, output_tokens=50)
        pricing_warnings = [
            w for w in caught if issubclass(w.category, PricingMatrixOutdatedWarning)
        ]
        assert len(pricing_warnings) == 0
        assert cost == 0.0


class TestLocalOllamaPricing:
    """Tests for Local/Ollama zero-cost provider support."""

    def test_ollama_prefix_zero_cost(self) -> None:
        registry = PricingRegistry()
        cost = registry.get_cost("ollama/llama3", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_local_prefix_zero_cost(self) -> None:
        registry = PricingRegistry()
        cost = registry.get_cost("local/mistral", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_llama_cpp_prefix_zero_cost(self) -> None:
        registry = PricingRegistry()
        cost = registry.get_cost("llama.cpp/phi", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_ollama_no_warning(self) -> None:
        registry = PricingRegistry()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry.get_cost("ollama/llama3", input_tokens=1000, output_tokens=500)
        pricing_warnings = [
            w for w in caught if issubclass(w.category, PricingMatrixOutdatedWarning)
        ]
        assert len(pricing_warnings) == 0

    def test_ollama_model_cached_after_prefix_match(self) -> None:
        registry = PricingRegistry()
        assert registry.has_model("ollama/llama3") is False
        registry.get_cost("ollama/llama3", input_tokens=100, output_tokens=50)
        assert registry.has_model("ollama/llama3") is True


class TestBoundaryPrefixMatching:
    """Tests for boundary-aware prefix matching via _is_boundary_match."""

    def test_o3_does_not_match_o3000(self) -> None:
        """'o3' should NOT match 'o3000' — no boundary after 'o3'."""
        registry = PricingRegistry()
        with pytest.warns(PricingMatrixOutdatedWarning):
            cost = registry.get_cost("o3000", input_tokens=100, output_tokens=50)
        assert cost == 0.0

    def test_gpt4o_matches_gpt4o_dated(self) -> None:
        """'gpt-4o-2024-08-06' should match 'gpt-4o' (hyphen boundary)."""
        registry = PricingRegistry()
        base_cost = registry.get_cost("gpt-4o", input_tokens=100, output_tokens=50)
        versioned_cost = registry.get_cost("gpt-4o-2024-08-06", input_tokens=100, output_tokens=50)
        assert versioned_cost == approx(base_cost)
        assert versioned_cost > 0.0

    def test_ollama_prefix_still_works(self) -> None:
        """'ollama/llama3' should match 'ollama/' (slash-terminated prefix)."""
        registry = PricingRegistry()
        cost = registry.get_cost("ollama/llama3", input_tokens=100, output_tokens=50)
        assert cost == 0.0


class TestBoundedCache:
    """Tests for bounded dynamic cache in PricingRegistry."""

    def test_cache_does_not_grow_unbounded(self) -> None:
        """Querying many unknown models should not let _models grow without bound."""
        registry = PricingRegistry()
        builtin_count = len(registry._models)

        # Query 1100 unique unknown model names
        for i in range(1100):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PricingMatrixOutdatedWarning)
                registry.get_cost(f"unknown-model-{i}", input_tokens=1, output_tokens=1)

        # _models should be bounded: builtins + at most _MAX_DYNAMIC_CACHE entries + small margin
        max_allowed = builtin_count + PricingRegistry._MAX_DYNAMIC_CACHE + 10
        assert len(registry._models) <= max_allowed, (
            f"Cache grew to {len(registry._models)} entries, "
            f"expected at most {max_allowed} (builtin={builtin_count})"
        )


class TestCostRounding:
    """Tests that cost calculation rounds to prevent floating-point drift."""

    def test_get_cost_is_rounded(self) -> None:
        """Cost should be cleanly rounded, not have floating-point noise."""
        registry = PricingRegistry()
        # gpt-4o: input=2.5e-06, output=10.0e-06
        # 1000 * 2.5e-06 + 500 * 10.0e-06 = 0.0025 + 0.005 = 0.0075
        cost = registry.get_cost("gpt-4o", 1000, 500)
        assert cost == 0.0075  # exact, no floating-point noise

    def test_no_drift_after_many_accumulations(self) -> None:
        """Summing many rounded costs should not drift."""
        registry = PricingRegistry()
        single = registry.get_cost("gpt-4o", 100, 50)
        total = sum(registry.get_cost("gpt-4o", 100, 50) for _ in range(10000))
        expected = round(single * 10000, 10)
        assert total == pytest.approx(expected, abs=1e-9)
