"""Unit tests for PricingRegistry."""

from __future__ import annotations

from pytest import approx

from llm_budget.pricing import PricingRegistry


class TestPricingRegistry:
    """Tests for PricingRegistry cost calculation and model registration."""

    def test_register_model_then_get_cost(self) -> None:
        registry = PricingRegistry()
        registry.register_model("gpt-4o", 0.005, 0.015)
        cost = registry.get_cost("gpt-4o", input_tokens=100, output_tokens=50)
        assert cost == approx(100 * 0.005 + 50 * 0.015)

    def test_get_cost_unknown_model_returns_zero(self) -> None:
        registry = PricingRegistry()
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
