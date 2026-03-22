"""Smoke tests to validate package imports and exception hierarchy."""

from __future__ import annotations


def test_package_version() -> None:
    from llm_toll import __version__

    assert __version__ == "0.6.0"


def test_public_api_exports() -> None:
    import llm_toll

    assert hasattr(llm_toll, "track_costs")
    assert hasattr(llm_toll, "BudgetExceededError")
    assert hasattr(llm_toll, "LocalRateLimitError")
    assert hasattr(llm_toll, "PricingMatrixOutdatedWarning")
    assert hasattr(llm_toll, "PricingRegistry")
    assert hasattr(llm_toll, "UsageStore")
    assert hasattr(llm_toll, "CostReporter")
    assert hasattr(llm_toll, "RateLimiter")


def test_budget_exceeded_error_is_exception() -> None:
    from llm_toll import BudgetExceededError

    assert issubclass(BudgetExceededError, Exception)


def test_local_rate_limit_error_is_exception() -> None:
    from llm_toll import LocalRateLimitError

    assert issubclass(LocalRateLimitError, Exception)


def test_pricing_matrix_outdated_warning_is_warning() -> None:
    from llm_toll import PricingMatrixOutdatedWarning

    assert issubclass(PricingMatrixOutdatedWarning, UserWarning)


def test_track_costs_bare_decorator() -> None:
    from llm_toll import track_costs

    @track_costs
    def my_func() -> str:
        return "hello"

    assert my_func() == "hello"


def test_track_costs_with_args() -> None:
    from llm_toll import track_costs

    @track_costs(project="test", max_budget=10.0)
    def my_func() -> str:
        return "hello"

    assert my_func() == "hello"
