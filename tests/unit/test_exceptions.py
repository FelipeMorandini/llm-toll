"""Tests for custom exceptions in llm_toll."""

from __future__ import annotations

import warnings

import pytest

from llm_toll.exceptions import (
    BudgetExceededError,
    LocalRateLimitError,
    PricingMatrixOutdatedWarning,
)


class TestBudgetExceededError:
    """Tests for BudgetExceededError."""

    def test_structured_construction(self) -> None:
        err = BudgetExceededError(project="proj", current_cost=5.02, max_budget=5.0)
        assert err.project == "proj"
        assert err.current_cost == 5.02
        assert err.max_budget == 5.0
        msg = str(err)
        assert "proj" in msg
        assert "5.02" in msg
        assert "5.0" in msg

    def test_legacy_string_construction(self) -> None:
        err = BudgetExceededError("plain msg")
        assert str(err) == "plain msg"
        assert err.project is None
        assert err.current_cost is None
        assert err.max_budget is None

    def test_is_exception_subclass(self) -> None:
        assert issubclass(BudgetExceededError, Exception)

    def test_catchable_as_exception(self) -> None:
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError(project="proj", current_cost=5.02, max_budget=5.0)

    def test_message_format(self) -> None:
        err = BudgetExceededError(project="proj", current_cost=5.02, max_budget=5.0)
        assert str(err) == "Budget exceeded for project 'proj': $5.0200 >= $5.0000"

    def test_zero_budget(self) -> None:
        err = BudgetExceededError(project="x", current_cost=0.0, max_budget=0.0)
        assert err.project == "x"
        assert err.current_cost == 0.0
        assert err.max_budget == 0.0
        assert "x" in str(err)

    def test_partial_kwargs(self) -> None:
        err = BudgetExceededError(project="myproj")
        assert err.project == "myproj"
        assert err.current_cost is None
        assert err.max_budget is None
        assert str(err) == ""


class TestLocalRateLimitError:
    """Tests for LocalRateLimitError."""

    def test_structured_construction_rpm(self) -> None:
        err = LocalRateLimitError(limit_type="rpm", limit_value=60, retry_after=2.5)
        assert err.limit_type == "rpm"
        assert err.limit_value == 60
        assert err.retry_after == 2.5
        msg = str(err)
        assert "60" in msg
        assert "rpm" in msg
        assert "2.5" in msg

    def test_structured_construction_tpm(self) -> None:
        err = LocalRateLimitError(limit_type="tpm", limit_value=100000, retry_after=None)
        assert err.limit_type == "tpm"
        assert err.limit_value == 100000
        assert err.retry_after is None
        assert "Retry after" not in str(err)

    def test_legacy_string_construction(self) -> None:
        err = LocalRateLimitError("plain msg")
        assert str(err) == "plain msg"
        assert err.limit_type is None
        assert err.limit_value is None
        assert err.retry_after is None

    def test_is_exception_subclass(self) -> None:
        assert issubclass(LocalRateLimitError, Exception)

    def test_message_with_retry(self) -> None:
        err = LocalRateLimitError(limit_type="rpm", limit_value=60, retry_after=2.5)
        assert "Retry after 2.5s" in str(err)

    def test_message_without_retry(self) -> None:
        err = LocalRateLimitError(limit_type="rpm", limit_value=60, retry_after=None)
        assert "Retry after" not in str(err)


class TestPricingMatrixOutdatedWarning:
    """Tests for PricingMatrixOutdatedWarning."""

    def test_is_user_warning_subclass(self) -> None:
        assert issubclass(PricingMatrixOutdatedWarning, UserWarning)

    def test_can_be_raised_as_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "pricing may be outdated",
                PricingMatrixOutdatedWarning,
                stacklevel=1,
            )
            assert len(caught) == 1
            assert issubclass(caught[0].category, PricingMatrixOutdatedWarning)
            assert "pricing may be outdated" in str(caught[0].message)
