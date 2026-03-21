"""Custom exceptions for llm_budget."""

from __future__ import annotations


class BudgetExceededError(Exception):
    """Raised when cumulative cost exceeds the configured budget cap.

    Attributes:
        project: The project name that exceeded its budget.
        current_cost: The accumulated cost at the time of the error.
        max_budget: The configured budget cap.
    """

    project: str | None
    current_cost: float | None
    max_budget: float | None

    def __init__(
        self,
        message: str | None = None,
        *,
        project: str | None = None,
        current_cost: float | None = None,
        max_budget: float | None = None,
    ) -> None:
        self.project = project
        self.current_cost = current_cost
        self.max_budget = max_budget
        if (
            message is None
            and project is not None
            and current_cost is not None
            and max_budget is not None
        ):
            message = (
                f"Budget exceeded for project '{project}': "
                f"${current_cost:.4f} >= ${max_budget:.4f}"
            )
        super().__init__(message or "")


class LocalRateLimitError(Exception):
    """Raised when local RPM/TPM limit is breached before the API call is made.

    Attributes:
        limit_type: The type of limit breached (``"rpm"`` or ``"tpm"``).
        limit_value: The configured limit value.
        retry_after: Seconds until the next request is allowed, or ``None``.
    """

    limit_type: str | None
    limit_value: int | None
    retry_after: float | None

    def __init__(
        self,
        message: str | None = None,
        *,
        limit_type: str | None = None,
        limit_value: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.retry_after = retry_after
        if message is None and limit_type is not None and limit_value is not None:
            msg = f"Rate limit exceeded: {limit_value} {limit_type} limit reached."
            if retry_after is not None:
                msg += f" Retry after {retry_after:.1f}s."
            message = msg
        super().__init__(message or "")


class PricingMatrixOutdatedWarning(UserWarning):
    """Emitted when a model is not found in the pricing registry."""
