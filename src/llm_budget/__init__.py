"""llm_budget — Lightweight decorator to track LLM API costs and enforce budgets."""

from llm_budget.decorator import set_reporter, set_store, track_costs
from llm_budget.exceptions import (
    BudgetExceededError,
    LocalRateLimitError,
    PricingMatrixOutdatedWarning,
)
from llm_budget.pricing import PricingRegistry, default_registry
from llm_budget.rate_limiter import RateLimiter
from llm_budget.reporter import CostReporter
from llm_budget.store import UsageStore

__version__ = "0.1.0"

__all__ = [
    "BudgetExceededError",
    "CostReporter",
    "LocalRateLimitError",
    "PricingMatrixOutdatedWarning",
    "PricingRegistry",
    "RateLimiter",
    "UsageStore",
    "__version__",
    "default_registry",
    "set_reporter",
    "set_store",
    "track_costs",
]
