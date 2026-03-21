"""llm_toll — Lightweight decorator to track LLM API costs and enforce budgets."""

from llm_toll.decorator import set_reporter, set_store, track_costs
from llm_toll.exceptions import (
    BudgetExceededError,
    LocalRateLimitError,
    PricingMatrixOutdatedWarning,
)
from llm_toll.pricing import PricingRegistry, default_registry
from llm_toll.rate_limiter import RateLimiter
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore

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
