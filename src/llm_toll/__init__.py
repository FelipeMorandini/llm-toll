"""llm_toll — Lightweight decorator to track LLM API costs and enforce budgets."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

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

try:
    _v = _pkg_version("llm-toll")
    __version__: str = _v if isinstance(_v, str) else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"

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
