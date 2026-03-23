"""llm_toll — Lightweight decorator to track LLM API costs and enforce budgets."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from llm_toll.decorator import set_reporter, set_store, track_costs
from llm_toll.exceptions import (
    BudgetExceededError,
    LocalRateLimitError,
    PricingMatrixOutdatedWarning,
)
from llm_toll.integrations.langchain import LangChainCallback
from llm_toll.integrations.litellm import LiteLLMCallback
from llm_toll.pricing import PricingRegistry, default_registry
from llm_toll.rate_limiter import RateLimiter
from llm_toll.remote_pricing import update_pricing
from llm_toll.reporter import CostReporter
from llm_toll.store import BaseStore, SQLiteStore, UsageStore, create_store

try:
    __version__: str = _pkg_version("llm-toll")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
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
