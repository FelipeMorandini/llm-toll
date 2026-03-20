"""In-memory registry for per-model cost-per-token pricing."""

from __future__ import annotations


class PricingRegistry:
    """In-memory store of per-model cost-per-token pricing.

    Starts empty. Register pricing for models via :meth:`register_model`.
    Calling :meth:`get_cost` for an unregistered model returns ``0.0``.
    Built-in pricing data will be added in a future release.
    """

    def __init__(self) -> None:
        self._models: dict[str, tuple[float, float]] = {}

    def register_model(
        self,
        model: str,
        input_cost_per_token: float,
        output_cost_per_token: float,
    ) -> None:
        """Register or override pricing for a model."""
        self._models[model] = (input_cost_per_token, output_cost_per_token)

    def get_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate the cost for a given model and token counts."""
        if model not in self._models:
            return 0.0
        input_cost, output_cost = self._models[model]
        return input_tokens * input_cost + output_tokens * output_cost
