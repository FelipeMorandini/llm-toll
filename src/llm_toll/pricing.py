"""In-memory registry for per-model cost-per-token pricing."""

from __future__ import annotations

import threading
import warnings

from llm_toll.exceptions import PricingMatrixOutdatedWarning

# Per-token pricing (input_cost, output_cost). Last updated: 2025-05.
_BUILTIN_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.5e-06, 10.0e-06),
    "gpt-4o-mini": (1.5e-07, 6.0e-07),
    "gpt-4-turbo": (1.0e-05, 3.0e-05),
    "gpt-3.5-turbo": (5.0e-07, 1.5e-06),
    "o1": (1.5e-05, 6.0e-05),
    "o1-mini": (3.0e-06, 1.2e-05),
    "o3": (1.0e-05, 4.0e-05),
    "o3-mini": (1.1e-06, 4.4e-06),
    "o4-mini": (1.1e-06, 4.4e-06),
    # Anthropic
    "claude-sonnet-4-20250514": (3.0e-06, 1.5e-05),
    "claude-3.5-sonnet": (3.0e-06, 1.5e-05),
    "claude-3-haiku": (2.5e-07, 1.25e-06),
    "claude-3-opus": (1.5e-05, 7.5e-05),
    "claude-3.5-haiku": (8.0e-07, 4.0e-06),
    # Google Gemini
    "gemini-1.5-pro": (1.25e-06, 5.0e-06),
    "gemini-1.5-flash": (7.5e-08, 3.0e-07),
    "gemini-2.0-flash": (1.0e-07, 4.0e-07),
    "gemini-2.5-pro": (1.25e-06, 1.0e-05),
    "gemini-2.5-flash": (1.5e-07, 6.0e-07),
    # Local / Ollama — $0 cost, tracked for rate limiting and monitoring
    "ollama/": (0.0, 0.0),
    "local/": (0.0, 0.0),
    "llama.cpp/": (0.0, 0.0),
}


COST_ROUND_PLACES: int = 10
"""Number of decimal places to round costs to, preventing floating-point drift."""


def _calc(input_tokens: int, output_tokens: int, pricing: tuple[float, float]) -> float:
    """Compute cost and round to :data:`COST_ROUND_PLACES` decimal places."""
    return round(input_tokens * pricing[0] + output_tokens * pricing[1], COST_ROUND_PLACES)


class PricingRegistry:
    """In-memory store of per-model cost-per-token pricing.

    Pre-loaded with pricing for OpenAI, Anthropic, and Gemini models.
    Supports custom overrides via :meth:`register_model` and optional
    fallback pricing for unrecognized models via :meth:`set_fallback_pricing`.

    Model lookup uses exact match first, then longest-prefix match
    (e.g., ``"gpt-4o-2024-08-06"`` resolves to ``"gpt-4o"``).
    """

    _MAX_DYNAMIC_CACHE: int = 1000

    def __init__(self) -> None:
        self._models: dict[str, tuple[float, float]] = _BUILTIN_PRICING.copy()
        self._protected_keys: set[str] = set(_BUILTIN_PRICING)
        self._dynamic_count: int = 0
        self._fallback: tuple[float, float] | None = None
        self._lock = threading.Lock()

    def register_model(
        self,
        model: str,
        input_cost_per_token: float,
        output_cost_per_token: float,
    ) -> None:
        """Register or override pricing for a model."""
        if input_cost_per_token < 0:
            raise ValueError(
                f"input_cost_per_token must be non-negative, got {input_cost_per_token}"
            )
        if output_cost_per_token < 0:
            raise ValueError(
                f"output_cost_per_token must be non-negative, got {output_cost_per_token}"
            )
        with self._lock:
            self._models[model] = (input_cost_per_token, output_cost_per_token)
            self._protected_keys.add(model)

    def set_fallback_pricing(
        self,
        input_cost_per_token: float,
        output_cost_per_token: float,
    ) -> None:
        """Set fallback pricing used for any model not yet resolved.

        Note: models already queried (and cached as unknown) will not
        retroactively use the fallback. Set fallback before first use.
        """
        if input_cost_per_token < 0:
            raise ValueError(
                f"input_cost_per_token must be non-negative, got {input_cost_per_token}"
            )
        if output_cost_per_token < 0:
            raise ValueError(
                f"output_cost_per_token must be non-negative, got {output_cost_per_token}"
            )
        with self._lock:
            self._fallback = (input_cost_per_token, output_cost_per_token)

    def get_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate the cost for a given model and token counts.

        Lookup order:
        1. Exact match in registered models.
        2. Longest-prefix match (cached for subsequent calls).
        3. Fallback pricing if configured.
        4. Emits :class:`PricingMatrixOutdatedWarning` and returns ``0.0``.
        """
        pricing = self._models.get(model)
        if pricing is not None:
            return _calc(input_tokens, output_tokens, pricing)

        # Try prefix match — find the longest registered key that is a prefix of model
        pricing = self._resolve_prefix(model)
        if pricing is not None:
            return _calc(input_tokens, output_tokens, pricing)

        # Fallback or unknown — guard with lock for atomic check-and-cache
        with self._lock:
            # Re-check under lock (another thread may have resolved it)
            pricing = self._models.get(model)
            if pricing is not None:
                return _calc(input_tokens, output_tokens, pricing)

            if self._fallback is not None:
                self._cache_dynamic(model, self._fallback)
                return _calc(input_tokens, output_tokens, self._fallback)

            # Unknown model — warn once, then cache (0, 0)
            warnings.warn(
                f"Model '{model}' not found in pricing registry. Cost will be reported as $0.00.",
                PricingMatrixOutdatedWarning,
                stacklevel=2,
            )
            self._cache_dynamic(model, (0.0, 0.0))
        return 0.0

    def load_remote_pricing(self, models: dict[str, tuple[float, float]]) -> int:
        """Bulk-load pricing from a remote source.

        Merges *models* into the registry, overriding existing entries.
        Returns the number of models loaded.
        """
        count = 0
        for name, (inp, out) in models.items():
            self.register_model(name, inp, out)
            count += 1
        return count

    def has_model(self, model: str) -> bool:
        """Check if a model has pricing registered (exact match only)."""
        return model in self._models

    def list_models(self) -> list[str]:
        """Return all registered model names."""
        with self._lock:
            return sorted(self._models.keys())

    def _cache_dynamic(self, model: str, pricing: tuple[float, float]) -> None:
        """Cache a dynamically resolved entry, evicting if over limit.

        Must be called while ``self._lock`` is held.
        """
        if model in self._models:
            return
        if self._dynamic_count >= self._MAX_DYNAMIC_CACHE:
            # Evict the first dynamic entry
            for key in list(self._models):
                if key not in self._protected_keys:
                    del self._models[key]
                    self._dynamic_count -= 1
                    break
        self._models[model] = pricing
        self._dynamic_count += 1

    @staticmethod
    def _is_boundary_match(model: str, key: str) -> bool:
        """Check that *key* matches at a word boundary in *model*.

        The character after the prefix must be ``-``, ``/``, or
        end-of-string.  Keys that already end with ``/`` or ``-``
        are treated as namespace prefixes and always match.
        This prevents ``"o3"`` from matching ``"o3000"``.
        """
        if not model.startswith(key):
            return False
        if len(model) == len(key):
            return True
        # Namespace prefixes (e.g. "ollama/") match any suffix
        if key.endswith(("/", "-")):
            return True
        return model[len(key)] in ("-", "/")

    def _resolve_prefix(self, model: str) -> tuple[float, float] | None:
        """Find the longest registered key that is a prefix of *model*."""
        best_match: str | None = None
        best_len = 0
        with self._lock:
            keys = list(self._models)
        for key in keys:
            if self._is_boundary_match(model, key) and len(key) > best_len:
                best_match = key
                best_len = len(key)
        if best_match is not None:
            with self._lock:
                pricing = self._models.get(best_match)
                if pricing is not None:
                    self._cache_dynamic(model, pricing)
                    return pricing
        return None


default_registry = PricingRegistry()
