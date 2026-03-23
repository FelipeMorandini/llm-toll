"""Remote pricing updates with local caching."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from llm_toll.pricing import PricingRegistry, default_registry

DEFAULT_PRICING_URL = (
    "https://raw.githubusercontent.com/FelipeMorandini/llm-toll/main/pricing.json"
)
DEFAULT_CACHE_DIR = Path.home() / ".llm_toll"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "pricing_cache.json"
DEFAULT_TTL_HOURS = 24


def _fetch_remote_pricing(url: str, timeout: int = 5) -> dict[str, tuple[float, float]]:
    """Fetch pricing JSON from *url* and return a validated dict.

    Raises :class:`URLError` on network failure and :class:`ValueError`
    on malformed data.
    """
    with urlopen(url, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode())

    if not isinstance(raw, dict):
        raise ValueError("Pricing JSON must be a top-level object")

    models: dict[str, tuple[float, float]] = {}
    for name, costs in raw.items():
        if not isinstance(name, str) or not isinstance(costs, list) or len(costs) != 2:
            continue
        inp, out = costs
        if not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
            continue
        if inp < 0 or out < 0:
            continue
        models[name] = (float(inp), float(out))

    if not models:
        raise ValueError("Pricing JSON contained no valid model entries")

    return models


def _read_cache(
    cache_path: Path,
) -> tuple[dict[str, tuple[float, float]], datetime] | None:
    """Read the local pricing cache. Returns ``None`` if missing or corrupt."""
    try:
        data = json.loads(cache_path.read_text())
        updated_at = datetime.fromisoformat(data["updated_at"])
        models: dict[str, tuple[float, float]] = {}
        for name, costs in data["models"].items():
            if isinstance(costs, list) and len(costs) == 2:
                models[name] = (float(costs[0]), float(costs[1]))
        if not models:
            return None
        return models, updated_at
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _write_cache(cache_path: Path, models: dict[str, tuple[float, float]]) -> None:
    """Write the pricing cache atomically."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": {k: list(v) for k, v in models.items()},
    }
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, cache_path)


def update_pricing(
    *,
    url: str | None = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    cache_path: Path | None = None,
    registry: PricingRegistry | None = None,
) -> bool:
    """Fetch latest model pricing and apply it to the registry.

    Returns ``True`` if pricing was updated (from cache or remote),
    ``False`` on network failure (built-in pricing is retained).

    Parameters
    ----------
    url:
        URL to fetch pricing JSON from.  Defaults to the GitHub-hosted
        ``pricing.json`` in the llm-toll repository.
    ttl_hours:
        Cache time-to-live in hours.  If a fresh cache exists, no
        network request is made.
    cache_path:
        Path to the local cache file.  Defaults to
        ``~/.llm_toll/pricing_cache.json``.
    registry:
        The :class:`PricingRegistry` to update.  Defaults to the
        shared ``default_registry``.
    """
    if url is None:
        url = DEFAULT_PRICING_URL
    if cache_path is None:
        cache_path = DEFAULT_CACHE_FILE
    if registry is None:
        registry = default_registry

    # Check cache first
    cached = _read_cache(cache_path)
    if cached is not None:
        models, updated_at = cached
        age_hours = (datetime.now(timezone.utc) - updated_at).total_seconds() / 3600
        if age_hours < ttl_hours:
            registry.load_remote_pricing(models)
            return True

    # Fetch from remote
    try:
        models = _fetch_remote_pricing(url)
    except (URLError, ValueError, TimeoutError, OSError) as exc:
        warnings.warn(
            f"Failed to fetch pricing from {url}: {exc}. Using built-in pricing.",
            stacklevel=2,
        )
        # If we have a stale cache, use it as fallback
        if cached is not None:
            registry.load_remote_pricing(cached[0])
            return True
        return False

    _write_cache(cache_path, models)
    registry.load_remote_pricing(models)
    return True
