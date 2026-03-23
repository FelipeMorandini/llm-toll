"""Tests for llm_toll.remote_pricing module."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from llm_toll.pricing import PricingRegistry

VALID_PRICING = {
    "gpt-4o": [2.5e-06, 10.0e-06],
    "claude-3-haiku": [2.5e-07, 1.25e-06],
}


def _mock_urlopen_cm(data: dict) -> MagicMock:
    cm = MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = json.dumps(data).encode()
    return cm


def _rp():
    """Return a freshly reloaded remote_pricing module."""
    import llm_toll.remote_pricing

    importlib.reload(llm_toll.remote_pricing)
    return llm_toll.remote_pricing


class TestFetchRemotePricing:
    def test_fetch_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rp = _rp()
        monkeypatch.setattr(rp, "urlopen", lambda *a, **kw: _mock_urlopen_cm(VALID_PRICING))
        result = rp._fetch_remote_pricing("https://example.com/p.json")
        assert result["gpt-4o"] == (2.5e-06, 10.0e-06)
        assert result["claude-3-haiku"] == (2.5e-07, 1.25e-06)

    def test_fetch_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rp = _rp()
        monkeypatch.setattr(
            rp, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(URLError("refused"))
        )
        with pytest.raises(URLError):
            rp._fetch_remote_pricing("https://example.com/p.json")

    def test_fetch_malformed_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rp = _rp()
        cm = MagicMock()
        cm.__enter__ = lambda s: s
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = b"not json{{"
        monkeypatch.setattr(rp, "urlopen", lambda *a, **kw: cm)
        with pytest.raises((json.JSONDecodeError, ValueError)):
            rp._fetch_remote_pricing("https://example.com/p.json")

    def test_fetch_empty_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rp = _rp()
        bad = {"bad": [-1, 0.5], "wrong": "nope"}
        monkeypatch.setattr(rp, "urlopen", lambda *a, **kw: _mock_urlopen_cm(bad))
        with pytest.raises(ValueError, match="no valid"):
            rp._fetch_remote_pricing("https://example.com/p.json")


class TestCache:
    def test_write_and_read_cache(self, tmp_path: Path) -> None:
        rp = _rp()
        p = tmp_path / "cache.json"
        rp._write_cache(p, {"gpt-4o": (2.5e-06, 10.0e-06)})
        result = rp._read_cache(p)
        assert result is not None
        models, ts = result
        assert models["gpt-4o"] == (2.5e-06, 10.0e-06)
        assert isinstance(ts, datetime)

    def test_read_missing(self, tmp_path: Path) -> None:
        rp = _rp()
        assert rp._read_cache(tmp_path / "nope.json") is None

    def test_read_corrupt(self, tmp_path: Path) -> None:
        rp = _rp()
        p = tmp_path / "bad.json"
        p.write_text("garbage")
        assert rp._read_cache(p) is None


class TestUpdatePricing:
    def test_fetches_and_applies(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rp = _rp()
        monkeypatch.setattr(
            rp, "urlopen", lambda *a, **kw: _mock_urlopen_cm({"new": [1e-06, 2e-06]})
        )
        reg = PricingRegistry()
        assert rp.update_pricing(
            url="https://x.com/p.json", cache_path=tmp_path / "c.json", registry=reg
        )
        assert reg.has_model("new")

    def test_uses_fresh_cache(self, tmp_path: Path) -> None:
        rp = _rp()
        p = tmp_path / "c.json"
        rp._write_cache(p, {"cached": (3e-06, 6e-06)})
        reg = PricingRegistry()
        assert rp.update_pricing(
            url="https://x.com/p.json", ttl_hours=24, cache_path=p, registry=reg
        )
        assert reg.has_model("cached")

    def test_refetches_stale(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rp = _rp()
        p = tmp_path / "c.json"
        p.write_text(
            json.dumps(
                {"updated_at": "2020-01-01T00:00:00+00:00", "models": {"old": [1e-06, 2e-06]}}
            )
        )
        monkeypatch.setattr(
            rp, "urlopen", lambda *a, **kw: _mock_urlopen_cm({"fresh": [5e-06, 10e-06]})
        )
        reg = PricingRegistry()
        assert rp.update_pricing(
            url="https://x.com/p.json", ttl_hours=24, cache_path=p, registry=reg
        )
        assert reg.has_model("fresh")

    def test_network_failure_returns_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rp = _rp()

        def _raise(*a, **kw):
            raise URLError("timeout")

        monkeypatch.setattr(rp, "urlopen", _raise)
        reg = PricingRegistry()
        assert not rp.update_pricing(
            url="https://x.com/p.json", cache_path=tmp_path / "c.json", registry=reg
        )

    def test_network_failure_uses_stale_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rp = _rp()
        p = tmp_path / "c.json"
        p.write_text(
            json.dumps(
                {"updated_at": "2020-01-01T00:00:00+00:00", "models": {"stale": [1e-06, 2e-06]}}
            )
        )

        def _raise(*a, **kw):
            raise URLError("down")

        monkeypatch.setattr(rp, "urlopen", _raise)
        reg = PricingRegistry()
        assert rp.update_pricing(
            url="https://x.com/p.json", ttl_hours=24, cache_path=p, registry=reg
        )
        assert reg.has_model("stale")


class TestLoadRemotePricing:
    def test_load_remote_pricing(self) -> None:
        reg = PricingRegistry()
        count = reg.load_remote_pricing({"test-model": (1e-06, 2e-06)})
        assert count == 1
        assert reg.has_model("test-model")
        assert reg.get_cost("test-model", 1000, 500) == pytest.approx(1000 * 1e-06 + 500 * 2e-06)
