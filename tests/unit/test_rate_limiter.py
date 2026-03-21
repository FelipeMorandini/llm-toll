"""Unit tests for RateLimiter."""

from __future__ import annotations

from llm_toll.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter instantiation and stub behavior."""

    def test_instantiate_with_no_args(self) -> None:
        limiter = RateLimiter()
        assert limiter._rpm is None
        assert limiter._tpm is None

    def test_instantiate_with_rpm_and_tpm(self) -> None:
        limiter = RateLimiter(rpm=60, tpm=100_000)
        assert limiter._rpm == 60
        assert limiter._tpm == 100_000

    def test_check_does_not_raise(self) -> None:
        limiter = RateLimiter(rpm=10, tpm=5000)
        limiter.check(tokens=100)
