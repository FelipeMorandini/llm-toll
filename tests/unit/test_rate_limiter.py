"""Unit tests for RateLimiter."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from llm_toll.exceptions import LocalRateLimitError
from llm_toll.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clock(start: float = 0.0) -> tuple[list[float], Callable[[], float]]:
    """Return a mutable time container and a mock clock function."""
    clock_time = [start]

    def mock_clock() -> float:
        return clock_time[0]

    return clock_time, mock_clock


# ---------------------------------------------------------------------------
# TestRateLimiterInstantiation
# ---------------------------------------------------------------------------


class TestRateLimiterInstantiation:
    """Tests for RateLimiter construction."""

    def test_instantiate_with_no_args(self) -> None:
        limiter = RateLimiter()
        assert limiter._rpm is None
        assert limiter._tpm is None

    def test_instantiate_with_rpm_and_tpm(self) -> None:
        limiter = RateLimiter(rpm=60, tpm=100_000)
        assert limiter._rpm == 60
        assert limiter._tpm == 100_000

    def test_instantiate_with_custom_clock(self) -> None:
        _clock_time, mock_clock = _make_clock(42.0)
        limiter = RateLimiter(rpm=10, tpm=5000, _clock=mock_clock)
        # The limiter should use our clock, not time.monotonic
        assert limiter._clock is mock_clock
        # Verify it reads the injected time
        limiter.record(tokens=1)
        assert limiter._request_timestamps[-1] == 42.0


# ---------------------------------------------------------------------------
# TestRPMEnforcement
# ---------------------------------------------------------------------------


class TestRPMEnforcement:
    """Tests for requests-per-minute enforcement."""

    def test_rpm_allows_up_to_limit(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=5, _clock=mock_clock)

        for _i in range(5):
            limiter.check()
            limiter.record()
            clock_time[0] += 0.1  # small advance so timestamps differ

    def test_rpm_blocks_at_limit(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=3, _clock=mock_clock)

        for _ in range(3):
            limiter.check()
            limiter.record()
            clock_time[0] += 0.01

        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()

        assert exc_info.value.limit_type == "rpm"
        assert exc_info.value.limit_value == 3

    def test_rpm_retry_after_is_positive(self) -> None:
        clock_time, mock_clock = _make_clock(100.0)
        limiter = RateLimiter(rpm=2, _clock=mock_clock)

        limiter.check()
        limiter.record()
        clock_time[0] = 100.5
        limiter.check()
        limiter.record()
        clock_time[0] = 101.0

        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()

        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after > 0

    def test_rpm_window_expiry(self) -> None:
        clock_time, mock_clock = _make_clock(0.0)
        limiter = RateLimiter(rpm=2, _clock=mock_clock)

        limiter.check()
        limiter.record()
        clock_time[0] = 1.0
        limiter.check()
        limiter.record()

        # At t=2.0 the window still contains both requests
        clock_time[0] = 2.0
        with pytest.raises(LocalRateLimitError):
            limiter.check()

        # Advance past 60s from the first record (t=0.0) => first entry expires
        clock_time[0] = 60.1
        limiter.check()  # should not raise

    def test_rpm_none_disables(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=None, tpm=1000, _clock=mock_clock)

        # Record many requests — RPM is disabled so only TPM matters
        for _ in range(200):
            limiter.check()
            limiter.record(tokens=0)
            clock_time[0] += 0.001


# ---------------------------------------------------------------------------
# TestTPMEnforcement
# ---------------------------------------------------------------------------


class TestTPMEnforcement:
    """Tests for tokens-per-minute enforcement."""

    def test_tpm_allows_within_limit(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(tpm=1000, _clock=mock_clock)

        limiter.record(tokens=400)
        clock_time[0] += 0.1
        limiter.record(tokens=400)
        clock_time[0] += 0.1
        limiter.check()  # 800 < 1000, should pass

    def test_tpm_blocks_at_limit(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(tpm=500, _clock=mock_clock)

        limiter.record(tokens=300)
        clock_time[0] += 0.1
        limiter.record(tokens=200)
        clock_time[0] += 0.1

        # 300 + 200 = 500 >= 500
        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()

        assert exc_info.value.limit_type == "tpm"
        assert exc_info.value.limit_value == 500

    def test_tpm_retry_after_is_positive(self) -> None:
        clock_time, mock_clock = _make_clock(10.0)
        limiter = RateLimiter(tpm=100, _clock=mock_clock)

        limiter.record(tokens=100)
        clock_time[0] = 15.0

        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()

        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after > 0

    def test_tpm_window_expiry(self) -> None:
        clock_time, mock_clock = _make_clock(0.0)
        limiter = RateLimiter(tpm=100, _clock=mock_clock)

        limiter.record(tokens=100)

        # Still in window
        clock_time[0] = 30.0
        with pytest.raises(LocalRateLimitError):
            limiter.check()

        # Advance past 60s => tokens expire
        clock_time[0] = 60.1
        limiter.check()  # should not raise

    def test_tpm_retry_after_accounts_for_token_distribution(self) -> None:
        """retry_after should point to when *enough* tokens expire, not just the oldest."""
        clock_time, mock_clock = _make_clock(0.0)
        limiter = RateLimiter(tpm=100, _clock=mock_clock)

        # Record 10 tokens at t=0
        limiter.record(tokens=10)
        # Record 90 tokens at t=1
        clock_time[0] = 1.0
        limiter.record(tokens=90)

        # At t=2 total is 100 (>= limit) — should raise
        clock_time[0] = 2.0
        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()

        assert exc_info.value.limit_type == "tpm"
        # We need to free at least 1 token (total=100, limit=100, need_to_free=1).
        # The 10-token batch at t=0 frees first, which is enough.
        # So retry_after should point to t=0 + 60 = 60, i.e. 60 - 2 = 58.
        assert exc_info.value.retry_after == pytest.approx(58.0, abs=0.1)

        # Now test a scenario where we need more tokens freed than the oldest batch
        clock_time2, mock_clock2 = _make_clock(0.0)
        limiter2 = RateLimiter(tpm=100, _clock=mock_clock2)

        # Record 10 tokens at t=0, 10 at t=5, 80 at t=10
        limiter2.record(tokens=10)
        clock_time2[0] = 5.0
        limiter2.record(tokens=10)
        clock_time2[0] = 10.0
        limiter2.record(tokens=80)

        # At t=11 total is 100 (>= limit)
        clock_time2[0] = 11.0
        with pytest.raises(LocalRateLimitError) as exc_info2:
            limiter2.check()

        # need_to_free = 100 - 100 + 1 = 1
        # First batch (10 tokens at t=0) is enough -> retry_after = 0 + 60 - 11 = 49
        assert exc_info2.value.retry_after == pytest.approx(49.0, abs=0.1)

        # Now make the first batch small and require the second batch too
        clock_time3, mock_clock3 = _make_clock(0.0)
        limiter3 = RateLimiter(tpm=100, _clock=mock_clock3)

        # Record 5 tokens at t=0, 90 tokens at t=30
        limiter3.record(tokens=5)
        clock_time3[0] = 30.0
        limiter3.record(tokens=90)
        # Add 10 more at t=31 to exceed limit (total=105)
        clock_time3[0] = 31.0
        limiter3.record(tokens=10)

        # At t=32 total is 105, need_to_free = 105 - 100 + 1 = 6
        clock_time3[0] = 32.0
        with pytest.raises(LocalRateLimitError) as exc_info3:
            limiter3.check()

        # 5 tokens at t=0 not enough (freed=5 < 6), need 90 tokens at t=30 too
        # retry_after = 30 + 60 - 32 = 58
        assert exc_info3.value.retry_after == pytest.approx(58.0, abs=0.1)

    def test_tpm_none_disables(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=1000, tpm=None, _clock=mock_clock)

        # Record huge token counts — TPM is disabled so only RPM matters
        for _ in range(10):
            limiter.check()
            limiter.record(tokens=999_999)
            clock_time[0] += 0.001


# ---------------------------------------------------------------------------
# TestRPMAndTPMCombined
# ---------------------------------------------------------------------------


class TestRPMAndTPMCombined:
    """Tests for interactions between RPM and TPM limits."""

    def test_rpm_trips_without_tpm(self) -> None:
        """RPM can be exceeded even when TPM headroom remains."""
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=2, tpm=100_000, _clock=mock_clock)

        limiter.check()
        limiter.record(tokens=1)
        clock_time[0] += 0.01
        limiter.check()
        limiter.record(tokens=1)
        clock_time[0] += 0.01

        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()
        assert exc_info.value.limit_type == "rpm"

    def test_tpm_trips_without_rpm(self) -> None:
        """TPM can be exceeded even when RPM headroom remains."""
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=100, tpm=500, _clock=mock_clock)

        limiter.check()
        limiter.record(tokens=500)
        clock_time[0] += 0.01

        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()
        assert exc_info.value.limit_type == "tpm"

    def test_both_limits_none_is_noop(self) -> None:
        """When both limits are None, check/record are fast no-ops."""
        limiter = RateLimiter(rpm=None, tpm=None)

        # Should complete instantly without acquiring the lock or recording
        for _ in range(1000):
            limiter.check()
            limiter.record(tokens=999_999)

        # Internal deques should remain empty (early return before lock)
        assert len(limiter._request_timestamps) == 0
        assert len(limiter._token_log) == 0


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for concurrent access."""

    def test_concurrent_check_and_record(self) -> None:
        """Spawn multiple threads doing check+record; no crashes expected."""
        _clock_time, mock_clock = _make_clock()
        # Use generous limits so most calls succeed
        limiter = RateLimiter(rpm=500, tpm=500_000, _clock=mock_clock)

        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def worker() -> None:
            try:
                barrier.wait()
                for _ in range(50):
                    try:
                        limiter.check()
                        limiter.record(tokens=10)
                    except LocalRateLimitError:
                        pass  # expected under contention
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Unexpected errors in threads: {errors}"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests."""

    def test_record_zero_tokens_still_counts_for_rpm(self) -> None:
        clock_time, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=2, _clock=mock_clock)

        limiter.check()
        limiter.record(tokens=0)
        clock_time[0] += 0.01
        limiter.check()
        limiter.record(tokens=0)
        clock_time[0] += 0.01

        with pytest.raises(LocalRateLimitError) as exc_info:
            limiter.check()
        assert exc_info.value.limit_type == "rpm"

    def test_check_without_prior_record(self) -> None:
        """First call to check always passes when limits are set."""
        _, mock_clock = _make_clock()
        limiter = RateLimiter(rpm=1, tpm=1, _clock=mock_clock)
        limiter.check()  # should not raise
