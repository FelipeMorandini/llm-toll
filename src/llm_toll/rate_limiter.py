"""Sliding-window rate limiter for RPM and TPM enforcement."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from llm_toll.exceptions import LocalRateLimitError

_WINDOW: float = 60.0  # one-minute sliding window


class RateLimiter:
    """Sliding-window rate limiter enforcing RPM and TPM limits locally.

    Prevents HTTP 429 errors by raising :class:`LocalRateLimitError`
    before the API call is made.  Each decorated function gets its own
    ``RateLimiter`` instance so limits are scoped per-decorator.

    Parameters
    ----------
    rpm:
        Maximum requests per minute, or ``None`` to disable.
    tpm:
        Maximum tokens per minute, or ``None`` to disable.
    """

    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
        *,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        if rpm is not None and rpm <= 0:
            raise ValueError(f"rpm must be a positive integer, got {rpm}")
        if tpm is not None and tpm <= 0:
            raise ValueError(f"tpm must be a positive integer, got {tpm}")
        self._rpm = rpm
        self._tpm = tpm
        self._clock = _clock or time.monotonic
        self._request_timestamps: deque[float] = deque()
        self._token_log: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - _WINDOW
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.popleft()

    def check(self) -> None:
        """Check if the next request would exceed rate limits.

        Raises :class:`LocalRateLimitError` if the RPM or TPM limit
        would be breached.  Does **not** record the request — call
        :meth:`record` after a successful API call.

        .. note::

           ``check()`` and ``record()`` each acquire the lock
           independently.  Under concurrent use, multiple threads may
           pass ``check()`` before any of them calls ``record()``,
           slightly exceeding the configured limit.  This is an
           accepted trade-off: the alternative (holding the lock across
           the API call) would serialize all LLM requests.
        """
        if self._rpm is None and self._tpm is None:
            return

        now = self._clock()
        with self._lock:
            self._prune(now)

            if self._rpm is not None and len(self._request_timestamps) >= self._rpm:
                retry_after = self._request_timestamps[0] + _WINDOW - now
                raise LocalRateLimitError(
                    limit_type="rpm",
                    limit_value=self._rpm,
                    retry_after=max(0.0, retry_after),
                )

            if self._tpm is not None:
                total_tokens = sum(t for _, t in self._token_log)
                if total_tokens >= self._tpm:
                    # Walk from oldest, find when enough tokens will expire
                    need_to_free = total_tokens - self._tpm + 1
                    freed = 0
                    retry_after = self._token_log[0][0] + _WINDOW - now
                    for ts, tok in self._token_log:
                        freed += tok
                        retry_after = ts + _WINDOW - now
                        if freed >= need_to_free:
                            break
                    raise LocalRateLimitError(
                        limit_type="tpm",
                        limit_value=self._tpm,
                        retry_after=max(0.0, retry_after),
                    )

    def record(self, tokens: int = 0) -> None:
        """Record a completed request for rate-limiting purposes.

        Called after a successful API call.  Adds the current timestamp
        to the RPM window and ``(timestamp, tokens)`` to the TPM window.
        """
        if self._rpm is None and self._tpm is None:
            return

        now = self._clock()
        with self._lock:
            self._prune(now)
            self._request_timestamps.append(now)
            self._token_log.append((now, tokens))
