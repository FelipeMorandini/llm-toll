"""Sliding-window rate limiter for RPM and TPM enforcement."""

from __future__ import annotations


class RateLimiter:
    """Sliding-window rate limiter enforcing RPM and TPM limits locally.

    Prevents HTTP 429 errors by raising LocalRateLimitError
    before the API call is made.
    """

    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> None:
        self._rpm = rpm
        self._tpm = tpm

    def check(self, tokens: int = 0) -> None:
        """Check if the current request would exceed rate limits.

        Currently a no-op stub. Will raise ``LocalRateLimitError``
        when sliding-window enforcement is implemented.
        """
