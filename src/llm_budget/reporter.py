"""Color-coded terminal cost reporter (stub)."""

from __future__ import annotations


class CostReporter:
    """Formats and prints color-coded terminal summaries of per-call and session costs.

    Currently a stub — methods are no-ops. Output will be implemented in a future release.
    """

    def __init__(self) -> None:
        self._session_cost: float = 0.0

    def report_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Print a summary of a single LLM API call's cost."""

    def report_session(self) -> None:
        """Print the total session cost summary."""
