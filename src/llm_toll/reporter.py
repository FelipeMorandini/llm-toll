"""Color-coded terminal cost reporter."""

from __future__ import annotations

import os
import sys
import threading
from typing import IO

from llm_toll.pricing import COST_ROUND_PLACES

# ANSI color codes
_CYAN = "36"
_GREEN = "32"
_YELLOW = "33"
_RED = "31"
_BOLD = "1"


class CostReporter:
    """Formats and prints color-coded terminal summaries of per-call and session costs.

    Output goes to *file* (default ``sys.stderr``) so it does not
    pollute program stdout.  Colors are suppressed when the ``NO_COLOR``
    environment variable is set (see https://no-color.org/).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        file: IO[str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._file: IO[str] = file or sys.stderr
        self._use_color = "NO_COLOR" not in os.environ
        self._session_cost: float = 0.0
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._call_count: int = 0
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def report_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Print a per-call cost summary and update session accumulators."""
        with self._lock:
            self._session_cost = round(self._session_cost + cost, COST_ROUND_PLACES)
            self._session_input_tokens += input_tokens
            self._session_output_tokens += output_tokens
            self._call_count += 1

        if not self._enabled:
            return

        colored_model = self._colorize(model, _CYAN)
        colored_cost = self._colorize(f"${cost:.4f}", self._cost_color(cost))
        line = (
            f"[cost] {colored_model}: "
            f"{input_tokens:,} in / {output_tokens:,} out -> {colored_cost}"
        )
        print(line, file=self._file)

    def report_session(self) -> None:
        """Print a session summary with totals."""
        with self._lock:
            if not self._enabled or self._call_count == 0:
                return
            cost = self._session_cost
            input_t = self._session_input_tokens
            output_t = self._session_output_tokens
            count = self._call_count

        colored_total = self._colorize(
            f"${cost:.4f}",
            f"{_BOLD};{self._cost_color(cost)}",
        )
        line = f"[session] {count} calls, {input_t:,} in / {output_t:,} out, total {colored_total}"
        print(line, file=self._file)

    def reset(self) -> None:
        """Zero all session accumulators."""
        with self._lock:
            self._session_cost = 0.0
            self._session_input_tokens = 0
            self._session_output_tokens = 0
            self._call_count = 0

    # -- private helpers -----------------------------------------------------

    def _colorize(self, text: str, code: str) -> str:
        """Wrap *text* in ANSI escape codes when colors are active."""
        if not self._use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _cost_color(self, cost: float) -> str:
        """Return the ANSI code for the cost threshold."""
        if cost < 0.01:
            return _GREEN
        if cost <= 0.10:
            return _YELLOW
        return _RED
