"""Local SQLite persistence layer for usage logs and budget state (stub)."""

from __future__ import annotations


class UsageStore:
    """Local persistence layer using SQLite.

    Will store per-call usage logs and per-project budget state
    in ~/.llm_budget.db (configurable). Currently a stub — no I/O is performed.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path

    def log_usage(
        self,
        project: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Log a single LLM API call's usage."""

    def get_total_cost(self, project: str) -> float:
        """Get the total accumulated cost for a project."""
        return 0.0
