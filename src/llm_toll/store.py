"""Local SQLite persistence layer for usage logs and budget state."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_toll.exceptions import BudgetExceededError


class UsageStore:
    """Local persistence layer using SQLite.

    Stores per-call usage logs and per-project budget state.
    Default database path is ``~/.llm_toll.db`` (configurable).

    The connection is opened lazily on first use. All write operations
    are serialized via a lock for thread safety.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            resolved = Path.home() / ".llm_toll.db"
        else:
            # Reject obvious path traversal before resolving
            if ".." in Path(db_path).parts:
                raise ValueError(f"db_path must not contain '..' segments: {db_path}")
            resolved = Path(db_path).expanduser().resolve()
        self._db_path: str = str(resolved)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def _get_conn(self) -> sqlite3.Connection:
        """Return the (lazily created) database connection.

        Uses double-checked locking to avoid races during lazy init.
        """
        if self._conn is not None:
            return self._conn
        with self._lock:
            # Re-check under lock
            if self._conn is not None:
                return self._conn
            try:
                db_path = Path(self._db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                is_new = not db_path.exists()
                conn = sqlite3.connect(self._db_path, check_same_thread=False)
                if is_new:
                    os.chmod(self._db_path, stat.S_IRUSR | stat.S_IWUSR)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA synchronous=NORMAL")
                self._init_schema(conn)
                self._conn = conn
                return conn
            except OSError as exc:
                raise sqlite3.OperationalError(
                    f"Failed to initialize store at {self._db_path}: {exc}"
                ) from exc

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        """Create tables and indexes if they do not exist."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project       TEXT    NOT NULL,
                model         TEXT    NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost          REAL    NOT NULL,
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS budgets (
                project       TEXT PRIMARY KEY,
                total_cost    REAL NOT NULL DEFAULT 0.0,
                last_reset_at TEXT,
                updated_at    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_usage_logs_project_created
                ON usage_logs (project, created_at DESC);
            """
        )

    def log_usage(
        self,
        project: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Log a single LLM API call and update the project budget.

        The INSERT into ``usage_logs`` and the budget UPSERT run in a
        single transaction for atomicity.
        """
        now = _utc_now_iso()
        try:
            with self._lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        "INSERT INTO usage_logs "
                        "(project, model, input_tokens, output_tokens, cost, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (project, model, input_tokens, output_tokens, cost, now),
                    )
                    conn.execute(
                        "INSERT INTO budgets (project, total_cost, updated_at) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT (project) DO UPDATE SET "
                        "total_cost = total_cost + excluded.total_cost, "
                        "updated_at = excluded.updated_at",
                        (project, cost, now),
                    )
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to log usage to {self._db_path}: {exc}",
                stacklevel=2,
            )

    def log_usage_if_within_budget(
        self,
        project: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        max_budget: float,
    ) -> float:
        """Atomically check budget and log usage in a single transaction.

        If the current accumulated cost already meets or exceeds
        *max_budget*, raises :class:`BudgetExceededError` without
        logging the usage.  If the new cost would push the total
        **over** *max_budget*, raises the same exception.  Otherwise
        the usage is logged and the new total cost is returned.

        Reaching *exactly* ``max_budget`` is allowed (i.e. the
        post-add comparison uses ``>`` not ``>=``).

        On database errors the call fails open: a warning is issued
        and the method returns ``0.0`` so callers may treat this as
        "no cost recorded" (unlike :meth:`log_usage`, which returns
        ``None``).
        """
        now = _utc_now_iso()
        try:
            with self._lock:
                conn = self._get_conn()
                with conn:
                    row = conn.execute(
                        "SELECT total_cost FROM budgets WHERE project = ?",
                        (project,),
                    ).fetchone()
                    current_cost = float(row[0]) if row is not None else 0.0

                    if current_cost >= max_budget:
                        raise BudgetExceededError(
                            project=project,
                            current_cost=current_cost,
                            max_budget=max_budget,
                        )

                    new_total = current_cost + cost
                    if new_total > max_budget:
                        raise BudgetExceededError(
                            project=project,
                            current_cost=new_total,
                            max_budget=max_budget,
                        )

                    conn.execute(
                        "INSERT INTO usage_logs "
                        "(project, model, input_tokens, output_tokens, cost, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (project, model, input_tokens, output_tokens, cost, now),
                    )
                    conn.execute(
                        "INSERT INTO budgets (project, total_cost, updated_at) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT (project) DO UPDATE SET "
                        "total_cost = total_cost + excluded.total_cost, "
                        "updated_at = excluded.updated_at",
                        (project, cost, now),
                    )
            return new_total
        except BudgetExceededError:
            raise
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to log usage to {self._db_path}: {exc}",
                stacklevel=2,
            )
            return 0.0

    def get_total_cost(self, project: str) -> float:
        """Get the total accumulated cost for a project.

        Returns ``0.0`` if the project has no recorded usage or on DB error.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT total_cost FROM budgets WHERE project = ?",
                    (project,),
                ).fetchone()
            if row is None:
                return 0.0
            return float(row[0])
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to read budget from {self._db_path}: {exc}",
                stacklevel=2,
            )
            return 0.0

    def get_usage_logs(
        self,
        project: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent usage log entries for a project.

        Results are ordered by timestamp descending (most recent first).
        Returns an empty list on DB error.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                cursor = conn.execute(
                    "SELECT id, project, model, input_tokens, output_tokens, "
                    "cost, created_at "
                    "FROM usage_logs WHERE project = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    (project, limit),
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to read usage logs from {self._db_path}: {exc}",
                stacklevel=2,
            )
            return []

    def get_all_project_summaries(self) -> list[dict[str, Any]]:
        """Return aggregated usage stats per project.

        Each dict contains ``project``, ``total_cost``,
        ``total_input_tokens``, ``total_output_tokens``, ``call_count``,
        and ``last_used``.  Ordered by total cost descending.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                cursor = conn.execute(
                    "SELECT project, "
                    "SUM(cost) AS total_cost, "
                    "SUM(input_tokens) AS total_input_tokens, "
                    "SUM(output_tokens) AS total_output_tokens, "
                    "COUNT(*) AS call_count, "
                    "MAX(created_at) AS last_used "
                    "FROM usage_logs GROUP BY project ORDER BY total_cost DESC",
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to read project summaries from {self._db_path}: {exc}",
                stacklevel=2,
            )
            return []

    def get_model_summaries(self, project: str | None = None) -> list[dict[str, Any]]:
        """Return aggregated usage stats per model.

        Optionally filtered by *project*.  Each dict contains ``model``,
        ``total_cost``, ``total_input_tokens``, ``total_output_tokens``,
        and ``call_count``.  Ordered by total cost descending.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                if project is not None:
                    cursor = conn.execute(
                        "SELECT model, "
                        "SUM(cost) AS total_cost, "
                        "SUM(input_tokens) AS total_input_tokens, "
                        "SUM(output_tokens) AS total_output_tokens, "
                        "COUNT(*) AS call_count "
                        "FROM usage_logs WHERE project = ? "
                        "GROUP BY model ORDER BY total_cost DESC",
                        (project,),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT model, "
                        "SUM(cost) AS total_cost, "
                        "SUM(input_tokens) AS total_input_tokens, "
                        "SUM(output_tokens) AS total_output_tokens, "
                        "COUNT(*) AS call_count "
                        "FROM usage_logs GROUP BY model ORDER BY total_cost DESC",
                    )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to read model summaries from {self._db_path}: {exc}",
                stacklevel=2,
            )
            return []

    def get_usage_logs_filtered(
        self,
        project: str | None = None,
        model: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return usage log entries with optional project/model filters.

        Results are ordered by timestamp descending.
        Returns an empty list on DB error.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                clauses: list[str] = []
                params: list[Any] = []
                if project is not None:
                    clauses.append("project = ?")
                    params.append(project)
                if model is not None:
                    clauses.append("model = ?")
                    params.append(model)
                where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
                params.append(limit)
                cursor = conn.execute(
                    "SELECT id, project, model, input_tokens, output_tokens, "
                    f"cost, created_at FROM usage_logs{where} "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    params,
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to read usage logs from {self._db_path}: {exc}",
                stacklevel=2,
            )
            return []

    def reset_budget(self, project: str) -> None:
        """Reset the accumulated cost for a project to zero."""
        now = _utc_now_iso()
        try:
            with self._lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        "UPDATE budgets SET total_cost = 0.0, "
                        "last_reset_at = ?, updated_at = ? "
                        "WHERE project = ?",
                        (now, now, project),
                    )
        except sqlite3.Error as exc:
            warnings.warn(
                f"Failed to reset budget in {self._db_path}: {exc}",
                stacklevel=2,
            )

    def close(self) -> None:
        """Close the database connection if open."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
