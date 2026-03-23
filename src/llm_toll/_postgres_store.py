"""PostgreSQL persistence layer for usage logs and budget state."""

from __future__ import annotations

import warnings
from typing import Any

from llm_toll.exceptions import BudgetExceededError
from llm_toll.store import BaseStore, _utc_now_iso


class PostgresStore(BaseStore):
    """PostgreSQL-backed persistence layer using psycopg2.

    Uses a :class:`~psycopg2.pool.ThreadedConnectionPool` for
    connection management.  The schema mirrors the SQLite store but
    uses PostgreSQL-native types (``SERIAL``, ``TIMESTAMPTZ``).

    Parameters
    ----------
    dsn:
        PostgreSQL connection string (``postgresql://...``).
    min_conn:
        Minimum number of connections in the pool.
    max_conn:
        Maximum number of connections in the pool.
    """

    def __init__(
        self,
        dsn: str,
        min_conn: int = 1,
        max_conn: int = 10,
    ) -> None:
        try:
            import psycopg2  # type: ignore[import-untyped]
            import psycopg2.pool  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install it with: pip install 'llm-toll[postgres]'"
            ) from exc

        self._dsn = dsn
        self._pool: psycopg2.pool.ThreadedConnectionPool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, dsn
        )
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                        CREATE TABLE IF NOT EXISTS usage_logs (
                            id            SERIAL PRIMARY KEY,
                            project       TEXT        NOT NULL,
                            model         TEXT        NOT NULL,
                            input_tokens  INTEGER     NOT NULL,
                            output_tokens INTEGER     NOT NULL,
                            cost          DOUBLE PRECISION NOT NULL,
                            created_at    TIMESTAMPTZ NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS budgets (
                            project       TEXT PRIMARY KEY,
                            total_cost    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                            last_reset_at TIMESTAMPTZ,
                            updated_at    TIMESTAMPTZ NOT NULL
                        );

                        CREATE INDEX IF NOT EXISTS idx_usage_logs_project_created
                            ON usage_logs (project, created_at DESC);
                        """
                )
        finally:
            self._pool.putconn(conn)

    def log_usage(
        self,
        project: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Log a single LLM API call and update the project budget."""
        now = _utc_now_iso()
        conn = self._pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO usage_logs "
                    "(project, model, input_tokens, output_tokens, cost, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (project, model, input_tokens, output_tokens, cost, now),
                )
                cur.execute(
                    "INSERT INTO budgets (project, total_cost, updated_at) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (project) DO UPDATE SET "
                    "total_cost = budgets.total_cost + EXCLUDED.total_cost, "
                    "updated_at = EXCLUDED.updated_at",
                    (project, cost, now),
                )
        except Exception as exc:
            warnings.warn(
                f"Failed to log usage to PostgreSQL: {exc}",
                stacklevel=2,
            )
        finally:
            self._pool.putconn(conn)

    def log_usage_if_within_budget(
        self,
        project: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        max_budget: float,
    ) -> float:
        """Atomically check budget and log usage using row-level locking."""
        now = _utc_now_iso()
        conn = self._pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                # Lock the budget row for this project
                cur.execute(
                    "SELECT total_cost FROM budgets WHERE project = %s FOR UPDATE",
                    (project,),
                )
                row = cur.fetchone()
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

                cur.execute(
                    "INSERT INTO usage_logs "
                    "(project, model, input_tokens, output_tokens, cost, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (project, model, input_tokens, output_tokens, cost, now),
                )
                cur.execute(
                    "INSERT INTO budgets (project, total_cost, updated_at) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (project) DO UPDATE SET "
                    "total_cost = budgets.total_cost + EXCLUDED.total_cost, "
                    "updated_at = EXCLUDED.updated_at",
                    (project, cost, now),
                )
            return new_total
        except BudgetExceededError:
            raise
        except Exception as exc:
            warnings.warn(
                f"Failed to log usage to PostgreSQL: {exc}",
                stacklevel=2,
            )
            return 0.0
        finally:
            self._pool.putconn(conn)

    def get_total_cost(self, project: str) -> float:
        """Get the total accumulated cost for a project."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT total_cost FROM budgets WHERE project = %s",
                    (project,),
                )
                row = cur.fetchone()
            if row is None:
                return 0.0
            return float(row[0])
        except Exception as exc:
            warnings.warn(
                f"Failed to read budget from PostgreSQL: {exc}",
                stacklevel=2,
            )
            return 0.0
        finally:
            self._pool.putconn(conn)

    def get_usage_logs(
        self,
        project: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent usage log entries for a project."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, project, model, input_tokens, output_tokens, "
                    "cost, created_at "
                    "FROM usage_logs WHERE project = %s "
                    "ORDER BY created_at DESC, id DESC LIMIT %s",
                    (project, limit),
                )
                if cur.description is None:
                    return []
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        except Exception as exc:
            warnings.warn(
                f"Failed to read usage logs from PostgreSQL: {exc}",
                stacklevel=2,
            )
            return []
        finally:
            self._pool.putconn(conn)

    def get_all_project_summaries(self) -> list[dict[str, Any]]:
        """Return aggregated usage stats per project."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT project, "
                    "SUM(cost) AS total_cost, "
                    "SUM(input_tokens) AS total_input_tokens, "
                    "SUM(output_tokens) AS total_output_tokens, "
                    "COUNT(*) AS call_count, "
                    "MAX(created_at) AS last_used "
                    "FROM usage_logs GROUP BY project ORDER BY total_cost DESC",
                )
                if cur.description is None:
                    return []
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        except Exception as exc:
            warnings.warn(
                f"Failed to read project summaries from PostgreSQL: {exc}",
                stacklevel=2,
            )
            return []
        finally:
            self._pool.putconn(conn)

    def get_model_summaries(self, project: str | None = None) -> list[dict[str, Any]]:
        """Return aggregated usage stats per model."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                if project is not None:
                    cur.execute(
                        "SELECT model, "
                        "SUM(cost) AS total_cost, "
                        "SUM(input_tokens) AS total_input_tokens, "
                        "SUM(output_tokens) AS total_output_tokens, "
                        "COUNT(*) AS call_count "
                        "FROM usage_logs WHERE project = %s "
                        "GROUP BY model ORDER BY total_cost DESC",
                        (project,),
                    )
                else:
                    cur.execute(
                        "SELECT model, "
                        "SUM(cost) AS total_cost, "
                        "SUM(input_tokens) AS total_input_tokens, "
                        "SUM(output_tokens) AS total_output_tokens, "
                        "COUNT(*) AS call_count "
                        "FROM usage_logs GROUP BY model ORDER BY total_cost DESC",
                    )
                if cur.description is None:
                    return []
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        except Exception as exc:
            warnings.warn(
                f"Failed to read model summaries from PostgreSQL: {exc}",
                stacklevel=2,
            )
            return []
        finally:
            self._pool.putconn(conn)

    def get_project_summaries_for_model(self, model: str) -> list[dict[str, Any]]:
        """Return aggregated usage stats per project for a specific model."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT project, "
                    "SUM(cost) AS total_cost, "
                    "SUM(input_tokens) AS total_input_tokens, "
                    "SUM(output_tokens) AS total_output_tokens, "
                    "COUNT(*) AS call_count "
                    "FROM usage_logs WHERE model = %s "
                    "GROUP BY project ORDER BY total_cost DESC",
                    (model,),
                )
                if cur.description is None:
                    return []
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        except Exception as exc:
            warnings.warn(
                f"Failed to read project summaries from PostgreSQL: {exc}",
                stacklevel=2,
            )
            return []
        finally:
            self._pool.putconn(conn)

    def get_usage_logs_filtered(
        self,
        project: str | None = None,
        model: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return usage log entries with optional project/model filters."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                clauses: list[str] = []
                params: list[Any] = []
                if project is not None:
                    clauses.append("project = %s")
                    params.append(project)
                if model is not None:
                    clauses.append("model = %s")
                    params.append(model)
                where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
                limit_clause = ""
                if limit > 0:
                    limit_clause = " LIMIT %s"
                    params.append(limit)
                cur.execute(
                    "SELECT id, project, model, input_tokens, output_tokens, "
                    f"cost, created_at FROM usage_logs{where} "
                    f"ORDER BY created_at DESC, id DESC{limit_clause}",
                    params,
                )
                if cur.description is None:
                    return []
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
        except Exception as exc:
            warnings.warn(
                f"Failed to read usage logs from PostgreSQL: {exc}",
                stacklevel=2,
            )
            return []
        finally:
            self._pool.putconn(conn)

    def reset_budget(self, project: str) -> None:
        """Reset the accumulated cost for a project to zero."""
        now = _utc_now_iso()
        conn = self._pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE budgets SET total_cost = 0.0, "
                    "last_reset_at = %s, updated_at = %s "
                    "WHERE project = %s",
                    (now, now, project),
                )
        except Exception as exc:
            warnings.warn(
                f"Failed to reset budget in PostgreSQL: {exc}",
                stacklevel=2,
            )
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()
