"""CLI dashboard for viewing LLM cost and usage statistics."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

from llm_toll import __version__
from llm_toll.store import UsageStore

# ANSI color codes
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_NO_COLOR = bool(os.environ.get("NO_COLOR"))


def _c(text: str, code: str) -> str:
    """Colorize *text* unless NO_COLOR is set."""
    if _NO_COLOR:
        return text
    return f"{code}{text}{_RESET}"


def _cost_color(cost: float) -> str:
    """Return a color-coded cost string."""
    formatted = f"${cost:,.4f}"
    if cost < 0.01:
        return _c(formatted, _GREEN)
    if cost <= 0.10:
        return _c(formatted, _YELLOW)
    return _c(formatted, _RED)


def _print_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> None:
    """Print a fixed-width table with optional ANSI colors.

    *alignments* is a list of ``"<"`` (left) or ``">"`` (right) per column.
    """
    if not rows:
        print("No usage data found.")
        return

    # Strip ANSI for width calculation
    def _visible_len(s: str) -> int:
        import re

        return len(re.sub(r"\033\[[0-9;]*m", "", s))

    all_rows = [headers, *rows]
    col_count = len(headers)
    widths = [0] * col_count
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(cell))

    if alignments is None:
        alignments = ["<"] * col_count

    def _fmt_row(row: list[str]) -> str:
        parts: list[str] = []
        for i, cell in enumerate(row):
            pad = widths[i] - _visible_len(cell)
            if alignments[i] == ">":
                parts.append(" " * pad + cell)
            else:
                parts.append(cell + " " * pad)
        return "  ".join(parts)

    print(_fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(_fmt_row(row))


def _cmd_stats(store: UsageStore, args: argparse.Namespace) -> None:
    """Handle --stats command."""
    if args.model:
        # Per-project breakdown for a specific model
        logs = store.get_usage_logs_filtered(project=args.project, model=args.model, limit=10000)
        if not logs:
            print(f"No usage data found for model '{args.model}'.")
            return

        # Aggregate by project
        agg: dict[str, dict[str, Any]] = {}
        for log in logs:
            p = log["project"]
            if p not in agg:
                agg[p] = {"cost": 0.0, "in": 0, "out": 0, "calls": 0}
            agg[p]["cost"] += log["cost"]
            agg[p]["in"] += log["input_tokens"]
            agg[p]["out"] += log["output_tokens"]
            agg[p]["calls"] += 1

        title = f"Usage for model: {args.model}"
        if args.project:
            title += f" (project: {args.project})"
        print(_c(title, _BOLD))
        print()

        headers = ["Project", "Calls", "Input Tokens", "Output Tokens", "Total Cost"]
        rows = []
        total_cost = 0.0
        for p, d in sorted(agg.items(), key=lambda x: -x[1]["cost"]):
            rows.append(
                [
                    p,
                    f"{d['calls']:,}",
                    f"{d['in']:,}",
                    f"{d['out']:,}",
                    _cost_color(d["cost"]),
                ]
            )
            total_cost += d["cost"]

        rows.append(
            [
                _c("TOTAL", _BOLD),
                "",
                "",
                "",
                _c(f"${total_cost:,.4f}", _BOLD),
            ]
        )
        _print_table(headers, rows, ["<", ">", ">", ">", ">"])
        return

    if args.project:
        # Per-model breakdown for a specific project
        summaries = store.get_model_summaries(project=args.project)
        if not summaries:
            print(f"No usage data found for project '{args.project}'.")
            return

        print(_c(f"Usage for project: {args.project}", _BOLD))
        print()

        headers = ["Model", "Calls", "Input Tokens", "Output Tokens", "Total Cost"]
        rows = []
        total_cost = 0.0
        for s in summaries:
            rows.append(
                [
                    s["model"],
                    f"{s['call_count']:,}",
                    f"{s['total_input_tokens']:,}",
                    f"{s['total_output_tokens']:,}",
                    _cost_color(s["total_cost"]),
                ]
            )
            total_cost += s["total_cost"]

        rows.append(
            [
                _c("TOTAL", _BOLD),
                "",
                "",
                "",
                _c(f"${total_cost:,.4f}", _BOLD),
            ]
        )
        _print_table(headers, rows, ["<", ">", ">", ">", ">"])
        return

    # All projects summary
    summaries = store.get_all_project_summaries()
    if not summaries:
        print("No usage data found.")
        return

    print(_c("llm-toll Usage Summary", _BOLD))
    print()

    headers = ["Project", "Calls", "Input Tokens", "Output Tokens", "Total Cost"]
    rows = []
    total_cost = 0.0
    for s in summaries:
        rows.append(
            [
                s["project"],
                f"{s['call_count']:,}",
                f"{s['total_input_tokens']:,}",
                f"{s['total_output_tokens']:,}",
                _cost_color(s["total_cost"]),
            ]
        )
        total_cost += s["total_cost"]

    rows.append(
        [
            _c("TOTAL", _BOLD),
            "",
            "",
            "",
            _c(f"${total_cost:,.4f}", _BOLD),
        ]
    )
    _print_table(headers, rows, ["<", ">", ">", ">", ">"])


def _cmd_reset(store: UsageStore, args: argparse.Namespace) -> None:
    """Handle --reset command."""
    if not args.project:
        print("Error: --reset requires --project", file=sys.stderr)
        sys.exit(1)

    current = store.get_total_cost(args.project)
    if current == 0.0:
        print(f"Project '{args.project}' has no recorded cost.")
        return

    store.reset_budget(args.project)
    print(f"Budget reset for project '{args.project}' (was ${current:,.4f}).")


def _cmd_export(store: UsageStore, args: argparse.Namespace) -> None:
    """Handle --export csv command."""
    logs = store.get_usage_logs_filtered(project=args.project, model=args.model, limit=100000)
    if not logs:
        print("No usage data to export.", file=sys.stderr)
        sys.exit(1)

    output = open(args.output, "w", newline="") if args.output else sys.stdout  # noqa: SIM115
    try:
        writer = csv.writer(output)
        writer.writerow(
            ["project", "model", "input_tokens", "output_tokens", "cost", "created_at"]
        )
        for log in logs:
            writer.writerow(
                [
                    log["project"],
                    log["model"],
                    log["input_tokens"],
                    log["output_tokens"],
                    log["cost"],
                    log["created_at"],
                ]
            )
    finally:
        if args.output:
            output.close()


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-toll",
        description="View LLM cost and usage statistics.",
    )
    parser.add_argument("--version", action="version", version=f"llm-toll {__version__}")
    parser.add_argument("--db", metavar="PATH", help="Path to the SQLite database")
    parser.add_argument("--project", metavar="NAME", help="Filter by project name")
    parser.add_argument("--model", metavar="NAME", help="Filter by model name")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stats", action="store_true", help="Show usage statistics")
    group.add_argument("--reset", action="store_true", help="Reset budget for a project")
    group.add_argument(
        "--export",
        choices=["csv"],
        metavar="FORMAT",
        help="Export usage logs (csv)",
    )

    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Output file for --export (default: stdout)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    store = UsageStore(db_path=args.db)
    try:
        if args.stats:
            _cmd_stats(store, args)
        elif args.reset:
            _cmd_reset(store, args)
        elif args.export:
            _cmd_export(store, args)
    finally:
        store.close()
