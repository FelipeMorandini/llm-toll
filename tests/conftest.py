"""Shared test fixtures for llm_toll."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_llm_toll.db")
