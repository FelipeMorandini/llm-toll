"""Unit tests for the async streaming module."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from llm_toll.async_streaming import _is_async_stream, wrap_async_stream
from llm_toll.decorator import set_store, track_costs
from llm_toll.pricing import default_registry
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore

# ---------------------------------------------------------------------------
# Mock async iterator helper
# ---------------------------------------------------------------------------


class _AsyncIter:
    """A simple async iterator over a list of items."""

    def __init__(self, items: list[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock chunk classes (same as test_streaming.py)
# ---------------------------------------------------------------------------


class _MockDelta:
    def __init__(self, content: str | None = None) -> None:
        self.content = content


class _MockChoice:
    def __init__(self, delta: _MockDelta) -> None:
        self.delta = delta


class _MockChunkUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockOpenAIChunk:
    """Mimics an OpenAI ChatCompletionChunk."""

    def __init__(
        self,
        model: str,
        content: str | None = None,
        usage: _MockChunkUsage | None = None,
    ) -> None:
        self.model = model
        self.choices = [_MockChoice(_MockDelta(content))]
        self.usage = usage


def _make_openai_chunks(
    model: str,
    texts: list[str],
    final_usage: _MockChunkUsage | None = None,
) -> list[_MockOpenAIChunk]:
    """Build a list of OpenAI-like chunks, with optional usage on the last chunk."""
    chunks = [_MockOpenAIChunk(model=model, content=t) for t in texts]
    if final_usage is not None:
        chunks.append(_MockOpenAIChunk(model=model, content=None, usage=final_usage))
    return chunks


# ===========================================================================
# TestIsAsyncStream
# ===========================================================================


class TestIsAsyncStream:
    def test_async_generator_is_stream(self) -> None:
        async def gen():
            yield 1

        assert _is_async_stream(gen()) is True

    def test_sync_generator_not_async_stream(self) -> None:
        def gen():
            yield 1

        assert _is_async_stream(gen()) is False

    def test_regular_object_not_async_stream(self) -> None:
        assert _is_async_stream("hello") is False
        assert _is_async_stream(42) is False
        assert _is_async_stream({"key": "value"}) is False
        assert _is_async_stream(None) is False

    def test_async_iterator_with_aclose_is_stream(self) -> None:
        """SDK-like async stream objects with __aiter__, __anext__, aclose should be detected."""
        stream = _AsyncIter([1, 2, 3])
        assert _is_async_stream(stream) is True

    def test_list_not_async_stream(self) -> None:
        assert _is_async_stream([1, 2, 3]) is False

    def test_bytes_not_async_stream(self) -> None:
        assert _is_async_stream(b"data") is False


# ===========================================================================
# TestWrapAsyncStream
# ===========================================================================


class TestWrapAsyncStream:
    """Tests for wrap_async_stream."""

    def _make_deps(self, db_path: str) -> tuple[UsageStore, CostReporter]:
        store = UsageStore(db_path=db_path)
        reporter = CostReporter()
        return store, reporter

    @pytest.mark.asyncio
    async def test_yields_all_chunks(self, tmp_db_path: str) -> None:
        store, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=10, completion_tokens=20)
        chunks = _make_openai_chunks("gpt-4o", ["a", "b", "c"], usage)

        wrapped = wrap_async_stream(
            _AsyncIter(chunks),
            project="test",
            model_override=None,
            max_budget=None,
            store=store,
            registry=default_registry,
            reporter=reporter,
        )
        collected = []
        async for chunk in wrapped:
            collected.append(chunk)

        assert len(collected) == len(chunks)
        store.close()

    @pytest.mark.asyncio
    async def test_logs_cost_after_exhaustion(self, tmp_db_path: str) -> None:
        store, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hello", " world"], usage)

        wrapped = wrap_async_stream(
            _AsyncIter(chunks),
            project="proj",
            model_override=None,
            max_budget=None,
            store=store,
            registry=default_registry,
            reporter=reporter,
        )
        async for _ in wrapped:
            pass

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("proj") == pytest.approx(expected_cost)
        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4o"
        store.close()

    @pytest.mark.asyncio
    async def test_logs_cost_on_early_break(self, tmp_db_path: str) -> None:
        store, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["a", "b", "c", "d"], usage)

        wrapped = wrap_async_stream(
            _AsyncIter(chunks),
            project="proj",
            model_override=None,
            max_budget=None,
            store=store,
            registry=default_registry,
            reporter=reporter,
        )

        count = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            async for _ in wrapped:
                count += 1
                if count == 2:
                    break
            # Explicitly close the async generator to trigger the finally block
            await wrapped.aclose()

        # After early break, partial text was accumulated. Fallback estimation logs usage.
        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4o"
        store.close()

    @pytest.mark.asyncio
    async def test_budget_enforcement_in_async_stream(self, tmp_db_path: str) -> None:
        store, reporter = self._make_deps(tmp_db_path)

        # Pre-fill cost that already exceeds the budget
        store.log_usage("proj", "gpt-4o", 1000, 500, 5.0)

        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hello"], usage)

        wrapped = wrap_async_stream(
            _AsyncIter(chunks),
            project="proj",
            model_override=None,
            max_budget=0.001,  # far below the existing 5.0 cost
            store=store,
            registry=default_registry,
            reporter=reporter,
        )

        # Consume the stream -- budget exceeded warning should be issued
        with pytest.warns(match="exceeded budget"):
            async for _ in wrapped:
                pass

        # Cost IS logged even though budget was exceeded (to keep totals accurate)
        logs = store.get_usage_logs("proj")
        assert len(logs) == 2  # pre-fill + the streaming call
        store.close()


# ===========================================================================
# TestDecoratorWithAsyncStreaming
# ===========================================================================


class TestDecoratorWithAsyncStreaming:
    """Integration tests: @track_costs on async generator functions."""

    def teardown_method(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None:
            store.close()
            self._store = None  # type: ignore[assignment]
        set_store(None)

    def _make_store(self, tmp_db_path: str) -> UsageStore:
        store = UsageStore(db_path=tmp_db_path)
        set_store(store)
        self._store = store  # type: ignore[assignment]
        return store

    @pytest.mark.asyncio
    async def test_decorated_async_generator(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hello", " world"], usage)

        @track_costs(project="async-stream-proj")
        async def streaming_call() -> Any:
            for chunk in chunks:
                yield chunk

        # The decorated async gen wrapper yields chunks directly
        collected = []
        async for chunk in streaming_call():
            collected.append(chunk)

        assert len(collected) == len(chunks)

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("async-stream-proj") == pytest.approx(expected_cost)
