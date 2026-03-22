"""Unit tests for the streaming module."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from llm_toll.decorator import set_store, track_costs
from llm_toll.pricing import PricingRegistry, default_registry
from llm_toll.reporter import CostReporter
from llm_toll.store import UsageStore
from llm_toll.streaming import (
    StreamAccumulator,
    _is_sync_stream,
    estimate_tokens,
    wrap_sync_stream,
)

# ---------------------------------------------------------------------------
# Mock classes for OpenAI streaming chunks
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


# ---------------------------------------------------------------------------
# Mock classes for Anthropic streaming events
# ---------------------------------------------------------------------------


class _MockAnthropicUsage:
    def __init__(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockAnthropicMessage:
    def __init__(self, model: str, usage: _MockAnthropicUsage) -> None:
        self.model = model
        self.usage = usage


class _MockAnthropicEvent:
    """Mimics an Anthropic streaming event with a .type attribute."""

    def __init__(
        self,
        event_type: str,
        *,
        message: _MockAnthropicMessage | None = None,
        delta: Any = None,
        usage: _MockAnthropicUsage | None = None,
    ) -> None:
        self.type = event_type
        if message is not None:
            self.message = message
        if delta is not None:
            self.delta = delta
        if usage is not None:
            self.usage = usage


class _MockTextDelta:
    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Mock classes for Gemini streaming chunks
# ---------------------------------------------------------------------------


class _MockGeminiUsageMetadata:
    def __init__(
        self,
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _MockGeminiPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockGeminiContent:
    def __init__(self, parts: list[_MockGeminiPart]) -> None:
        self.parts = parts


class _MockGeminiCandidate:
    def __init__(self, content: _MockGeminiContent) -> None:
        self.content = content


class _MockGeminiChunk:
    """Mimics a Gemini GenerateContentResponse streaming chunk."""

    def __init__(
        self,
        text: str | None = None,
        usage_metadata: _MockGeminiUsageMetadata | None = None,
        model_version: str | None = None,
    ) -> None:
        if text is not None:
            self.candidates = [_MockGeminiCandidate(_MockGeminiContent([_MockGeminiPart(text)]))]
        else:
            self.candidates = []
        self.usage_metadata = usage_metadata
        if model_version is not None:
            self.model_version = model_version


def _make_gemini_chunks(
    texts: list[str],
    final_usage: _MockGeminiUsageMetadata | None = None,
    model_version: str | None = None,
) -> list[_MockGeminiChunk]:
    """Build a list of Gemini-like chunks, with optional usage on the last chunk."""
    chunks = [_MockGeminiChunk(text=t, model_version=model_version) for t in texts]
    if final_usage is not None:
        chunks.append(
            _MockGeminiChunk(text=None, usage_metadata=final_usage, model_version=model_version)
        )
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_anthropic_events(
    model: str,
    input_tokens: int,
    texts: list[str],
    output_tokens: int | None = None,
) -> list[_MockAnthropicEvent]:
    """Build a list of Anthropic-like streaming events."""
    events: list[_MockAnthropicEvent] = [
        _MockAnthropicEvent(
            "message_start",
            message=_MockAnthropicMessage(
                model=model,
                usage=_MockAnthropicUsage(input_tokens=input_tokens),
            ),
        ),
    ]
    for t in texts:
        events.append(
            _MockAnthropicEvent(
                "content_block_delta",
                delta=_MockTextDelta(t),
            ),
        )
    if output_tokens is not None:
        events.append(
            _MockAnthropicEvent(
                "message_delta",
                usage=_MockAnthropicUsage(output_tokens=output_tokens),
            ),
        )
    events.append(_MockAnthropicEvent("message_stop"))
    return events


# ===========================================================================
# TestEstimateTokens
# ===========================================================================


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        text = "Hi"  # len 2, 2//4 = 0 -> max(1,0) = 1
        assert estimate_tokens(text) == max(1, len(text) // 4)

    def test_typical_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        result = estimate_tokens(text)
        assert result == max(1, len(text) // 4)
        # Sanity: should be a reasonable number
        assert 5 < result < 50


# ===========================================================================
# TestIsSyncStream
# ===========================================================================


class TestIsSyncStream:
    def test_generator_is_stream(self) -> None:
        def gen():
            yield 1

        assert _is_sync_stream(gen()) is True

    def test_plain_iterator_not_stream(self) -> None:
        # Plain iterators (iter(), map, filter) lack close() and are not streams
        assert _is_sync_stream(iter([1, 2, 3])) is False
        assert _is_sync_stream(map(str, [1])) is False

    def test_sdk_stream_like_object_is_stream(self) -> None:
        class _FakeSDKStream:
            def __iter__(self) -> _FakeSDKStream:
                return self

            def __next__(self) -> str:
                raise StopIteration

            def close(self) -> None:
                pass

        assert _is_sync_stream(_FakeSDKStream()) is True

    def test_string_not_stream(self) -> None:
        assert _is_sync_stream("hello") is False

    def test_bytes_not_stream(self) -> None:
        assert _is_sync_stream(b"hello") is False

    def test_dict_not_stream(self) -> None:
        assert _is_sync_stream({"key": "value"}) is False

    def test_list_not_stream(self) -> None:
        assert _is_sync_stream([1, 2, 3]) is False

    def test_none_not_stream(self) -> None:
        assert _is_sync_stream(None) is False

    def test_regular_object_not_stream(self) -> None:
        class Obj:
            pass

        assert _is_sync_stream(Obj()) is False


# ===========================================================================
# TestStreamAccumulatorOpenAI
# ===========================================================================


class TestStreamAccumulatorOpenAI:
    def test_accumulates_openai_chunks_with_usage(self) -> None:
        usage = _MockChunkUsage(prompt_tokens=50, completion_tokens=120)
        chunks = _make_openai_chunks("gpt-4o", ["Hello", " world", "!"], usage)

        acc = StreamAccumulator()
        for chunk in chunks:
            acc.process_chunk(chunk)

        result = acc.get_usage()
        assert result is not None
        model, inp, out = result
        assert model == "gpt-4o"
        assert inp == 50
        assert out == 120

    def test_accumulates_openai_chunks_without_usage(self) -> None:
        """No final usage chunk -> falls back to character-based estimation."""
        chunks = _make_openai_chunks("gpt-4o", ["Hello", " world", "!"])

        acc = StreamAccumulator()
        for chunk in chunks:
            acc.process_chunk(chunk)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = acc.get_usage()

        assert result is not None
        model, inp, out = result
        assert model == "gpt-4o"
        assert inp == 0  # unknown from stream alone
        assert out == estimate_tokens("Hello world!")
        # Should have issued a warning about estimation
        assert len(w) == 1
        assert "estimate" in str(w[0].message).lower()

    def test_empty_stream(self) -> None:
        acc = StreamAccumulator()
        assert acc.get_usage() is None

    def test_none_content_in_chunks(self) -> None:
        """Chunks with None content should not cause errors."""
        chunk_with_none = _MockOpenAIChunk(model="gpt-4o", content=None)
        usage = _MockChunkUsage(prompt_tokens=10, completion_tokens=20)
        final_chunk = _MockOpenAIChunk(model="gpt-4o", content=None, usage=usage)

        acc = StreamAccumulator()
        acc.process_chunk(chunk_with_none)
        acc.process_chunk(final_chunk)

        result = acc.get_usage()
        assert result is not None
        model, inp, out = result
        assert model == "gpt-4o"
        assert inp == 10
        assert out == 20


# ===========================================================================
# TestStreamAccumulatorAnthropic
# ===========================================================================


class TestStreamAccumulatorAnthropic:
    def test_accumulates_anthropic_events(self) -> None:
        events = _make_anthropic_events(
            model="claude-sonnet-4-20250514",
            input_tokens=200,
            texts=["Hello", " from", " Claude"],
            output_tokens=80,
        )

        acc = StreamAccumulator()
        for event in events:
            acc.process_chunk(event)

        result = acc.get_usage()
        assert result is not None
        model, inp, out = result
        assert model == "claude-sonnet-4-20250514"
        assert inp == 200
        assert out == 80

    def test_anthropic_missing_output_tokens(self) -> None:
        """message_delta with output_tokens omitted -> partial usage."""
        events = _make_anthropic_events(
            model="claude-sonnet-4-20250514",
            input_tokens=150,
            texts=["Partial"],
            output_tokens=None,  # no message_delta event appended
        )

        acc = StreamAccumulator()
        for event in events:
            acc.process_chunk(event)

        result = acc.get_usage()
        assert result is not None
        model, inp, out = result
        assert model == "claude-sonnet-4-20250514"
        assert inp == 150
        assert out == 0  # _has_api_usage is True from message_start, output defaults to 0


# ===========================================================================
# TestStreamAccumulatorGemini
# ===========================================================================


class TestStreamAccumulatorGemini:
    def test_accumulates_gemini_chunks_with_usage(self) -> None:
        usage = _MockGeminiUsageMetadata(prompt_token_count=150, candidates_token_count=60)
        chunks = _make_gemini_chunks(
            ["Hello", " from", " Gemini"],
            final_usage=usage,
            model_version="gemini-1.5-pro",
        )

        acc = StreamAccumulator()
        for chunk in chunks:
            acc.process_chunk(chunk)

        result = acc.get_usage()
        assert result is not None
        model, inp, out = result
        assert model == "gemini-1.5-pro"
        assert inp == 150
        assert out == 60

    def test_accumulates_gemini_chunks_without_usage(self) -> None:
        """No usage metadata on any chunk -> falls back to character-based estimation."""
        chunks = _make_gemini_chunks(
            ["Hello", " from", " Gemini"],
            model_version="gemini-1.5-pro",
        )

        acc = StreamAccumulator()
        for chunk in chunks:
            acc.process_chunk(chunk)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = acc.get_usage()

        assert result is not None
        model, inp, out = result
        assert model == "gemini-1.5-pro"
        assert inp == 0  # unknown from stream alone
        assert out == estimate_tokens("Hello from Gemini")
        # Should have issued a warning about estimation
        assert len(w) == 1
        assert "estimate" in str(w[0].message).lower()

    def test_gemini_chunks_without_model_version(self) -> None:
        """Chunks without model_version -> model is empty string."""
        usage = _MockGeminiUsageMetadata(prompt_token_count=50, candidates_token_count=25)
        chunks = _make_gemini_chunks(
            ["text"],
            final_usage=usage,
            model_version=None,  # no model_version attribute
        )

        acc = StreamAccumulator()
        for chunk in chunks:
            acc.process_chunk(chunk)

        result = acc.get_usage()
        assert result is not None
        model, inp, out = result
        assert model == ""
        assert inp == 50
        assert out == 25


# ===========================================================================
# TestWrapSyncStream
# ===========================================================================


class TestWrapSyncStream:
    """Tests for wrap_sync_stream."""

    def _make_deps(self, db_path: str) -> tuple[UsageStore, PricingRegistry, CostReporter]:
        store = UsageStore(db_path=db_path)
        registry = default_registry
        reporter = CostReporter()
        return store, registry, reporter

    def test_yields_all_chunks(self, tmp_db_path: str) -> None:
        store, registry, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=10, completion_tokens=20)
        chunks = _make_openai_chunks("gpt-4o", ["a", "b", "c"], usage)

        wrapped = wrap_sync_stream(
            iter(chunks),
            project="test",
            model_override=None,
            max_budget=None,
            store=store,
            registry=registry,
            reporter=reporter,
        )
        collected = list(wrapped)
        assert len(collected) == len(chunks)
        store.close()

    def test_logs_cost_after_exhaustion(self, tmp_db_path: str) -> None:
        store, registry, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hello", " world"], usage)

        wrapped = wrap_sync_stream(
            iter(chunks),
            project="proj",
            model_override=None,
            max_budget=None,
            store=store,
            registry=registry,
            reporter=reporter,
        )
        # Consume the stream
        for _ in wrapped:
            pass

        expected_cost = registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("proj") == pytest.approx(expected_cost)
        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4o"
        store.close()

    def test_logs_cost_on_early_break(self, tmp_db_path: str) -> None:
        store, registry, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["a", "b", "c", "d"], usage)

        wrapped = wrap_sync_stream(
            iter(chunks),
            project="proj",
            model_override=None,
            max_budget=None,
            store=store,
            registry=registry,
            reporter=reporter,
        )
        # Break after 2 chunks -- the finally block should still run
        count = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in wrapped:
                count += 1
                if count == 2:
                    break
            # Explicitly close the generator to trigger the finally block
            wrapped.close()

        # After early break, only partial text was accumulated (no API usage chunk seen).
        # The fallback estimation kicks in and logs usage based on accumulated text.
        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4o"
        store.close()

    def test_no_logging_for_empty_stream(self, tmp_db_path: str) -> None:
        store, registry, reporter = self._make_deps(tmp_db_path)

        def empty_gen():
            return
            yield

        wrapped = wrap_sync_stream(
            empty_gen(),
            project="proj",
            model_override=None,
            max_budget=None,
            store=store,
            registry=registry,
            reporter=reporter,
        )
        for _ in wrapped:
            pass

        assert store.get_total_cost("proj") == 0.0
        assert store.get_usage_logs("proj") == []
        store.close()

    def test_budget_enforcement_in_stream(self, tmp_db_path: str) -> None:
        store, registry, reporter = self._make_deps(tmp_db_path)

        # Pre-fill cost that already exceeds the budget
        store.log_usage("proj", "gpt-4o", 1000, 500, 5.0)

        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hello"], usage)

        # Set max_budget below the already-logged cost so the new call exceeds it
        wrapped = wrap_sync_stream(
            iter(chunks),
            project="proj",
            model_override=None,
            max_budget=0.001,  # far below the existing 5.0 cost
            store=store,
            registry=registry,
            reporter=reporter,
        )

        # Consume the stream -- budget exceeded but cost is still logged
        # because the stream was already consumed (tokens spent)
        with pytest.warns(match="exceeded budget"):
            for _ in wrapped:
                pass

        # Cost IS logged even though budget was exceeded (to keep totals accurate)
        logs = store.get_usage_logs("proj")
        assert len(logs) == 2  # pre-fill + the streaming call
        store.close()

    def test_model_override(self, tmp_db_path: str) -> None:
        store, registry, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=10, completion_tokens=20)
        chunks = _make_openai_chunks("gpt-4o", ["text"], usage)

        wrapped = wrap_sync_stream(
            iter(chunks),
            project="proj",
            model_override="custom-model",
            max_budget=None,
            store=store,
            registry=registry,
            reporter=reporter,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in wrapped:
                pass

        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        assert logs[0]["model"] == "custom-model"
        store.close()

    def test_mid_stream_exception_logs_partial_usage(self, tmp_db_path: str) -> None:
        """If the underlying iterator raises mid-stream, partial cost is logged."""
        store, registry, reporter = self._make_deps(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        good_chunks = _make_openai_chunks("gpt-4o", ["Hello", " world"], usage)

        def _failing_iter():
            yield good_chunks[0]
            yield good_chunks[1]
            raise RuntimeError("connection lost")

        wrapped = wrap_sync_stream(
            _failing_iter(),
            project="proj",
            model_override=None,
            max_budget=None,
            store=store,
            registry=registry,
            reporter=reporter,
        )

        collected = []
        with pytest.raises(RuntimeError, match="connection lost"):
            for chunk in wrapped:
                collected.append(chunk)

        assert len(collected) == 2
        # Partial usage: no final chunk with API usage, so falls back to estimation
        total = store.get_total_cost("proj")
        # Estimation from "Hello world" (11 chars // 4 = 2 tokens) at gpt-4o output rate
        assert total >= 0.0
        logs = store.get_usage_logs("proj")
        assert len(logs) == 1
        store.close()


# ===========================================================================
# TestDecoratorWithStreaming
# ===========================================================================


class TestDecoratorWithStreaming:
    """Integration tests: @track_costs on functions that return generators."""

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

    def test_decorated_generator_function(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)
        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hello", " world"], usage)

        @track_costs(project="stream-proj")
        def streaming_call() -> Any:
            yield from chunks

        result = streaming_call()
        # Result should be a generator wrapper; consuming it logs cost
        collected = list(result)
        assert len(collected) == len(chunks)

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("stream-proj") == pytest.approx(expected_cost)

    def test_decorated_streaming_with_budget(self, tmp_db_path: str) -> None:
        store = self._make_store(tmp_db_path)

        usage = _MockChunkUsage(prompt_tokens=100, completion_tokens=50)
        chunks = _make_openai_chunks("gpt-4o", ["Hi"], usage)

        @track_costs(project="budget-stream", max_budget=100.0)
        def streaming_call() -> Any:
            yield from chunks

        result = streaming_call()
        collected = list(result)
        assert len(collected) == len(chunks)

        expected_cost = default_registry.get_cost("gpt-4o", 100, 50)
        assert store.get_total_cost("budget-stream") == pytest.approx(expected_cost)
