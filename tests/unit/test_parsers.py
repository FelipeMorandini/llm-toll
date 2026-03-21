"""Unit tests for SDK parsers and auto-detection."""

from __future__ import annotations

from llm_toll.parsers import auto_detect_usage
from llm_toll.parsers.anthropic import parse_anthropic_response
from llm_toll.parsers.gemini import parse_gemini_response
from llm_toll.parsers.openai import parse_openai_response


class _MockUsageOpenAI:
    def __init__(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockOpenAIResponse:
    def __init__(
        self, model: str, usage: _MockUsageOpenAI | None, choices: list | None = None
    ) -> None:
        self.model = model
        self.usage = usage
        self.choices = choices if choices is not None else []


class _MockUsageAnthropic:
    def __init__(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockAnthropicResponse:
    def __init__(
        self, model: str, usage: _MockUsageAnthropic | None, stop_reason: str = "end_turn"
    ) -> None:
        self.model = model
        self.usage = usage
        self.stop_reason = stop_reason
        self.content = []


class TestParsers:
    """Tests for individual SDK parsers returning None for arbitrary objects."""

    def test_parse_openai_response_returns_none(self) -> None:
        assert parse_openai_response({"not": "an openai response"}) is None

    def test_parse_anthropic_response_returns_none(self) -> None:
        assert parse_anthropic_response({"not": "an anthropic response"}) is None

    def test_parse_gemini_response_returns_none(self) -> None:
        assert parse_gemini_response({"not": "a gemini response"}) is None

    def test_auto_detect_usage_returns_none(self) -> None:
        assert auto_detect_usage({"arbitrary": "object"}) is None

    def test_auto_detect_usage_short_circuits_on_match(self) -> None:
        """When a parser returns a result, auto_detect_usage returns it immediately."""
        from unittest.mock import patch

        expected = ("gpt-4o", 100, 50)
        # Patch the source module so the imported reference in the tuple is replaced
        with patch(
            "llm_toll.parsers.openai.parse_openai_response", return_value=expected
        ) as mock_parser:
            # Re-import to pick up the patched function
            import importlib

            import llm_toll.parsers

            importlib.reload(llm_toll.parsers)
            from llm_toll.parsers import auto_detect_usage as reloaded_auto_detect

            result = reloaded_auto_detect({"mock": "response"})
        assert result == expected
        mock_parser.assert_called_once()


class TestOpenAIParser:
    """Tests for the OpenAI SDK response parser."""

    def test_happy_path(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=100, completion_tokens=50)
        resp = _MockOpenAIResponse(model="gpt-4o", usage=usage)
        result = parse_openai_response(resp)
        assert result == ("gpt-4o", 100, 50)

    def test_usage_none(self) -> None:
        resp = _MockOpenAIResponse(model="gpt-4o", usage=None)
        assert parse_openai_response(resp) is None

    def test_missing_choices(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=10, completion_tokens=5)
        resp = type("Obj", (), {"model": "gpt-4o", "usage": usage})()
        assert parse_openai_response(resp) is None

    def test_missing_model(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=10, completion_tokens=5)
        resp = type("Obj", (), {"usage": usage, "choices": []})()
        assert parse_openai_response(resp) is None

    def test_missing_usage(self) -> None:
        resp = type("Obj", (), {"model": "gpt-4o", "choices": []})()
        assert parse_openai_response(resp) is None

    def test_prompt_tokens_none(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=None, completion_tokens=50)
        resp = _MockOpenAIResponse(model="gpt-4o", usage=usage)
        result = parse_openai_response(resp)
        assert result == ("gpt-4o", 0, 50)

    def test_completion_tokens_none(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=100, completion_tokens=None)
        resp = _MockOpenAIResponse(model="gpt-4o", usage=usage)
        result = parse_openai_response(resp)
        assert result == ("gpt-4o", 100, 0)

    def test_both_tokens_zero(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=0, completion_tokens=0)
        resp = _MockOpenAIResponse(model="gpt-4o", usage=usage)
        result = parse_openai_response(resp)
        assert result == ("gpt-4o", 0, 0)

    def test_dict_input(self) -> None:
        assert parse_openai_response({"model": "gpt-4o", "usage": {}}) is None

    def test_string_input(self) -> None:
        assert parse_openai_response("not a response") is None

    def test_none_input(self) -> None:
        assert parse_openai_response(None) is None


class TestAnthropicParser:
    """Tests for the Anthropic SDK response parser."""

    def test_happy_path(self) -> None:
        usage = _MockUsageAnthropic(input_tokens=200, output_tokens=80)
        resp = _MockAnthropicResponse(model="claude-sonnet-4-20250514", usage=usage)
        result = parse_anthropic_response(resp)
        assert result == ("claude-sonnet-4-20250514", 200, 80)

    def test_usage_none(self) -> None:
        resp = _MockAnthropicResponse(model="claude-sonnet-4-20250514", usage=None)
        assert parse_anthropic_response(resp) is None

    def test_missing_stop_reason(self) -> None:
        usage = _MockUsageAnthropic(input_tokens=10, output_tokens=5)
        resp = type(
            "Obj", (), {"model": "claude-sonnet-4-20250514", "usage": usage, "content": []}
        )()
        assert parse_anthropic_response(resp) is None

    def test_missing_content(self) -> None:
        usage = _MockUsageAnthropic(input_tokens=10, output_tokens=5)
        resp = type(
            "Obj",
            (),
            {"model": "claude-sonnet-4-20250514", "usage": usage, "stop_reason": "end_turn"},
        )()
        assert parse_anthropic_response(resp) is None

    def test_missing_usage(self) -> None:
        resp = type(
            "Obj",
            (),
            {"model": "claude-sonnet-4-20250514", "stop_reason": "end_turn", "content": []},
        )()
        assert parse_anthropic_response(resp) is None

    def test_input_tokens_none(self) -> None:
        usage = _MockUsageAnthropic(input_tokens=None, output_tokens=80)
        resp = _MockAnthropicResponse(model="claude-sonnet-4-20250514", usage=usage)
        result = parse_anthropic_response(resp)
        assert result == ("claude-sonnet-4-20250514", 0, 80)

    def test_output_tokens_none(self) -> None:
        usage = _MockUsageAnthropic(input_tokens=200, output_tokens=None)
        resp = _MockAnthropicResponse(model="claude-sonnet-4-20250514", usage=usage)
        result = parse_anthropic_response(resp)
        assert result == ("claude-sonnet-4-20250514", 200, 0)

    def test_openai_response_rejected(self) -> None:
        usage = _MockUsageOpenAI(prompt_tokens=100, completion_tokens=50)
        resp = _MockOpenAIResponse(model="gpt-4o", usage=usage)
        assert parse_anthropic_response(resp) is None

    def test_anthropic_response_rejected_by_openai(self) -> None:
        usage = _MockUsageAnthropic(input_tokens=200, output_tokens=80)
        resp = _MockAnthropicResponse(model="claude-sonnet-4-20250514", usage=usage)
        assert parse_openai_response(resp) is None


class TestAutoDetectIntegration:
    """Integration tests for auto_detect_usage with mock SDK responses."""

    @staticmethod
    def _fresh_auto_detect() -> object:
        """Return a freshly-reloaded auto_detect_usage to avoid stale refs."""
        import importlib

        import llm_toll.parsers

        importlib.reload(llm_toll.parsers)
        return llm_toll.parsers.auto_detect_usage

    def test_openai_response_auto_detected(self) -> None:
        detect = self._fresh_auto_detect()
        usage = _MockUsageOpenAI(prompt_tokens=100, completion_tokens=50)
        resp = _MockOpenAIResponse(model="gpt-4o", usage=usage)
        result = detect(resp)
        assert result == ("gpt-4o", 100, 50)

    def test_anthropic_response_auto_detected(self) -> None:
        detect = self._fresh_auto_detect()
        usage = _MockUsageAnthropic(input_tokens=200, output_tokens=80)
        resp = _MockAnthropicResponse(model="claude-sonnet-4-20250514", usage=usage)
        result = detect(resp)
        assert result == ("claude-sonnet-4-20250514", 200, 80)

    def test_unknown_object_returns_none(self) -> None:
        detect = self._fresh_auto_detect()
        assert detect({"unknown": "object"}) is None
