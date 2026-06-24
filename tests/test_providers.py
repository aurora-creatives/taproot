import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from taproot.providers.base import LLMResponse


def _anthropic_text_response(text: str = "Done.") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "end_turn"
    return resp


def _anthropic_tool_response(tool_name: str, tool_id: str, tool_input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "tool_use"
    return resp


def _openai_text_response(text: str = "Done.") -> MagicMock:
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _openai_tool_response(tool_name: str, tool_id: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = tool_id
    tc.function = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(arguments)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        with patch("taproot.providers.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_anthropic_text_response("Analysis done."))

            from taproot.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-20250514")
            response = await provider.complete([{"role": "user", "content": "Hello"}])

        assert isinstance(response, LLMResponse)
        assert response.content == "Analysis done."
        assert response.stop_reason == "end_turn"
        assert response.provider == "anthropic"
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_complete_maps_tool_use_blocks(self) -> None:
        with patch("taproot.providers.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=_anthropic_tool_response("fetch_tickets", "tu_001", {"days": 30})
            )

            from taproot.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(api_key="test-key")
            response = await provider.complete([{"role": "user", "content": "Analyse"}])

        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "fetch_tickets"
        assert response.tool_calls[0].tool_use_id == "tu_001"
        assert response.tool_calls[0].input == {"days": 30}

    @pytest.mark.asyncio
    async def test_complete_simple_returns_string(self) -> None:
        with patch("taproot.providers.anthropic_provider.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_anthropic_text_response("Reranked."))

            from taproot.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(api_key="test-key")
            result = await provider.complete_simple("Rerank these tickets.")

        assert isinstance(result, str)
        assert result == "Reranked."


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=_openai_text_response("Done."))

            from taproot.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
            response = await provider.complete([{"role": "user", "content": "Hello"}])

        assert isinstance(response, LLMResponse)
        assert response.content == "Done."
        assert response.stop_reason == "end_turn"
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_complete_maps_tool_calls(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                return_value=_openai_tool_response("fetch_tickets", "call_001", {"days": 30})
            )

            from taproot.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="test-key")
            response = await provider.complete([{"role": "user", "content": "Analyse"}])

        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "fetch_tickets"
        assert response.tool_calls[0].input == {"days": 30}

    @pytest.mark.asyncio
    async def test_translates_anthropic_tool_format_to_openai(self) -> None:
        """complete() with Anthropic-format tools must pass OpenAI function format to the API."""
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=_openai_text_response())

            from taproot.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="test-key")

            anthropic_tools = [
                {
                    "name": "fetch_tickets",
                    "description": "Fetch tickets",
                    "input_schema": {
                        "type": "object",
                        "properties": {"days": {"type": "integer"}},
                        "required": [],
                    },
                }
            ]
            await provider.complete([{"role": "user", "content": "Go"}], tools=anthropic_tools)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        passed_tools = call_kwargs["tools"]
        assert passed_tools[0]["type"] == "function"
        assert "parameters" in passed_tools[0]["function"]
        assert "input_schema" not in passed_tools[0]["function"]

    @pytest.mark.asyncio
    async def test_finish_reason_stop_maps_to_end_turn(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=_openai_text_response())

            from taproot.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="test-key")
            response = await provider.complete([{"role": "user", "content": "Hi"}])

        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_simple_returns_string(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=_openai_text_response("Reranked."))

            from taproot.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="test-key")
            result = await provider.complete_simple("Rerank these.")

        assert isinstance(result, str)
        assert result == "Reranked."


class TestAzureOpenAIProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        with patch("taproot.providers.azure_openai_provider.openai.AsyncAzureOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=_openai_text_response("Azure done."))

            from taproot.providers.azure_openai_provider import AzureOpenAIProvider
            provider = AzureOpenAIProvider(
                endpoint="https://resource.openai.azure.com",
                api_key="key",
                deployment="gpt-4o",
            )
            response = await provider.complete([{"role": "user", "content": "Hello"}])

        assert isinstance(response, LLMResponse)
        assert response.provider == "azure_openai"
        assert response.stop_reason == "end_turn"

    def test_provider_name_is_azure_openai(self) -> None:
        with patch("taproot.providers.azure_openai_provider.openai.AsyncAzureOpenAI"):
            from taproot.providers.azure_openai_provider import AzureOpenAIProvider
            p = AzureOpenAIProvider("https://ep.azure.com", "key", "dep")
        assert p.provider_name == "azure_openai"


class TestScrubbedProvider:
    @pytest.mark.asyncio
    async def test_scrubs_content_before_calling_inner(self) -> None:
        from taproot.providers.scrubbed_provider import ScrubbedProvider
        from taproot.scrubbing.scrubber import DataScrubber

        inner = MagicMock()
        inner.provider_name = "openai"
        inner.model_name = "gpt-4o"

        captured: dict = {}

        async def mock_complete(messages, **kwargs):
            captured["messages"] = messages
            return LLMResponse(content="ok", stop_reason="end_turn", provider="openai", model="gpt-4o")

        inner.complete = mock_complete

        provider = ScrubbedProvider(inner, DataScrubber())
        messages = [{"role": "user", "content": "Contact alice@corp.com for info."}]
        await provider.complete(messages)

        assert "alice@corp.com" not in captured["messages"][0]["content"]
        assert "<<EMAIL_1>>" in captured["messages"][0]["content"]

    def test_provider_name_prefixed(self) -> None:
        from taproot.providers.scrubbed_provider import ScrubbedProvider
        from taproot.scrubbing.scrubber import DataScrubber

        inner = MagicMock()
        inner.provider_name = "anthropic"
        inner.model_name = "claude-sonnet-4-20250514"
        provider = ScrubbedProvider(inner, DataScrubber())
        assert provider.provider_name == "scrubbed_anthropic"
