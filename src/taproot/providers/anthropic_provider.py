from __future__ import annotations

import logging

import anthropic

from taproot.providers.base import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self.model_name = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools  # already in Anthropic format

        response = await self._client.messages.create(**kwargs)

        tool_calls = [
            ToolCall(
                tool_name=block.name,
                tool_use_id=block.id,
                input=dict(block.input),
            )
            for block in response.content
            if block.type == "tool_use"
        ]
        content_text = " ".join(
            block.text for block in response.content if block.type == "text"
        )

        stop_reason = response.stop_reason or "end_turn"
        # Normalise stop reasons
        if stop_reason == "tool_use" and not tool_calls:
            stop_reason = "end_turn"

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            provider=self.provider_name,
            model=self.model_name,
        )

    async def complete_simple(self, prompt: str, max_tokens: int = 1024) -> str:
        response = await self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return " ".join(
            block.text for block in response.content if block.type == "text"
        )
