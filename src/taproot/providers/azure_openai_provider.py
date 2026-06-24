from __future__ import annotations

import json
import logging

import openai

from taproot.providers.base import LLMResponse, ToolCall
from taproot.providers.openai_provider import _translate_messages, _translate_tools

logger = logging.getLogger(__name__)


class AzureOpenAIProvider:
    """
    Azure OpenAI Service — recommended for regulated and enterprise environments.
    Data stays within the customer's Azure tenant. Satisfies GDPR, SOC 2, HIPAA BAA.
    """

    provider_name = "azure_openai"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-12-01-preview",
    ) -> None:
        self.model_name = deployment
        self._deployment = deployment
        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        oai_messages = _translate_messages(messages, system)
        kwargs: dict = {
            "model": self._deployment,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = _translate_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        stop_reason = "end_turn"
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    tool_name=tc.function.name,
                    tool_use_id=tc.id,
                    input=json.loads(tc.function.arguments),
                ))

        content_text = choice.message.content or ""

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            provider=self.provider_name,
            model=self.model_name,
        )

    async def complete_simple(self, prompt: str, max_tokens: int = 1024) -> str:
        response = await self._client.chat.completions.create(
            model=self._deployment,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
