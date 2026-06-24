from __future__ import annotations

import json
import logging

import openai

from taproot.providers.base import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"


def _translate_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _translate_messages(messages: list[dict], system: str | None) -> list[dict]:
    """Convert Anthropic-format messages to OpenAI format."""
    result: list[dict] = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "assistant":
            if isinstance(content, list):
                tool_calls = []
                text_parts = []
                for block in content:
                    if block["type"] == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        })
                    elif block["type"] == "text":
                        text_parts.append(block["text"])
                oai_msg: dict = {"role": "assistant"}
                if text_parts:
                    oai_msg["content"] = " ".join(text_parts)
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
            else:
                result.append({"role": "assistant", "content": content})
        elif role == "user":
            if isinstance(content, list):
                for block in content:
                    if block["type"] == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
                    elif block.get("type") == "text":
                        result.append({"role": "user", "content": block.get("text", "")})
            else:
                result.append({"role": "user", "content": content})
        else:
            result.append(msg)
    return result


class OpenAIProvider:
    provider_name = "openai"

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self.model_name = model
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        oai_messages = _translate_messages(messages, system)
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = _translate_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        # Normalise finish_reason → stop_reason
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
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
