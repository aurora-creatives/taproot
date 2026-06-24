from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3

from taproot.providers.base import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def _translate_tools_bedrock(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to Bedrock converse tool format."""
    return [
        {
            "toolSpec": {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": {"json": t.get("input_schema", {"type": "object", "properties": {}})},
            }
        }
        for t in tools
    ]


def _translate_messages_bedrock(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-format messages to Bedrock converse format."""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "content": [{"text": content}]})
        elif isinstance(content, list):
            bedrock_content: list[dict] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    bedrock_content.append({"text": block.get("text", "")})
                elif btype == "tool_use":
                    bedrock_content.append({
                        "toolUse": {
                            "toolUseId": block["id"],
                            "name": block["name"],
                            "input": block["input"],
                        }
                    })
                elif btype == "tool_result":
                    bedrock_content.append({
                        "toolResult": {
                            "toolUseId": block["tool_use_id"],
                            "content": [{"text": block["content"]}],
                        }
                    })
            if bedrock_content:
                result.append({"role": role, "content": bedrock_content})
        else:
            result.append({"role": role, "content": [{"text": str(content)}]})
    return result


class AWSBedrockProvider:
    """
    AWS Bedrock — recommended for regulated and enterprise environments.
    Data stays within the customer's AWS account.
    Uses the Bedrock converse API for tool use across all supported models.
    """

    provider_name = "aws_bedrock"

    def __init__(self, region: str, model_id: str = _DEFAULT_MODEL_ID) -> None:
        self.model_name = model_id
        self._model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def _sync_complete(self, kwargs: dict) -> dict[str, Any]:
        return self._client.converse(**kwargs)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        bedrock_messages = _translate_messages_bedrock(messages)
        kwargs: dict = {
            "modelId": self._model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {"tools": _translate_tools_bedrock(tools)}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._sync_complete(kwargs))

        stop_reason_raw = response.get("stopReason", "end_turn")
        stop_reason_map = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}
        stop_reason = stop_reason_map.get(stop_reason_raw, "end_turn")

        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        tool_calls = []
        text_parts = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    tool_name=tu["name"],
                    tool_use_id=tu["toolUseId"],
                    input=tu.get("input", {}),
                ))

        return LLMResponse(
            content=" ".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            provider=self.provider_name,
            model=self.model_name,
        )

    async def complete_simple(self, prompt: str, max_tokens: int = 1024) -> str:
        kwargs: dict = {
            "modelId": self._model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._sync_complete(kwargs))
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        return " ".join(b["text"] for b in content_blocks if "text" in b)
