from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_name: str
    tool_use_id: str
    input: dict


class LLMResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = []
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"
    provider: str
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    """
    Provider-agnostic interface for LLM completions.
    All taproot LLM calls go through this interface.
    Tools are always in Anthropic format — each provider translates internally.
    """

    provider_name: str
    model_name: str

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages (Anthropic format) to the LLM and return a structured response."""
        ...

    async def complete_simple(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Lightweight single-turn completion with no tool use. Returns plain text."""
        ...
