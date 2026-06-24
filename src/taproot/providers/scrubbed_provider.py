from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from taproot.providers.base import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from taproot.scrubbing.scrubber import DataScrubber

logger = logging.getLogger(__name__)


class ScrubbedProvider:
    """
    Wraps any LLMProvider with a local PII scrubbing layer.
    Ticket content is anonymised before any LLM call.
    Placeholders are restored in the response before returning.
    Enable with: ENABLE_SCRUBBING=true in .env
    """

    def __init__(self, provider: LLMProvider, scrubber: DataScrubber) -> None:
        self._provider = provider
        self._scrubber = scrubber
        self.provider_name = f"scrubbed_{provider.provider_name}"
        self.model_name = provider.model_name

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> LLMResponse:
        scrubbed_messages, mapping = self._scrubber.scrub_messages(messages)
        scrubbed_system = None
        if system:
            scrubbed_system, sys_mapping = self._scrubber.scrub_text(system)
            mapping.update(sys_mapping)

        response = await self._provider.complete(
            scrubbed_messages,
            tools=tools,
            max_tokens=max_tokens,
            system=scrubbed_system,
        )

        # Restore placeholders in response text
        restored_content = self._scrubber.restore_text(response.content, mapping)
        return response.model_copy(update={"content": restored_content})

    async def complete_simple(self, prompt: str, max_tokens: int = 1024) -> str:
        scrubbed_prompt, mapping = self._scrubber.scrub_text(prompt)
        result = await self._provider.complete_simple(scrubbed_prompt, max_tokens)
        return self._scrubber.restore_text(result, mapping)
