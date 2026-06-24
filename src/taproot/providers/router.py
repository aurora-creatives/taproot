from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from taproot.providers.anthropic_provider import AnthropicProvider
from taproot.providers.azure_openai_provider import AzureOpenAIProvider
from taproot.providers.base import LLMProvider
from taproot.providers.bedrock_provider import AWSBedrockProvider
from taproot.providers.openai_provider import OpenAIProvider
from taproot.providers.scrubbed_provider import ScrubbedProvider

if TYPE_CHECKING:
    from taproot.config import Settings
    from taproot.scrubbing.scrubber import DataScrubber

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """The three distinct LLM tasks taproot performs."""

    RERANK = "rerank"
    ANALYSIS = "analysis"
    DRAFT = "draft"


class LLMRouter:
    """
    Routes LLM calls to the appropriate provider based on task type.

    LLM_MODE=single (default):
        All tasks use the same provider and model.

    LLM_MODE=multi:
        Each task gets its own configured provider and model.
        - rerank   → small, fast, cheap model
        - analysis → large, capable model
        - draft    → mid-tier, structured output
    """

    def __init__(self, settings: Settings) -> None:
        self._mode = settings.LLM_MODE

        # Lazy import to avoid circular imports
        from taproot.scrubbing.scrubber import DataScrubber

        scrubber: DataScrubber | None = DataScrubber() if settings.ENABLE_SCRUBBING else None

        if settings.LLM_MODE == "single":
            provider = _build_provider(settings, task=None)
            self._single = _maybe_wrap(provider, scrubber)
        else:
            self._providers: dict[TaskType, LLMProvider] = {
                TaskType.RERANK: _maybe_wrap(_build_provider(settings, task=TaskType.RERANK), scrubber),
                TaskType.ANALYSIS: _maybe_wrap(_build_provider(settings, task=TaskType.ANALYSIS), scrubber),
                TaskType.DRAFT: _maybe_wrap(_build_provider(settings, task=TaskType.DRAFT), scrubber),
            }

    def get(self, task: TaskType) -> LLMProvider:
        """Return the provider for the given task."""
        if self._mode == "single":
            return self._single
        return self._providers[task]

    @property
    def mode(self) -> str:
        return self._mode


def _build_provider(settings: Settings, task: TaskType | None) -> LLMProvider:
    """Build a provider instance for the given task. task=None means single mode."""
    if task is None:
        provider_name = settings.LLM_PROVIDER
        model = settings.LLM_MODEL
    else:
        prefix = task.value.upper()
        provider_name = getattr(settings, f"LLM_{prefix}_PROVIDER")
        model = getattr(settings, f"LLM_{prefix}_MODEL")

    match provider_name:
        case "anthropic":
            return AnthropicProvider(settings.ANTHROPIC_API_KEY, model)
        case "openai":
            return OpenAIProvider(settings.openai_api_key, model)
        case "azure_openai":
            return AzureOpenAIProvider(
                settings.AZURE_OPENAI_ENDPOINT,
                settings.AZURE_OPENAI_API_KEY,
                settings.AZURE_OPENAI_DEPLOYMENT,
                settings.AZURE_OPENAI_API_VERSION,
            )
        case "aws_bedrock":
            return AWSBedrockProvider(settings.AWS_BEDROCK_REGION, model)
        case _:
            task_label = f"LLM_{task.value.upper()}_PROVIDER" if task else "LLM_PROVIDER"
            raise ValueError(
                f"Unknown provider '{provider_name}' in {task_label}. "
                f"Valid options: anthropic, openai, azure_openai, aws_bedrock"
            )


def _maybe_wrap(provider: LLMProvider, scrubber: DataScrubber | None) -> LLMProvider:
    """Wrap provider in ScrubbedProvider if scrubbing is enabled."""
    if scrubber is None:
        return provider
    return ScrubbedProvider(provider, scrubber)
