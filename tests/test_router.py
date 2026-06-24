from unittest.mock import MagicMock, patch

import pytest

from taproot.providers.router import LLMRouter, TaskType
from taproot.providers.scrubbed_provider import ScrubbedProvider


def _make_settings(
    mode: str = "single",
    provider: str = "openai",
    openai_key: str = "test-key",
    enable_scrubbing: bool = False,
    **kwargs,
):
    """Build a Settings-like mock for router tests."""
    s = MagicMock()
    s.LLM_MODE = mode
    s.LLM_PROVIDER = provider
    s.LLM_MODEL = "gpt-4o"
    s.LLM_RERANK_PROVIDER = kwargs.get("rerank_provider", "openai")
    s.LLM_RERANK_MODEL = kwargs.get("rerank_model", "gpt-4o-mini")
    s.LLM_ANALYSIS_PROVIDER = kwargs.get("analysis_provider", "openai")
    s.LLM_ANALYSIS_MODEL = kwargs.get("analysis_model", "gpt-4o")
    s.LLM_DRAFT_PROVIDER = kwargs.get("draft_provider", "openai")
    s.LLM_DRAFT_MODEL = kwargs.get("draft_model", "gpt-4o")
    s.openai_api_key = openai_key
    s.ANTHROPIC_API_KEY = kwargs.get("anthropic_key", "")
    s.AZURE_OPENAI_ENDPOINT = kwargs.get("azure_endpoint", "")
    s.AZURE_OPENAI_API_KEY = kwargs.get("azure_key", "")
    s.AZURE_OPENAI_DEPLOYMENT = kwargs.get("azure_deployment", "")
    s.AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
    s.AWS_BEDROCK_REGION = "us-east-1"
    s.ENABLE_SCRUBBING = enable_scrubbing
    return s


class TestLLMRouterSingleMode:
    def test_single_mode_returns_same_instance_for_all_tasks(self) -> None:
        """In single mode, all three task types return the exact same provider instance."""
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(mode="single", provider="openai"))
        p_rerank = router.get(TaskType.RERANK)
        p_analysis = router.get(TaskType.ANALYSIS)
        p_draft = router.get(TaskType.DRAFT)
        assert p_rerank is p_analysis
        assert p_analysis is p_draft

    def test_mode_property_returns_single(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(mode="single"))
        assert router.mode == "single"

    def test_single_mode_provider_name(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(mode="single", provider="openai"))
        assert router.get(TaskType.ANALYSIS).provider_name == "openai"


class TestLLMRouterMultiMode:
    def _multi_settings(self, **kwargs):
        return _make_settings(
            mode="multi",
            rerank_provider="openai",
            rerank_model="gpt-4o-mini",
            analysis_provider="openai",
            analysis_model="gpt-4o",
            draft_provider="openai",
            draft_model="gpt-4o",
            **kwargs,
        )

    def test_multi_mode_returns_different_instances(self) -> None:
        """In multi mode, each task type should return a distinct provider instance."""
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(self._multi_settings())
        p_rerank = router.get(TaskType.RERANK)
        p_analysis = router.get(TaskType.ANALYSIS)
        p_draft = router.get(TaskType.DRAFT)
        # Each is a separate instance even if same provider type
        assert p_rerank is not p_analysis
        assert p_analysis is not p_draft

    def test_mode_property_returns_multi(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(self._multi_settings())
        assert router.mode == "multi"

    def test_rerank_uses_configured_model(self) -> None:
        s = _make_settings(
            mode="multi",
            rerank_provider="openai", rerank_model="gpt-4o-mini",
            analysis_provider="openai", analysis_model="gpt-4o",
            draft_provider="openai", draft_model="gpt-4o",
        )
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(s)
        assert router.get(TaskType.RERANK).model_name == "gpt-4o-mini"

    def test_analysis_uses_configured_model(self) -> None:
        s = _make_settings(
            mode="multi",
            rerank_provider="openai", rerank_model="gpt-4o-mini",
            analysis_provider="openai", analysis_model="gpt-4o",
            draft_provider="openai", draft_model="gpt-4o",
        )
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(s)
        assert router.get(TaskType.ANALYSIS).model_name == "gpt-4o"


class TestLLMRouterScrubbing:
    def test_scrubbing_wraps_provider_in_single_mode(self) -> None:
        """When ENABLE_SCRUBBING=True, get() returns a ScrubbedProvider."""
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(mode="single", enable_scrubbing=True))
        provider = router.get(TaskType.ANALYSIS)
        assert isinstance(provider, ScrubbedProvider)

    def test_scrubbing_wraps_all_providers_in_multi_mode(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(
                mode="multi",
                rerank_provider="openai", rerank_model="gpt-4o-mini",
                analysis_provider="openai", analysis_model="gpt-4o",
                draft_provider="openai", draft_model="gpt-4o",
                enable_scrubbing=True,
            ))
        for task in TaskType:
            assert isinstance(router.get(task), ScrubbedProvider)

    def test_scrubbed_provider_name_contains_inner_name(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(mode="single", provider="openai", enable_scrubbing=True))
        provider = router.get(TaskType.ANALYSIS)
        assert "openai" in provider.provider_name

    def test_no_scrubbing_by_default(self) -> None:
        with patch("taproot.providers.openai_provider.openai.AsyncOpenAI"):
            router = LLMRouter(_make_settings(mode="single", enable_scrubbing=False))
        assert not isinstance(router.get(TaskType.ANALYSIS), ScrubbedProvider)


class TestLLMRouterUnknownProvider:
    def test_unknown_provider_raises_value_error(self) -> None:
        s = _make_settings(mode="single", provider="unknown_provider")
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMRouter(s)

    def test_error_names_the_config_field(self) -> None:
        s = _make_settings(mode="single", provider="nonexistent")
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            LLMRouter(s)
