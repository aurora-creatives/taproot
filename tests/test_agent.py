from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from taproot.models.problem import AnalysisSummary
from taproot.providers.base import LLMResponse, ToolCall


def _make_llm_response(
    stop_reason: str = "end_turn",
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    provider: str = "openai",
    model: str = "gpt-4o",
) -> LLMResponse:
    """Build a mock LLMResponse."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        provider=provider,
        model=model,
    )


def _make_tool_call(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(tool_name=name, tool_use_id=call_id, input=arguments)


def _make_mock_router(analysis_responses: list, draft_responses: list | None = None):
    """Create a mock LLMRouter with preconfigured responses."""
    analysis_provider = AsyncMock()
    analysis_provider.provider_name = "openai"
    analysis_provider.model_name = "gpt-4o"

    draft_provider = AsyncMock()
    draft_provider.provider_name = "openai"
    draft_provider.model_name = "gpt-4o"

    rerank_provider = MagicMock()
    rerank_provider.provider_name = "openai"
    rerank_provider.model_name = "gpt-4o-mini"

    analysis_provider.complete = AsyncMock(side_effect=analysis_responses)
    draft_provider.complete = AsyncMock(
        side_effect=draft_responses or [_make_llm_response(content="Draft done.")]
    )

    router = MagicMock()
    router.mode = "single"

    def get_provider(task):
        from taproot.providers.router import TaskType
        if task == TaskType.RERANK:
            return rerank_provider
        if task == TaskType.DRAFT:
            return draft_provider
        return analysis_provider

    router.get = MagicMock(side_effect=get_provider)
    return router, analysis_provider, draft_provider, rerank_provider


@pytest.mark.asyncio
async def test_agent_handles_tool_use_then_end_turn():
    """
    Agent loop should execute a tool_use response, append results,
    then stop on end_turn. Returns an AnalysisSummary.
    """
    from taproot.agent import run_analysis
    from taproot.tools.problems import clear_draft_store

    clear_draft_store()

    tool_call = _make_tool_call("tc_001", "fetch_tickets", {"days": 30})
    analysis_responses = [
        _make_llm_response("tool_use", tool_calls=[tool_call]),
        _make_llm_response("end_turn", content="Analysis complete. No patterns found."),
    ]

    router, analysis_provider, _, _ = _make_mock_router(analysis_responses)

    with patch("taproot.agent.LLMRouter", return_value=router), \
         patch("taproot.agent.get_settings") as mock_settings, \
         patch("taproot.agent.PageIndex") as MockPageIndex, \
         patch("taproot.agent.MockDataLoader"), \
         patch("taproot.agent.set_page_index"), \
         patch("taproot.tools.tickets.get_settings") as mock_tool_settings:

        mock_settings.return_value.LLM_MODE = "single"
        mock_settings.return_value.PAGEINDEX_USE_SEMANTIC = False
        mock_settings.return_value.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        mock_settings.return_value.configure_logging = MagicMock()
        mock_tool_settings.return_value.use_mock_data = True

        MockPageIndex.return_value.build = MagicMock()
        MockPageIndex.return_value.search = MagicMock(return_value=[])

        summary = await run_analysis(days=30, max_problem_records=5)

    assert isinstance(summary, AnalysisSummary)
    assert analysis_provider.complete.call_count == 2


@pytest.mark.asyncio
async def test_agent_stops_at_max_iterations():
    """Agent loop should stop after max_iterations even if the model keeps requesting tools."""
    from taproot.agent import _MAX_ITERATIONS, run_analysis
    from taproot.tools.problems import clear_draft_store

    clear_draft_store()

    tool_call = _make_tool_call("tc_001", "fetch_tickets", {"days": 30})
    always_tool = _make_llm_response("tool_use", tool_calls=[tool_call])

    analysis_responses = [always_tool] * (_MAX_ITERATIONS + 5)
    router, analysis_provider, _, _ = _make_mock_router(analysis_responses)

    with patch("taproot.agent.LLMRouter", return_value=router), \
         patch("taproot.agent.get_settings") as mock_settings, \
         patch("taproot.agent.PageIndex") as MockPageIndex, \
         patch("taproot.agent.MockDataLoader"), \
         patch("taproot.agent.set_page_index"), \
         patch("taproot.tools.tickets.get_settings") as mock_tool_settings:

        mock_settings.return_value.LLM_MODE = "single"
        mock_settings.return_value.PAGEINDEX_USE_SEMANTIC = False
        mock_settings.return_value.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        mock_settings.return_value.configure_logging = MagicMock()
        mock_tool_settings.return_value.use_mock_data = True

        MockPageIndex.return_value.build = MagicMock()
        MockPageIndex.return_value.search = MagicMock(return_value=[])

        summary = await run_analysis(days=30, max_problem_records=5)

    assert isinstance(summary, AnalysisSummary)
    assert analysis_provider.complete.call_count <= _MAX_ITERATIONS


@pytest.mark.asyncio
async def test_agent_returns_analysis_summary_on_completion():
    """run_analysis() should always return an AnalysisSummary regardless of content."""
    from taproot.agent import run_analysis
    from taproot.tools.problems import clear_draft_store

    clear_draft_store()

    analysis_responses = [_make_llm_response("end_turn", content="No recurring patterns found.")]
    router, _, _, _ = _make_mock_router(analysis_responses)

    with patch("taproot.agent.LLMRouter", return_value=router), \
         patch("taproot.agent.get_settings") as mock_settings, \
         patch("taproot.agent.PageIndex") as MockPageIndex, \
         patch("taproot.agent.MockDataLoader"), \
         patch("taproot.agent.set_page_index"):

        mock_settings.return_value.LLM_MODE = "single"
        mock_settings.return_value.PAGEINDEX_USE_SEMANTIC = False
        mock_settings.return_value.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        mock_settings.return_value.configure_logging = MagicMock()

        MockPageIndex.return_value.build = MagicMock()

        summary = await run_analysis(days=30)

    assert isinstance(summary, AnalysisSummary)
    assert isinstance(summary.drafted_records, list)
    assert summary.analysis_duration_seconds >= 0


@pytest.mark.asyncio
async def test_agent_routes_draft_task_to_draft_provider():
    """When draft_problem_record tool is called, the NEXT iteration uses draft_provider."""
    from taproot.agent import run_analysis
    from taproot.tools.problems import clear_draft_store

    clear_draft_store()

    draft_tc = _make_tool_call("tc_draft", "draft_problem_record", {
        "title": "Auth token expiry",
        "description": "Recurring auth failure",
        "root_cause": "JWT TTL misconfiguration",
        "contributing_factors": ["No token refresh"],
        "affected_services": ["user-auth-service"],
        "related_incident_ids": ["INC-2026-0001"],
        "suggested_permanent_fix": "Increase TTL",
        "workaround": "Re-login",
        "priority": "high",
        "confidence": "HIGH",
    })

    analysis_responses = [
        _make_llm_response("tool_use", tool_calls=[draft_tc]),
    ]
    draft_responses = [
        _make_llm_response("end_turn", content="Problem record drafted successfully."),
    ]

    router, analysis_provider, draft_provider, _ = _make_mock_router(
        analysis_responses, draft_responses
    )

    with patch("taproot.agent.LLMRouter", return_value=router), \
         patch("taproot.agent.get_settings") as mock_settings, \
         patch("taproot.agent.PageIndex") as MockPageIndex, \
         patch("taproot.agent.MockDataLoader"), \
         patch("taproot.agent.set_page_index"), \
         patch("taproot.tools.tickets.get_settings") as mock_tool_settings:

        mock_settings.return_value.LLM_MODE = "single"
        mock_settings.return_value.PAGEINDEX_USE_SEMANTIC = False
        mock_settings.return_value.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        mock_settings.return_value.configure_logging = MagicMock()
        mock_tool_settings.return_value.use_mock_data = True

        MockPageIndex.return_value.build = MagicMock()

        await run_analysis(days=30, max_problem_records=1)

    from taproot.providers.router import TaskType
    # Verify router.get was called with DRAFT task
    draft_calls = [c for c in router.get.call_args_list if c.args and c.args[0] == TaskType.DRAFT]
    assert len(draft_calls) >= 1, "router.get(TaskType.DRAFT) should have been called"
    # The draft_provider.complete should have been invoked
    assert draft_provider.complete.call_count >= 1
