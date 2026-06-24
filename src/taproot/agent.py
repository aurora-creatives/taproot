import json
import logging
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from taproot.config import get_settings
from taproot.mock.data_loader import MockDataLoader
from taproot.models.problem import AnalysisSummary
from taproot.pageindex import PageIndex
from taproot.providers import LLMRouter, TaskType
from taproot.tools import (
    TOOL_DEFINITIONS,
    analyze_ticket_cluster,
    draft_problem_record,
    fetch_tickets,
    get_existing_problems,
    get_ticket_details,
    search_similar_tickets,
)
from taproot.tools.problems import clear_draft_store, get_draft_store
from taproot.tools.tickets import set_page_index

logger = logging.getLogger(__name__)
console = Console()

_SYSTEM_PROMPT = """\
You are an expert ITSM problem manager with deep knowledge of ITIL problem management practices.

Your mission is to analyse a corpus of historical incident tickets and surface recurring patterns \
that have not been formally documented as problem records.

Follow this methodology:
1. Fetch tickets first — use a broad time window (60-90 days) to capture enough history.
2. Review the tickets and identify candidates that look like they could be part of a recurring pattern.
3. For each candidate, use search_similar_tickets to find related tickets.
4. When you have identified a cluster of 3+ tickets that appear to share a root cause, \
use analyze_ticket_cluster to perform a deep analysis.
5. Before drafting a problem record, always call get_existing_problems to check for duplicates.
6. Draft a problem record only when: the cluster is genuine, the analysis is thorough, \
and no duplicate exists.
7. Continue until you have reviewed the full corpus or reached the maximum problem record limit.

Quality standards:
- Only draft problem records for genuine recurring patterns (3+ tickets, clear common cause).
- Set confidence to HIGH only when evidence is strong and consistent across 5+ tickets.
- Do not draft records for one-off incidents, even if serious.
- Each problem record must have a specific, actionable suggested permanent fix.
- Be precise: noise tickets exist in the corpus — do not force patterns where none exist.
"""

_TOOL_DISPATCH: dict[str, Any] = {
    "fetch_tickets": fetch_tickets,
    "get_ticket_details": get_ticket_details,
    "search_similar_tickets": search_similar_tickets,
    "get_existing_problems": get_existing_problems,
    "analyze_ticket_cluster": analyze_ticket_cluster,
    "draft_problem_record": draft_problem_record,
}

_MAX_ITERATIONS = 20


def _execute_tool(name: str, tool_input: dict) -> Any:
    """Dispatch a tool call and return a JSON-serialisable result."""
    fn = _TOOL_DISPATCH.get(name)
    if fn is None:
        raise ValueError(f"Unknown tool: {name}")
    result = fn(**tool_input)
    if isinstance(result, list):
        return [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in result]
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result


def _make_status_panel(
    current_tool: str,
    tickets_seen: int,
    records_drafted: int,
    elapsed: float,
    iteration: int,
    model_info: str = "",
) -> Panel:
    """Build the Rich live-progress panel displayed during agent execution."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")
    if model_info:
        table.add_row("Model", model_info)
    table.add_row("Tool", current_tool or "—")
    table.add_row("Tickets seen", str(tickets_seen))
    table.add_row("Records drafted", str(records_drafted))
    table.add_row("Iteration", f"{iteration} / {_MAX_ITERATIONS}")
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    return Panel(table, title="[bold green]taproot — agent running[/bold green]", border_style="green")


async def run_analysis(
    days: int = 30,
    service: str | None = None,
    max_problem_records: int = 10,
) -> AnalysisSummary:
    """
    Run the agentic ITSM analysis loop.

    Fetches tickets, identifies recurring patterns, and drafts problem records.
    Returns an AnalysisSummary when the loop completes.
    """
    settings = get_settings()
    settings.configure_logging()

    router = LLMRouter(settings)
    analysis_provider = router.get(TaskType.ANALYSIS)
    draft_provider = router.get(TaskType.DRAFT)
    rerank_provider = router.get(TaskType.RERANK)

    # Build PageIndex and inject into the tools layer
    page_index = PageIndex(
        use_semantic=settings.PAGEINDEX_USE_SEMANTIC,
        embedding_model=settings.PAGEINDEX_EMBEDDING_MODEL,
        rerank_provider=rerank_provider,
    )
    all_tickets = MockDataLoader().get_all_tickets()
    page_index.build(all_tickets)
    set_page_index(page_index)

    clear_draft_store()

    # Build model info string for the status panel
    if router.mode == "single":
        model_info = f"{analysis_provider.model_name} (single)"
    else:
        model_info = f"analysis={analysis_provider.model_name} / draft={draft_provider.model_name}"

    user_message = (
        f"Analyse the incident ticket history for the last {days} days"
        + (f" for service '{service}'" if service else "")
        + f". Identify recurring patterns and draft up to {max_problem_records} "
        "ITIL-compliant problem records. Be thorough — look at all services and "
        "categories. Noise tickets (one-offs) should be ignored."
    )

    # Messages in Anthropic format (system passed separately to complete())
    messages: list[dict] = [{"role": "user", "content": user_message}]

    start_time = time.monotonic()
    iteration = 0
    current_tool = ""
    tickets_seen = 0
    records_drafted = 0

    # Start with analysis provider; switch to draft after draft_problem_record calls
    current_provider = analysis_provider

    with Live(
        _make_status_panel(current_tool, tickets_seen, records_drafted, 0.0, iteration, model_info),
        console=console,
        refresh_per_second=4,
    ) as live:
        while iteration < _MAX_ITERATIONS:
            iteration += 1
            elapsed = time.monotonic() - start_time
            live.update(
                _make_status_panel(current_tool, tickets_seen, records_drafted, elapsed, iteration, model_info)
            )

            response = await current_provider.complete(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                system=_SYSTEM_PROMPT,
                max_tokens=4096,
            )

            logger.debug("Iteration %d: stop_reason=%s", iteration, response.stop_reason)

            # Build assistant content block in Anthropic format
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.tool_use_id,
                    "name": tc.tool_name,
                    "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

            tool_results: list[dict] = []
            hit_limit = False
            used_draft = False

            for tc in response.tool_calls:
                current_tool = tc.tool_name
                elapsed = time.monotonic() - start_time
                live.update(
                    _make_status_panel(current_tool, tickets_seen, records_drafted, elapsed, iteration, model_info)
                )
                logger.debug("Tool call: %s", tc.tool_name)

                try:
                    result = _execute_tool(tc.tool_name, tc.input)

                    if tc.tool_name == "fetch_tickets" and isinstance(result, list):
                        tickets_seen = max(tickets_seen, len(result))
                    elif tc.tool_name == "draft_problem_record":
                        records_drafted = len(get_draft_store())
                        used_draft = True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.tool_use_id,
                        "content": json.dumps(result, default=str),
                    })

                    if records_drafted >= max_problem_records:
                        hit_limit = True

                except Exception as exc:
                    logger.error("Tool %s failed: %s", tc.tool_name, exc)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.tool_use_id,
                        "content": json.dumps({"error": str(exc)}),
                    })

            messages.append({"role": "user", "content": tool_results})

            # Route next iteration: draft provider after a drafting turn
            current_provider = draft_provider if used_draft else analysis_provider

            if hit_limit:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have drafted {records_drafted} problem records, "
                        f"which is the maximum allowed ({max_problem_records}). "
                        "Please conclude your analysis now."
                    ),
                })

        elapsed = time.monotonic() - start_time
        live.update(
            _make_status_panel("complete", tickets_seen, records_drafted, elapsed, iteration, model_info)
        )

    drafted = get_draft_store()
    return AnalysisSummary(
        tickets_analysed=tickets_seen,
        clusters_found=records_drafted,
        problem_records_drafted=records_drafted,
        duplicate_patterns_skipped=0,
        analysis_duration_seconds=time.monotonic() - start_time,
        drafted_records=drafted,
    )
