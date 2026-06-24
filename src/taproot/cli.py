import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from taproot.config import get_settings
from taproot.models.problem import ProblemRecord, ProblemStatus
from taproot.tools.problems import get_existing_problems
from taproot.tools.tickets import fetch_tickets

app = typer.Typer(
    name="taproot",
    help="Finds the root beneath the noise. Surfaces ITSM problem records that should exist but don't.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger(__name__)


def _get_latest_draft_file(output_dir: Path) -> Path | None:
    """Return the most recently created drafts file in output_dir, or None."""
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("drafts_*.json"), reverse=True)
    return candidates[0] if candidates else None


@app.command()
def run(
    days: int = typer.Option(30, "--days", help="Number of days of ticket history to analyse."),
    service: Optional[str] = typer.Option(None, "--service", help="Filter analysis to a specific service."),
    max_records: int = typer.Option(10, "--max-records", help="Maximum number of problem records to draft."),
) -> None:
    """Run the agentic analysis to discover recurring incident patterns."""
    from taproot.agent import run_analysis

    settings = get_settings()
    settings.configure_logging()

    console.print(
        Panel(
            f"Analysing [bold]{days}[/bold] days of ticket history"
            + (f" for service [bold]{service}[/bold]" if service else "")
            + f"\nMax problem records: [bold]{max_records}[/bold]",
            title="[bold green]taproot run[/bold green]",
            border_style="green",
        )
    )

    try:
        summary = asyncio.run(run_analysis(days=days, service=service, max_problem_records=max_records))
    except ValueError as exc:
        err_console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[bold red]Analysis failed:[/bold red] {exc}")
        logger.exception("run_analysis failed")
        raise typer.Exit(code=1) from exc

    # Summary table
    table = Table(title="Analysis Complete", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Tickets analysed", str(summary.tickets_analysed))
    table.add_row("Clusters found", str(summary.clusters_found))
    table.add_row("Problem records drafted", str(summary.problem_records_drafted))
    table.add_row("Duration", f"{summary.analysis_duration_seconds:.1f}s")
    console.print(table)

    if not summary.drafted_records:
        console.print("[yellow]No problem records were drafted.[/yellow]")
        return

    # Save drafts to output dir
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    draft_path = settings.output_dir / f"drafts_{timestamp}.json"
    records_data = [r.model_dump(mode="json") for r in summary.drafted_records]
    draft_path.write_text(json.dumps(records_data, indent=2, default=str), encoding="utf-8")

    console.print(f"\n[bold green]Drafts saved to:[/bold green] {draft_path}")
    console.print("\nRun [bold]taproot review[/bold] to approve or reject draft records.")


@app.command()
def review(
    file: Optional[Path] = typer.Option(None, "--file", help="Path to a specific drafts JSON file."),
) -> None:
    """Interactively review draft problem records — approve, reject, or skip each one."""
    settings = get_settings()

    if file is None:
        file = _get_latest_draft_file(settings.output_dir)
        if file is None:
            err_console.print(
                "[bold red]No draft file found.[/bold red] "
                "Run [bold]taproot run[/bold] first to generate draft records."
            )
            raise typer.Exit(code=1)

    if not file.exists():
        err_console.print(f"[bold red]File not found:[/bold red] {file}")
        raise typer.Exit(code=1)

    raw = json.loads(file.read_text(encoding="utf-8"))
    records = [ProblemRecord.model_validate(r) for r in raw]

    if not records:
        console.print("[yellow]No draft records to review.[/yellow]")
        return

    console.print(
        Panel(
            f"Reviewing [bold]{len(records)}[/bold] draft problem records\n"
            f"Source: {file}",
            title="[bold blue]taproot review[/bold blue]",
            border_style="blue",
        )
    )

    approved: list[ProblemRecord] = []
    rejected: list[ProblemRecord] = []
    skipped: list[ProblemRecord] = []

    for i, record in enumerate(records, 1):
        console.print(f"\n[bold]Record {i} of {len(records)}[/bold]")
        _display_problem_record(record)

        while True:
            choice = Prompt.ask(
                r"[bold cyan]\[A][/bold cyan]pprove / "
                r"[bold red]\[R][/bold red]eject / "
                r"[bold yellow]\[E][/bold yellow]dit notes / "
                r"[bold white]\[S][/bold white]kip",
                default="S",
            ).strip().upper()

            if choice == "A":
                notes = Prompt.ask("Optional reviewer notes (press Enter to skip)", default="")
                updated = record.model_copy(
                    update={
                        "status": ProblemStatus.APPROVED,
                        "reviewed_at": datetime.now(tz=timezone.utc),
                        "reviewer_notes": notes,
                    }
                )
                approved.append(updated)
                console.print("[bold green]Approved.[/bold green]")
                break
            elif choice == "R":
                reason = Prompt.ask("Rejection reason (optional)", default="")
                updated = record.model_copy(
                    update={
                        "status": ProblemStatus.REJECTED,
                        "reviewed_at": datetime.now(tz=timezone.utc),
                        "reviewer_notes": reason,
                    }
                )
                rejected.append(updated)
                console.print("[bold red]Rejected.[/bold red]")
                break
            elif choice == "E":
                notes = Prompt.ask("Enter your notes for this record")
                record = record.model_copy(update={"reviewer_notes": notes})
                _display_problem_record(record)
            elif choice == "S":
                skipped.append(record)
                console.print("[yellow]Skipped.[/yellow]")
                break
            else:
                console.print("[yellow]Please enter A, R, E, or S.[/yellow]")

    # Summary
    summary_table = Table(title="Review Complete", show_header=True, header_style="bold")
    summary_table.add_column("Result", style="cyan")
    summary_table.add_column("Count", style="white")
    summary_table.add_row("Approved", str(len(approved)))
    summary_table.add_row("Rejected", str(len(rejected)))
    summary_table.add_row("Skipped", str(len(skipped)))
    console.print(summary_table)

    if approved:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        approved_path = settings.output_dir / f"approved_{timestamp}.json"
        approved_data = [r.model_dump(mode="json") for r in approved]
        approved_path.write_text(json.dumps(approved_data, indent=2, default=str), encoding="utf-8")
        console.print(f"\n[bold green]Approved records written to:[/bold green] {approved_path}")


def _display_problem_record(record: ProblemRecord) -> None:
    """Render a problem record as a rich panel in the terminal."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", justify="right", min_width=22)
    table.add_column(style="white")

    table.add_row("ID", record.problem_id)
    table.add_row("Title", record.title)
    table.add_row("Priority", record.priority.value.upper())
    table.add_row("Confidence", record.confidence)
    table.add_row("Status", record.status.value)
    table.add_row("Affected Services", ", ".join(record.affected_services))
    table.add_row("Related Incidents", ", ".join(record.related_incident_ids))
    table.add_row("", "")
    table.add_row("Description", record.description)
    table.add_row("Root Cause", record.root_cause)
    table.add_row(
        "Contributing Factors",
        "\n".join(f"• {f}" for f in record.contributing_factors),
    )
    table.add_row("Suggested Fix", record.suggested_permanent_fix)
    table.add_row("Workaround", record.workaround or "None")
    if record.reviewer_notes:
        table.add_row("Reviewer Notes", record.reviewer_notes)

    console.print(
        Panel(
            table,
            title=f"[bold]{record.problem_id}[/bold] — {record.title}",
            border_style="blue",
        )
    )


@app.command(name="list-tickets")
def list_tickets(
    days: int = typer.Option(30, "--days", help="Number of days back to fetch."),
    service: Optional[str] = typer.Option(None, "--service", help="Filter by service name."),
) -> None:
    """List incident tickets in the corpus matching the given filters."""
    settings = get_settings()
    settings.configure_logging()

    tickets = fetch_tickets(days=days, service=service)

    if not tickets:
        console.print("[yellow]No tickets found matching the specified filters.[/yellow]")
        return

    table = Table(
        title=f"Tickets — last {days} days" + (f" / {service}" if service else ""),
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", max_width=50)
    table.add_column("Service", style="magenta")
    table.add_column("Priority", style="yellow")
    table.add_column("Created", no_wrap=True)
    table.add_column("Status")

    for ticket in tickets:
        created = ticket.created_at.strftime("%Y-%m-%d")
        title_truncated = ticket.title[:47] + "..." if len(ticket.title) > 50 else ticket.title
        status_style = {
            "open": "red",
            "in_progress": "yellow",
            "resolved": "green",
            "closed": "dim",
        }.get(ticket.status.value, "white")
        table.add_row(
            ticket.ticket_id,
            title_truncated,
            ticket.service,
            ticket.priority.value,
            created,
            f"[{status_style}]{ticket.status.value}[/{status_style}]",
        )

    console.print(table)
    console.print(f"\nTotal: [bold]{len(tickets)}[/bold] tickets")


@app.command(name="list-problems")
def list_problems() -> None:
    """List all existing problem records."""
    settings = get_settings()
    settings.configure_logging()

    problems = get_existing_problems()

    if not problems:
        console.print("[yellow]No existing problem records found.[/yellow]")
        return

    table = Table(
        title="Problem Records",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", max_width=50)
    table.add_column("Priority", style="yellow")
    table.add_column("Status")
    table.add_column("Related Incidents", style="dim")

    for problem in problems:
        status_style = {
            "draft": "yellow",
            "approved": "green",
            "rejected": "red",
        }.get(problem.status.value, "white")
        table.add_row(
            problem.problem_id,
            problem.title[:47] + "..." if len(problem.title) > 50 else problem.title,
            problem.priority.value.upper(),
            f"[{status_style}]{problem.status.value}[/{status_style}]",
            str(len(problem.related_incident_ids)),
        )

    console.print(table)
    console.print(f"\nTotal: [bold]{len(problems)}[/bold] problem records")


def main() -> None:
    """Entry point for the taproot CLI."""
    app()


if __name__ == "__main__":
    main()
