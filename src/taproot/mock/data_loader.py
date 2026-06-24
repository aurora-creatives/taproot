import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from taproot.models.problem import ProblemRecord
from taproot.models.ticket import Ticket

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "fixtures"


class MockDataLoader:
    """Loads and filters ticket/problem data from JSON fixtures."""

    def __init__(self, fixtures_dir: Path = _FIXTURES_DIR) -> None:
        self._fixtures_dir = fixtures_dir
        self._tickets: list[Ticket] | None = None
        self._problems: list[ProblemRecord] | None = None

    def get_tickets(
        self,
        days: int = 30,
        service: str | None = None,
        priority: str | None = None,
        category: str | None = None,
    ) -> list[Ticket]:
        """Return tickets from fixtures, applying the given filters in memory."""
        all_tickets = self._load_tickets()
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

        results = []
        for ticket in all_tickets:
            created = ticket.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                continue
            if service and ticket.service != service:
                continue
            if priority and ticket.priority.value != priority:
                continue
            if category and ticket.category != category:
                continue
            results.append(ticket)

        logger.debug("fetch_tickets returned %d tickets (days=%d)", len(results), days)
        return results

    def get_ticket_by_id(self, ticket_id: str) -> Ticket:
        """Return a single ticket by ID, raising ValueError if not found."""
        all_tickets = self._load_tickets()
        for ticket in all_tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        raise ValueError(f"Ticket '{ticket_id}' not found in fixtures")

    def get_all_tickets(self) -> list[Ticket]:
        """Return all tickets without filtering."""
        return self._load_tickets()

    def get_problems(self) -> list[ProblemRecord]:
        """Return all existing problem records from fixtures."""
        return self._load_problems()

    def _load_tickets(self) -> list[Ticket]:
        if self._tickets is None:
            path = self._fixtures_dir / "tickets.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._tickets = [Ticket.model_validate(item) for item in raw]
            logger.debug("Loaded %d tickets from %s", len(self._tickets), path)
        return self._tickets

    def _load_problems(self) -> list[ProblemRecord]:
        if self._problems is None:
            path = self._fixtures_dir / "problems.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._problems = [ProblemRecord.model_validate(item) for item in raw]
            logger.debug("Loaded %d problems from %s", len(self._problems), path)
        return self._problems
