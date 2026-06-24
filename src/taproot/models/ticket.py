from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class TicketPriority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Ticket(BaseModel):
    """Represents a single ITSM incident ticket."""

    ticket_id: str
    title: str
    description: str
    resolution_notes: str = ""
    service: str
    category: str
    priority: TicketPriority
    status: TicketStatus
    reported_by: str
    assigned_team: str
    created_at: datetime
    resolved_at: datetime | None = None
    resolution_time_minutes: int | None = None
    tags: list[str] = []
