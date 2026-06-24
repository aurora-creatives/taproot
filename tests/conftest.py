from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from taproot.models.problem import ProblemPriority, ProblemRecord, ProblemStatus
from taproot.models.ticket import Ticket, TicketPriority, TicketStatus
from taproot.pageindex import PageIndex


@pytest.fixture
def sample_ticket() -> Ticket:
    """A single auth-service ticket matching Pattern 1."""
    return Ticket(
        ticket_id="INC-TEST-0001",
        title="Users unable to log in to portal",
        description="Multiple users reporting 401 errors after entering correct credentials. Session invalidation issue.",
        resolution_notes="Restarted auth service. Token expiry misconfiguration resolved.",
        service="user-auth-service",
        category="Authentication",
        priority=TicketPriority.P2,
        status=TicketStatus.CLOSED,
        reported_by="Test User",
        assigned_team="Platform Engineering",
        created_at=datetime(2024, 11, 1, 9, 0, tzinfo=timezone.utc),
        resolved_at=datetime(2024, 11, 1, 11, 0, tzinfo=timezone.utc),
        resolution_time_minutes=120,
        tags=["authentication", "login", "401-error", "session"],
    )


@pytest.fixture
def sample_ticket_list(sample_ticket: Ticket) -> list[Ticket]:
    """Five tickets: 3 from Pattern 1 (auth), 2 noise."""
    pattern_tickets = [
        sample_ticket,
        Ticket(
            ticket_id="INC-TEST-0002",
            title="Session keeps expiring during active work",
            description="Users getting logged out every 30 minutes. Session timeout misconfigured.",
            resolution_notes="Cleared session cache. Token TTL issue identified.",
            service="user-auth-service",
            category="Authentication",
            priority=TicketPriority.P3,
            status=TicketStatus.RESOLVED,
            reported_by="Nina Patel",
            assigned_team="Platform Engineering",
            created_at=datetime(2024, 11, 5, 11, 0, tzinfo=timezone.utc),
            resolved_at=datetime(2024, 11, 5, 15, 0, tzinfo=timezone.utc),
            resolution_time_minutes=240,
            tags=["authentication", "session-timeout", "token"],
        ),
        Ticket(
            ticket_id="INC-TEST-0003",
            title="Authentication failure — 401 after password change",
            description="User cannot log in after changing password. Session invalidation broken.",
            resolution_notes="Manual session invalidation. Token expiry configuration issue.",
            service="portal-frontend",
            category="Authentication",
            priority=TicketPriority.P3,
            status=TicketStatus.CLOSED,
            reported_by="Tom Hendricks",
            assigned_team="Platform Engineering",
            created_at=datetime(2024, 11, 10, 10, 0, tzinfo=timezone.utc),
            resolved_at=datetime(2024, 11, 10, 13, 0, tzinfo=timezone.utc),
            resolution_time_minutes=180,
            tags=["authentication", "password-change", "session"],
        ),
    ]
    noise_tickets = [
        Ticket(
            ticket_id="INC-TEST-0004",
            title="Network outage — Manchester office",
            description="Complete network outage affecting the Manchester office. ISP fibre cut.",
            resolution_notes="ISP restored connectivity after 4 hours.",
            service="network-infrastructure",
            category="Network",
            priority=TicketPriority.P1,
            status=TicketStatus.CLOSED,
            reported_by="Office Manager",
            assigned_team="Network Operations",
            created_at=datetime(2024, 10, 20, 8, 0, tzinfo=timezone.utc),
            resolved_at=datetime(2024, 10, 20, 12, 0, tzinfo=timezone.utc),
            resolution_time_minutes=240,
            tags=["network", "outage", "one-off"],
        ),
        Ticket(
            ticket_id="INC-TEST-0005",
            title="Disk space warning on archive server",
            description="Archive storage server at 92% capacity.",
            resolution_notes="Deleted archives older than 18 months.",
            service="archive-storage",
            category="Infrastructure",
            priority=TicketPriority.P4,
            status=TicketStatus.CLOSED,
            reported_by="Automated Monitoring",
            assigned_team="Infrastructure",
            created_at=datetime(2024, 11, 5, 2, 0, tzinfo=timezone.utc),
            resolved_at=datetime(2024, 11, 5, 10, 0, tzinfo=timezone.utc),
            resolution_time_minutes=480,
            tags=["disk-space", "archive", "one-off"],
        ),
    ]
    return pattern_tickets + noise_tickets


@pytest.fixture
def sample_problem_record() -> ProblemRecord:
    """A ProblemRecord in DRAFT status."""
    return ProblemRecord(
        problem_id="PRB1700000001",
        title="Auth service token expiry misconfiguration",
        description="Recurring authentication failures caused by token lifecycle management issues.",
        root_cause="JWT token TTL is misconfigured, causing premature session expiry.",
        contributing_factors=["No distributed token store", "TTL hardcoded in v2.3.1"],
        affected_services=["user-auth-service", "portal-frontend"],
        related_incident_ids=["INC-2026-0001", "INC-2026-0002", "INC-2026-0003"],
        suggested_permanent_fix="Implement configurable token TTL with sliding session expiry.",
        workaround="Users log out and back in; IT can manually clear sessions.",
        priority=ProblemPriority.HIGH,
        status=ProblemStatus.DRAFT,
        confidence="HIGH",
        created_at=datetime(2024, 11, 26, 12, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_settings():
    """Settings mock with openai as provider (avoids needing Anthropic key)."""
    s = MagicMock()
    s.LLM_MODE = "single"
    s.LLM_PROVIDER = "openai"
    s.LLM_MODEL = "gpt-4o"
    s.openai_api_key = "test-key"
    s.ANTHROPIC_API_KEY = ""
    s.ENABLE_SCRUBBING = False
    s.use_mock_data = True
    s.PAGEINDEX_USE_SEMANTIC = False
    s.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    s.log_level = "WARNING"
    s.configure_logging = MagicMock()
    return s


@pytest.fixture
def page_index(sample_ticket_list: list[Ticket]) -> PageIndex:
    """A built PageIndex (BM25 only, no semantic to avoid model downloads)."""
    index = PageIndex(use_semantic=False)
    index.build(sample_ticket_list)
    return index
