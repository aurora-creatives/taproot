from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from taproot.models.problem import (
    AnalysisSummary,
    ProblemPriority,
    ProblemRecord,
    ProblemStatus,
)
from taproot.models.ticket import Ticket, TicketPriority, TicketStatus


class TestTicketModel:
    def test_valid_ticket_parses(self):
        """A fully valid ticket dict should parse without error."""
        ticket = Ticket(
            ticket_id="INC-TEST-001",
            title="Test incident",
            description="Something broke",
            service="my-service",
            category="Performance",
            priority=TicketPriority.P2,
            status=TicketStatus.OPEN,
            reported_by="Alice",
            assigned_team="Backend",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert ticket.ticket_id == "INC-TEST-001"
        assert ticket.priority == TicketPriority.P2

    def test_invalid_priority_raises(self):
        """An invalid priority value should raise ValidationError."""
        with pytest.raises(ValidationError):
            Ticket(
                ticket_id="INC-TEST-001",
                title="Test",
                description="Test",
                service="svc",
                category="cat",
                priority="P9",  # invalid
                status=TicketStatus.OPEN,
                reported_by="Alice",
                assigned_team="Team",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_invalid_status_raises(self):
        """An invalid status value should raise ValidationError."""
        with pytest.raises(ValidationError):
            Ticket(
                ticket_id="INC-TEST-001",
                title="Test",
                description="Test",
                service="svc",
                category="cat",
                priority=TicketPriority.P1,
                status="broken",  # invalid
                reported_by="Alice",
                assigned_team="Team",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_defaults_applied(self):
        """Optional fields should use defaults when omitted."""
        ticket = Ticket(
            ticket_id="INC-TEST-001",
            title="Test",
            description="Test",
            service="svc",
            category="cat",
            priority=TicketPriority.P3,
            status=TicketStatus.OPEN,
            reported_by="Bob",
            assigned_team="Ops",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert ticket.resolution_notes == ""
        assert ticket.tags == []
        assert ticket.resolved_at is None
        assert ticket.resolution_time_minutes is None

    def test_datetime_parses_iso_string(self):
        """created_at should accept an ISO 8601 string."""
        ticket = Ticket.model_validate(
            {
                "ticket_id": "INC-TEST-001",
                "title": "Test",
                "description": "Test",
                "service": "svc",
                "category": "cat",
                "priority": "P3",
                "status": "open",
                "reported_by": "Alice",
                "assigned_team": "Team",
                "created_at": "2024-11-01T09:18:00Z",
            }
        )
        assert isinstance(ticket.created_at, datetime)
        assert ticket.created_at.year == 2024


class TestProblemRecordModel:
    def test_status_defaults_to_draft(self):
        """ProblemRecord.status should default to DRAFT."""
        record = ProblemRecord(
            problem_id="PRB001",
            title="Test problem",
            description="desc",
            root_cause="cause",
            contributing_factors=[],
            affected_services=["svc"],
            related_incident_ids=[],
            suggested_permanent_fix="fix",
            priority=ProblemPriority.MEDIUM,
            confidence="HIGH",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert record.status == ProblemStatus.DRAFT

    def test_workaround_defaults_to_empty_string(self):
        """workaround should default to empty string."""
        record = ProblemRecord(
            problem_id="PRB001",
            title="Test",
            description="desc",
            root_cause="cause",
            contributing_factors=[],
            affected_services=["svc"],
            related_incident_ids=[],
            suggested_permanent_fix="fix",
            priority=ProblemPriority.LOW,
            confidence="LOW",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert record.workaround == ""

    def test_invalid_priority_raises(self):
        """An invalid priority value should raise ValidationError."""
        with pytest.raises(ValidationError):
            ProblemRecord(
                problem_id="PRB001",
                title="Test",
                description="desc",
                root_cause="cause",
                contributing_factors=[],
                affected_services=[],
                related_incident_ids=[],
                suggested_permanent_fix="fix",
                priority="critical",  # invalid
                confidence="HIGH",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_reviewer_notes_defaults_empty(self):
        """reviewer_notes should default to empty string."""
        record = ProblemRecord(
            problem_id="PRB001",
            title="Test",
            description="desc",
            root_cause="cause",
            contributing_factors=[],
            affected_services=[],
            related_incident_ids=[],
            suggested_permanent_fix="fix",
            priority=ProblemPriority.HIGH,
            confidence="HIGH",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert record.reviewer_notes == ""
        assert record.reviewed_at is None


class TestAnalysisSummary:
    def test_aggregates_correctly(self, sample_problem_record):
        """AnalysisSummary should store drafted_records and expose counts."""
        summary = AnalysisSummary(
            tickets_analysed=35,
            clusters_found=3,
            problem_records_drafted=3,
            duplicate_patterns_skipped=1,
            analysis_duration_seconds=42.5,
            drafted_records=[sample_problem_record],
        )
        assert summary.tickets_analysed == 35
        assert summary.problem_records_drafted == 3
        assert len(summary.drafted_records) == 1
        assert summary.drafted_records[0].problem_id == "PRB1700000001"

    def test_empty_drafted_records(self):
        """AnalysisSummary with no records should be valid."""
        summary = AnalysisSummary(
            tickets_analysed=10,
            clusters_found=0,
            problem_records_drafted=0,
            duplicate_patterns_skipped=0,
            analysis_duration_seconds=5.0,
            drafted_records=[],
        )
        assert summary.drafted_records == []
