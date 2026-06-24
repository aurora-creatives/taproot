from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ProblemPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProblemStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class ProblemRecord(BaseModel):
    """An ITIL-compliant problem record identifying a recurring root cause."""

    problem_id: str
    title: str
    description: str
    root_cause: str
    contributing_factors: list[str]
    affected_services: list[str]
    related_incident_ids: list[str]
    suggested_permanent_fix: str
    workaround: str = ""
    priority: ProblemPriority
    status: ProblemStatus = ProblemStatus.DRAFT
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    created_at: datetime
    reviewed_at: datetime | None = None
    reviewer_notes: str = ""


class AnalysisSummary(BaseModel):
    """Summary of a completed taproot analysis run."""

    tickets_analysed: int
    clusters_found: int
    problem_records_drafted: int
    duplicate_patterns_skipped: int
    analysis_duration_seconds: float
    drafted_records: list[ProblemRecord]
