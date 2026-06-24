import logging
from datetime import datetime, timezone

from taproot.config import get_settings
from taproot.mock.data_loader import MockDataLoader
from taproot.models.problem import ProblemPriority, ProblemRecord, ProblemStatus

logger = logging.getLogger(__name__)

_loader = MockDataLoader()

# In-session store for draft records created during the current run
_draft_store: list[ProblemRecord] = []


def get_existing_problems() -> list[ProblemRecord]:
    """
    Return all existing problem records.

    Used by the agent to avoid creating duplicate problem records.
    In mock mode: loads from data/fixtures/problems.json.
    """
    settings = get_settings()
    if not settings.use_mock_data:
        raise NotImplementedError(
            "Real ITSM integration is not yet implemented. Set USE_MOCK_DATA=true."
        )
    return _loader.get_problems()


def draft_problem_record(
    title: str,
    description: str,
    root_cause: str,
    contributing_factors: list[str],
    affected_services: list[str],
    related_incident_ids: list[str],
    suggested_permanent_fix: str,
    workaround: str,
    priority: str,
    confidence: str,
) -> ProblemRecord:
    """
    Create and persist a draft problem record.

    Assigns a generated problem_id (format: PRB{timestamp}).
    Saves to an in-session store accessible to the CLI review command.
    Returns the created ProblemRecord.
    """
    now = datetime.now(tz=timezone.utc)
    problem_id = f"PRB{int(now.timestamp())}"

    record = ProblemRecord(
        problem_id=problem_id,
        title=title,
        description=description,
        root_cause=root_cause,
        contributing_factors=contributing_factors,
        affected_services=affected_services,
        related_incident_ids=related_incident_ids,
        suggested_permanent_fix=suggested_permanent_fix,
        workaround=workaround,
        priority=ProblemPriority(priority.lower()),
        status=ProblemStatus.DRAFT,
        confidence=confidence.upper(),
        created_at=now,
    )

    _draft_store.append(record)
    logger.info("Drafted problem record %s: %s", problem_id, title)
    return record


def get_draft_store() -> list[ProblemRecord]:
    """Return all problem records drafted in the current session."""
    return list(_draft_store)


def clear_draft_store() -> None:
    """Clear the in-session draft store (used between runs)."""
    _draft_store.clear()
