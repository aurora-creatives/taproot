import logging
from collections import Counter

from taproot.mock.data_loader import MockDataLoader

logger = logging.getLogger(__name__)

_loader = MockDataLoader()


def analyze_ticket_cluster(ticket_ids: list[str]) -> dict:
    """
    Perform deep analysis of a group of tickets to identify common root cause.

    Reads full ticket content for all provided IDs.
    Returns a dict with:
      - common_symptoms: list of shared symptoms across tickets
      - probable_root_cause: best hypothesis for underlying cause
      - confidence: HIGH | MEDIUM | LOW
      - affected_services: list of services appearing across tickets
      - pattern_description: plain-language description of the pattern
      - suggested_fix: recommended permanent resolution
      - workaround: interim workaround if applicable
    """
    if not ticket_ids:
        return {
            "common_symptoms": [],
            "probable_root_cause": "No tickets provided for analysis",
            "confidence": "LOW",
            "affected_services": [],
            "pattern_description": "No pattern identified — empty ticket list",
            "suggested_fix": "",
            "workaround": "",
        }

    tickets = []
    for tid in ticket_ids:
        try:
            tickets.append(_loader.get_ticket_by_id(tid))
        except ValueError:
            logger.warning("Ticket %s not found during cluster analysis — skipping", tid)

    if not tickets:
        return {
            "common_symptoms": [],
            "probable_root_cause": "No valid tickets found for the provided IDs",
            "confidence": "LOW",
            "affected_services": [],
            "pattern_description": "Analysis could not proceed — no valid tickets",
            "suggested_fix": "",
            "workaround": "",
        }

    # Aggregate services
    service_counts = Counter(t.service for t in tickets)
    affected_services = [svc for svc, _ in service_counts.most_common()]

    # Aggregate categories
    category_counts = Counter(t.category for t in tickets)
    dominant_category = category_counts.most_common(1)[0][0]

    # Aggregate tags for symptom extraction
    all_tags: list[str] = []
    for t in tickets:
        all_tags.extend(t.tags)
    tag_counts = Counter(all_tags)
    common_symptoms = [tag for tag, count in tag_counts.most_common(6) if count >= 2]

    # Build a plain-language pattern description from titles
    titles = [t.title for t in tickets]
    short_titles = "; ".join(titles[:4])
    if len(titles) > 4:
        short_titles += f" ... and {len(titles) - 4} more"

    # Derive confidence from cluster size and symptom overlap
    if len(tickets) >= 6 and len(common_symptoms) >= 3:
        confidence = "HIGH"
    elif len(tickets) >= 3 and len(common_symptoms) >= 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Derive probable root cause from resolution notes heuristics
    all_resolution_notes = " ".join(t.resolution_notes.lower() for t in tickets)
    root_cause = _infer_root_cause(all_resolution_notes, dominant_category, affected_services)

    suggested_fix = _suggest_fix(root_cause, dominant_category)
    workaround = _suggest_workaround(dominant_category)

    pattern_description = (
        f"A cluster of {len(tickets)} tickets across services "
        f"{', '.join(affected_services[:3])} sharing category '{dominant_category}'. "
        f"Recurring incidents: {short_titles}."
    )

    logger.info(
        "Cluster analysis: %d tickets, confidence=%s, services=%s",
        len(tickets),
        confidence,
        affected_services,
    )

    return {
        "common_symptoms": common_symptoms,
        "probable_root_cause": root_cause,
        "confidence": confidence,
        "affected_services": affected_services,
        "pattern_description": pattern_description,
        "suggested_fix": suggested_fix,
        "workaround": workaround,
    }


def _infer_root_cause(resolution_text: str, category: str, services: list[str]) -> str:
    """Infer root cause from aggregated resolution note text and service/category context."""
    service_str = services[0] if services else "unknown service"

    if "token" in resolution_text and "auth" in resolution_text:
        return (
            "Authentication token lifecycle management is misconfigured in the "
            f"{service_str}. Tokens expire prematurely, are not refreshed correctly, "
            "or are not propagated consistently across service instances."
        )
    if "timeout" in resolution_text or "slow" in resolution_text or "query" in resolution_text:
        return (
            f"The {service_str} is executing unoptimised database queries that exceed "
            "acceptable time limits under normal load. Missing indexes, stale materialized "
            "views, or lack of query resource limits are likely contributing."
        )
    if "queue" in resolution_text or "email" in resolution_text or "delay" in resolution_text:
        return (
            f"The {service_str} email delivery queue is processing messages too slowly, "
            "causing significant delivery delays. The queue may be single-threaded, "
            "lack dedicated capacity for high-priority messages, or be competing with "
            "bulk sends for the same resources."
        )
    if category == "Authentication":
        return f"Recurring authentication failures in {service_str} due to session management issues."
    if category == "Performance":
        return f"Performance degradation in {service_str} under normal operational load."
    if category == "Notifications":
        return f"Email notification delivery delays in {service_str}."
    return f"Recurring incidents in {service_str} with category '{category}'."


def _suggest_fix(root_cause: str, category: str) -> str:
    """Suggest a permanent fix based on the inferred root cause."""
    if "token" in root_cause.lower() or category == "Authentication":
        return (
            "Implement a proper token lifecycle management system: (1) make token TTL "
            "configurable via environment variable with a sensible default, (2) implement "
            "sliding session expiry that resets on activity, (3) add a distributed token "
            "store (Redis) to ensure consistent state across all service instances, "
            "(4) add automated alerts when token expiry rates exceed baseline."
        )
    if "query" in root_cause.lower() or category == "Performance":
        return (
            "Conduct a query performance audit: (1) add database query execution plan analysis "
            "for all report queries taking >1s, (2) implement query timeout limits at the "
            "application level (not just the gateway), (3) schedule materialized view refreshes "
            "aligned with data ingestion cadence, (4) add a dedicated connection pool for "
            "long-running reports to prevent blocking interactive queries."
        )
    if "queue" in root_cause.lower() or category == "Notifications":
        return (
            "Separate transactional and bulk email queues: (1) implement dedicated high-priority "
            "queue for P1/P2 alerts and password resets, (2) add rate limiting on bulk sends, "
            "(3) implement dead-letter queue with retry logic and alerting on queue depth, "
            "(4) add SLA-based monitoring for time-to-deliver on critical notification types."
        )
    return "Investigate root cause thoroughly and implement a permanent architectural fix."


def _suggest_workaround(category: str) -> str:
    """Suggest an interim workaround for the given category."""
    workarounds = {
        "Authentication": "Users can resolve individual incidents by logging out and back in, or IT support can manually clear the affected user's session token.",
        "Performance": "Users can narrow date ranges or apply more specific filters to reduce query scope. For exports, request during off-peak hours.",
        "Notifications": "Critical notifications should be verified through an alternative channel (Slack, phone) until the email queue issue is permanently resolved.",
    }
    return workarounds.get(category, "Contact IT support for assistance.")
