from taproot.tools.analysis import analyze_ticket_cluster
from taproot.tools.problems import draft_problem_record, get_existing_problems
from taproot.tools.tickets import (
    fetch_tickets,
    get_ticket_details,
    search_similar_tickets,
)

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "fetch_tickets",
        "description": (
            "Retrieve incident tickets from the corpus. Use this first to load the tickets "
            "you will analyse. Filter by time window, service, or priority to focus the analysis. "
            "Start with a broad window (30-90 days) to surface recurring patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days back from today to fetch. Default 30.",
                },
                "service": {
                    "type": "string",
                    "description": "Filter to a specific service name. Omit to fetch all services.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "Filter by priority level. Omit to fetch all priorities.",
                },
                "category": {
                    "type": "string",
                    "description": "Filter by ticket category. Omit to fetch all categories.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_ticket_details",
        "description": (
            "Return the full details of a single incident ticket by its ID. "
            "Use this when you need to inspect a specific ticket's description, "
            "resolution notes, or tags in full before analysing it as part of a cluster."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID, e.g. INC-2026-0001.",
                }
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "search_similar_tickets",
        "description": (
            "Find tickets that are operationally similar to a given ticket using hybrid "
            "BM25 + semantic search with optional LLM reranking. Use this to discover clusters "
            "of tickets that likely share the same underlying root cause. The match_reason field "
            "explains why each result was considered similar. Call this for every ticket that "
            "looks like it could be part of a recurring pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ID of the ticket to use as the search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar tickets to return. Default 10.",
                },
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "get_existing_problems",
        "description": (
            "Return all existing problem records. Always call this before drafting a new "
            "problem record to check whether the pattern you have identified is already "
            "documented. Do not create duplicate problem records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "analyze_ticket_cluster",
        "description": (
            "Perform a deep structural analysis of a group of tickets to identify their common "
            "root cause. Provide a list of ticket IDs that you believe share the same underlying "
            "issue. Returns the probable root cause, contributing factors, affected services, "
            "and a suggested permanent fix. Call this before drafting a problem record to ensure "
            "your analysis is thorough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticket IDs that form the cluster to analyse.",
                }
            },
            "required": ["ticket_ids"],
        },
    },
    {
        "name": "draft_problem_record",
        "description": (
            "Create and persist a draft ITIL-compliant problem record based on your analysis. "
            "Only call this after you have (1) searched for similar tickets, (2) analysed the "
            "cluster with analyze_ticket_cluster, and (3) confirmed no duplicate problem record "
            "exists via get_existing_problems. Set confidence to HIGH only when the pattern is "
            "clear across 5+ tickets with consistent symptoms and resolution notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Concise title describing the recurring problem.",
                },
                "description": {
                    "type": "string",
                    "description": "Full description of the problem and its impact.",
                },
                "root_cause": {
                    "type": "string",
                    "description": "The underlying root cause identified from analysis.",
                },
                "contributing_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of factors that contribute to or exacerbate the problem.",
                },
                "affected_services": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of service names affected by this problem.",
                },
                "related_incident_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of incident ticket IDs that are instances of this problem.",
                },
                "suggested_permanent_fix": {
                    "type": "string",
                    "description": "Recommended permanent resolution for this problem.",
                },
                "workaround": {
                    "type": "string",
                    "description": "Interim workaround available until the permanent fix is implemented.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Priority of this problem record.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Confidence in the root cause analysis.",
                },
            },
            "required": [
                "title",
                "description",
                "root_cause",
                "contributing_factors",
                "affected_services",
                "related_incident_ids",
                "suggested_permanent_fix",
                "workaround",
                "priority",
                "confidence",
            ],
        },
    },
]

__all__ = [
    "TOOL_DEFINITIONS",
    "fetch_tickets",
    "get_ticket_details",
    "search_similar_tickets",
    "get_existing_problems",
    "draft_problem_record",
    "analyze_ticket_cluster",
]
