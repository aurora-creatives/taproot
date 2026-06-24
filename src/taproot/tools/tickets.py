import logging

from taproot.config import get_settings
from taproot.mock.data_loader import MockDataLoader
from taproot.models.ticket import Ticket
from taproot.pageindex import PageIndex

logger = logging.getLogger(__name__)

_loader = MockDataLoader()
_page_index: PageIndex | None = None


def set_page_index(index: PageIndex) -> None:
    """Inject a pre-built PageIndex (used by the agent to share its configured index)."""
    global _page_index
    _page_index = index


def _get_page_index() -> PageIndex:
    """Return a lazily built PageIndex over the full ticket corpus."""
    global _page_index
    if _page_index is None:
        settings = get_settings()
        _page_index = PageIndex(
            use_semantic=settings.PAGEINDEX_USE_SEMANTIC,
            embedding_model=settings.PAGEINDEX_EMBEDDING_MODEL,
        )
        all_tickets = _loader.get_all_tickets()
        _page_index.build(all_tickets)
        logger.debug("PageIndex built with %d tickets", len(all_tickets))
    return _page_index


def fetch_tickets(
    days: int = 30,
    service: str | None = None,
    priority: str | None = None,
    category: str | None = None,
) -> list[Ticket]:
    """
    Fetch incident tickets from the corpus.

    Filters by time window (days back from today), optional service, priority,
    and category. Returns list of matching Ticket objects.
    In mock mode: loads from fixtures and applies filters in memory.
    In real mode: raises NotImplementedError with configuration guidance.
    """
    settings = get_settings()
    if not settings.use_mock_data:
        raise NotImplementedError(
            "Real ITSM integration is not yet implemented. "
            "Set USE_MOCK_DATA=true in your .env file to use fixture data, "
            "or implement a provider adapter in taproot/providers/."
        )
    return _loader.get_tickets(days=days, service=service, priority=priority, category=category)


def get_ticket_details(ticket_id: str) -> Ticket:
    """
    Return full details of a single ticket by ID.

    Raises ValueError if ticket_id not found.
    """
    settings = get_settings()
    if not settings.use_mock_data:
        raise NotImplementedError(
            "Real ITSM integration is not yet implemented. Set USE_MOCK_DATA=true."
        )
    return _loader.get_ticket_by_id(ticket_id)


def search_similar_tickets(ticket_id: str, top_k: int = 10) -> list[dict]:
    """
    Find tickets similar to the given ticket using PageIndex (BM25 + semantic + LLM rerank).

    Returns list of dicts with: ticket_id, title, similarity_score, match_reason.
    The match_reason is a short phrase explaining why it matched.
    """
    query_ticket = _loader.get_ticket_by_id(ticket_id)
    index = _get_page_index()
    results = index.search(query_ticket, top_k=top_k)

    # Normalise to the documented contract: similarity_score from rrf_score
    return [
        {
            "ticket_id": r["ticket_id"],
            "title": r["title"],
            "similarity_score": r.get("rrf_score", 0.0),
            "match_reason": r.get("match_reason", ""),
        }
        for r in results
    ]
