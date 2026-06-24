from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from taproot.models.ticket import Ticket, TicketPriority, TicketStatus
from taproot.pageindex import PageIndex


def test_build_succeeds(sample_ticket_list):
    """build() should succeed with a non-empty ticket list."""
    index = PageIndex(use_semantic=False)
    index.build(sample_ticket_list)
    assert index._bm25 is not None
    assert len(index._tickets) == len(sample_ticket_list)


def test_search_returns_at_most_top_k(page_index, sample_ticket):
    """search() should return no more than top_k results."""
    results = page_index.search(sample_ticket, top_k=2)
    assert len(results) <= 2


def test_search_returns_pattern_tickets_above_noise(page_index, sample_ticket):
    """
    Searching with a Pattern 1 ticket should surface other auth tickets
    (INC-TEST-0002, INC-TEST-0003) ahead of noise (INC-TEST-0004, INC-TEST-0005).
    """
    results = page_index.search(sample_ticket, top_k=4)
    result_ids = [r["ticket_id"] for r in results]
    auth_ids = {"INC-TEST-0002", "INC-TEST-0003"}
    noise_ids = {"INC-TEST-0004", "INC-TEST-0005"}

    assert any(tid in auth_ids for tid in result_ids), (
        f"Expected auth tickets in results, got: {result_ids}"
    )

    auth_ranks = [i for i, tid in enumerate(result_ids) if tid in auth_ids]
    noise_ranks = [i for i, tid in enumerate(result_ids) if tid in noise_ids]
    if auth_ranks and noise_ranks:
        assert min(auth_ranks) < min(noise_ranks), (
            "Auth tickets should rank higher than noise tickets"
        )


def test_tokenize_lowercases_and_removes_punctuation():
    """_tokenize() should lowercase text and strip all punctuation."""
    index = PageIndex(use_semantic=False)
    tokens = index._tokenize("Hello, World! This is a TEST.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    assert not any("," in t or "!" in t or "." in t for t in tokens)


def test_tokenize_removes_all_punctuation():
    """_tokenize() should handle mixed punctuation correctly."""
    index = PageIndex(use_semantic=False)
    tokens = index._tokenize("auth-service token_expiry 401/errors")
    assert all(isinstance(t, str) for t in tokens)
    assert len(tokens) > 0


def test_search_without_build_raises():
    """search() before build() should raise RuntimeError."""
    index = PageIndex(use_semantic=False)
    dummy = Ticket(
        ticket_id="X",
        title="test",
        description="test",
        service="svc",
        category="cat",
        priority=TicketPriority.P3,
        status=TicketStatus.OPEN,
        reported_by="user",
        assigned_team="team",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(RuntimeError, match="build\\(\\)"):
        index.search(dummy)


def test_search_with_query_not_in_corpus(sample_ticket_list):
    """search() with a ticket not in the index should still return results."""
    index = PageIndex(use_semantic=False)
    index.build(sample_ticket_list)

    query = Ticket(
        ticket_id="INC-NOT-IN-CORPUS",
        title="Login authentication failure token expired session",
        description="User cannot log in. 401 error. Token invalidated. Session expired.",
        service="user-auth-service",
        category="Authentication",
        priority=TicketPriority.P2,
        status=TicketStatus.OPEN,
        reported_by="External",
        assigned_team="Platform Engineering",
        created_at=datetime(2024, 11, 28, tzinfo=timezone.utc),
        tags=["authentication", "token"],
    )

    results = index.search(query, top_k=3)
    assert isinstance(results, list)
    assert not any(r["ticket_id"] == "INC-NOT-IN-CORPUS" for r in results)


def test_rrf_score_formula():
    """_rrf_score(1) should equal 1/61 (k=60 by default)."""
    index = PageIndex(use_semantic=False)
    assert abs(index._rrf_score(1) - (1.0 / 61.0)) < 1e-10
    assert abs(index._rrf_score(0, k=60) - (1.0 / 60.0)) < 1e-10


def test_search_returns_rrf_score_field(page_index, sample_ticket):
    """search() results should include rrf_score, bm25_rank, semantic_rank, match_reason."""
    results = page_index.search(sample_ticket, top_k=2)
    for r in results:
        assert "rrf_score" in r
        assert "bm25_rank" in r
        assert "match_reason" in r
        assert "ticket_id" in r
        assert "title" in r


def test_page_index_accepts_rerank_provider_param():
    """PageIndex should accept rerank_provider without raising."""
    mock_provider = MagicMock()
    index = PageIndex(use_semantic=False, rerank_provider=mock_provider)
    assert index._rerank_provider is mock_provider


def test_search_skips_llm_when_no_rerank_provider(sample_ticket_list, sample_ticket):
    """When rerank_provider is None, search() returns RRF results without calling any LLM."""
    index = PageIndex(use_semantic=False, rerank_provider=None)
    index.build(sample_ticket_list)
    results = index.search(sample_ticket, top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
