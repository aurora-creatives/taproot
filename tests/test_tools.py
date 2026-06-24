from unittest.mock import patch

import pytest

from taproot.models.problem import ProblemRecord, ProblemStatus
from taproot.models.ticket import Ticket


class TestFetchTickets:
    def test_returns_tickets_with_default_filters(self):
        """fetch_tickets() with no filters should return a non-empty list of Ticket objects."""
        from taproot.tools.tickets import fetch_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            results = fetch_tickets(days=90)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(t, Ticket) for t in results)

    def test_service_filter_narrows_results(self):
        """Service filter should return only tickets for that service."""
        from taproot.tools.tickets import fetch_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            results = fetch_tickets(days=90, service="user-auth-service")
        assert all(t.service == "user-auth-service" for t in results)

    def test_priority_filter_works(self):
        """Priority filter should return only tickets with that priority."""
        from taproot.tools.tickets import fetch_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            results = fetch_tickets(days=90, priority="P2")
        assert all(t.priority.value == "P2" for t in results)

    def test_empty_results_when_no_match(self):
        """Filtering with a non-existent service should return an empty list."""
        from taproot.tools.tickets import fetch_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            results = fetch_tickets(days=90, service="nonexistent-service-xyz")
        assert results == []

    def test_raises_when_mock_disabled(self):
        """fetch_tickets() should raise NotImplementedError when use_mock_data=False."""
        from taproot.tools.tickets import fetch_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = False
            with pytest.raises(NotImplementedError):
                fetch_tickets()


class TestGetTicketDetails:
    def test_returns_ticket_by_id(self):
        """get_ticket_details() should return the correct Ticket for a known ID."""
        from taproot.tools.tickets import get_ticket_details
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            ticket = get_ticket_details("INC-2026-0001")
        assert isinstance(ticket, Ticket)
        assert ticket.ticket_id == "INC-2026-0001"

    def test_raises_for_unknown_id(self):
        """get_ticket_details() should raise ValueError for an unknown ticket ID."""
        from taproot.tools.tickets import get_ticket_details
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            with pytest.raises(ValueError, match="not found"):
                get_ticket_details("INC-9999-9999")


class TestSearchSimilarTickets:
    def setup_method(self):
        """Reset the page_index singleton and configure settings for BM25-only mode."""
        import taproot.tools.tickets as tickets_module
        tickets_module._page_index = None

    def test_returns_list_of_dicts(self):
        """search_similar_tickets() should return a list of dicts with required keys."""
        from taproot.tools.tickets import search_similar_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            mock_cfg.return_value.PAGEINDEX_USE_SEMANTIC = False
            mock_cfg.return_value.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
            results = search_similar_tickets("INC-2026-0001", top_k=5)
        assert isinstance(results, list)
        for item in results:
            assert "ticket_id" in item
            assert "title" in item
            assert "similarity_score" in item
            assert "match_reason" in item

    def test_does_not_include_query_ticket(self):
        """search_similar_tickets() should not return the query ticket itself."""
        from taproot.tools.tickets import search_similar_tickets
        with patch("taproot.tools.tickets.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            mock_cfg.return_value.PAGEINDEX_USE_SEMANTIC = False
            mock_cfg.return_value.PAGEINDEX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
            results = search_similar_tickets("INC-2026-0001", top_k=10)
        result_ids = [r["ticket_id"] for r in results]
        assert "INC-2026-0001" not in result_ids

    def test_raises_for_unknown_id(self):
        """search_similar_tickets() should raise ValueError for unknown ticket IDs."""
        from taproot.tools.tickets import search_similar_tickets
        with pytest.raises(ValueError):
            search_similar_tickets("INC-INVALID-9999")


class TestGetExistingProblems:
    def test_returns_list(self):
        """get_existing_problems() should return a list (empty or otherwise)."""
        from taproot.tools.problems import get_existing_problems
        with patch("taproot.tools.problems.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            results = get_existing_problems()
        assert isinstance(results, list)

    def test_empty_fixture_returns_empty_list(self):
        """With an empty problems fixture, get_existing_problems() returns []."""
        from taproot.tools.problems import get_existing_problems
        with patch("taproot.tools.problems.get_settings") as mock_cfg:
            mock_cfg.return_value.use_mock_data = True
            results = get_existing_problems()
        assert results == []


class TestDraftProblemRecord:
    def test_creates_problem_record(self):
        """draft_problem_record() should create and return a valid ProblemRecord."""
        from taproot.tools.problems import clear_draft_store, draft_problem_record
        clear_draft_store()
        record = draft_problem_record(
            title="Test Problem",
            description="A test problem",
            root_cause="Test root cause",
            contributing_factors=["factor1"],
            affected_services=["test-service"],
            related_incident_ids=["INC-2026-0001"],
            suggested_permanent_fix="Fix it",
            workaround="Work around it",
            priority="high",
            confidence="HIGH",
        )
        assert isinstance(record, ProblemRecord)
        assert record.status == ProblemStatus.DRAFT
        assert record.problem_id.startswith("PRB")
        assert record.confidence == "HIGH"

    def test_record_is_stored_in_draft_store(self):
        """Drafted records should be retrievable via get_draft_store()."""
        from taproot.tools.problems import (
            clear_draft_store,
            draft_problem_record,
            get_draft_store,
        )
        clear_draft_store()
        draft_problem_record(
            title="Store Test",
            description="desc",
            root_cause="cause",
            contributing_factors=[],
            affected_services=["svc"],
            related_incident_ids=[],
            suggested_permanent_fix="fix",
            workaround="",
            priority="medium",
            confidence="MEDIUM",
        )
        store = get_draft_store()
        assert len(store) == 1
        assert store[0].title == "Store Test"


class TestAnalyzeTicketCluster:
    def test_returns_dict_with_required_keys(self):
        """analyze_ticket_cluster() should return a dict with the documented keys."""
        from taproot.tools.analysis import analyze_ticket_cluster
        result = analyze_ticket_cluster(["INC-2026-0001", "INC-2026-0002", "INC-2026-0003"])
        assert isinstance(result, dict)
        required_keys = {
            "common_symptoms",
            "probable_root_cause",
            "confidence",
            "affected_services",
            "pattern_description",
            "suggested_fix",
            "workaround",
        }
        assert required_keys.issubset(result.keys())

    def test_empty_cluster_returns_low_confidence(self):
        """An empty ticket_ids list should return LOW confidence."""
        from taproot.tools.analysis import analyze_ticket_cluster
        result = analyze_ticket_cluster([])
        assert result["confidence"] == "LOW"

    def test_invalid_ids_handled_gracefully(self):
        """Unknown ticket IDs should be skipped without raising."""
        from taproot.tools.analysis import analyze_ticket_cluster
        result = analyze_ticket_cluster(["INC-INVALID-0001", "INC-INVALID-0002"])
        assert result["confidence"] == "LOW"
        assert isinstance(result["common_symptoms"], list)

    def test_large_cluster_gets_higher_confidence(self):
        """A cluster of 6+ auth tickets should get MEDIUM or HIGH confidence."""
        from taproot.tools.analysis import analyze_ticket_cluster
        auth_ids = [
            "INC-2026-0001", "INC-2026-0002", "INC-2026-0003",
            "INC-2026-0004", "INC-2026-0005", "INC-2026-0006",
        ]
        result = analyze_ticket_cluster(auth_ids)
        assert result["confidence"] in ("HIGH", "MEDIUM")
        assert len(result["affected_services"]) > 0
