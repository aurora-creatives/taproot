from __future__ import annotations

import json
import logging
import re
import string
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.status import Status

from taproot.models.ticket import Ticket

if TYPE_CHECKING:
    from taproot.providers.base import LLMProvider

logger = logging.getLogger(__name__)
_console = Console(stderr=True)


class PageIndex:
    """
    Hybrid retrieval engine: BM25 + local sentence-transformer embeddings.
    Scores combined via Reciprocal Rank Fusion (RRF, k=60).

    PRIVACY: Runs entirely locally. No ticket data leaves the machine for
    retrieval. sentence-transformers models are downloaded once to ~/.cache.

    BM25 handles: exact keyword matches, error codes, service names.
    Semantic embeddings handle: equivalent symptoms with different vocabulary.
    RRF combines both without requiring score normalisation.
    """

    def __init__(
        self,
        use_semantic: bool = True,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        rerank_provider: LLMProvider | None = None,
    ) -> None:
        self._use_semantic = use_semantic
        self._embedding_model_name = embedding_model
        self._rerank_provider = rerank_provider
        self._bm25: BM25Okapi | None = None
        self._embedder = None  # SentenceTransformer, loaded lazily
        self._vectors: np.ndarray | None = None  # shape: (n_tickets, dim)
        self._ticket_ids: list[str] = []
        self._tickets: dict[str, Ticket] = {}

    def build(self, tickets: list[Ticket]) -> None:
        """
        Build BM25 index and (optionally) semantic vector index from ticket corpus.
        Must be called before search(). Downloads embedding model on first run (~130MB).
        """
        self._ticket_ids = [t.ticket_id for t in tickets]
        self._tickets = {t.ticket_id: t for t in tickets}

        texts = [self._build_ticket_text(t) for t in tickets]
        tokenized = [self._tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized)

        if self._use_semantic:
            self._load_embedder()
            if self._embedder is not None:
                self._vectors = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                logger.debug("Semantic vectors built: shape=%s", self._vectors.shape)

        n = len(tickets)
        mode = "BM25 + semantic" if (self._use_semantic and self._vectors is not None) else "BM25 only"
        logger.debug("PageIndex ready — %d tickets indexed (%s)", n, mode)

    def search(self, query_ticket: Ticket, top_k: int = 10) -> list[dict]:
        """
        Return top_k most similar tickets using hybrid retrieval.

        Steps:
        1. BM25 search → ranked list of (ticket_id, bm25_rank)
        2. Semantic search → ranked list of (ticket_id, semantic_rank) [if enabled]
        3. RRF fusion → combined score per ticket
        4. Optional LLM reranking via rerank_provider.complete_simple()
        5. Return top_k by score, excluding query ticket itself

        Returns list of dicts: ticket_id, title, rrf_score, bm25_rank, semantic_rank, match_reason
        """
        if self._bm25 is None:
            raise RuntimeError("PageIndex has not been built. Call build() first.")

        query_text = self._build_ticket_text(query_ticket)
        query_tokens = self._tokenize(query_text)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # BM25 ranking (ascending index = better rank)
        bm25_order = sorted(
            range(len(self._ticket_ids)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        bm25_rank: dict[str, int] = {}
        for rank, idx in enumerate(bm25_order):
            tid = self._ticket_ids[idx]
            if tid != query_ticket.ticket_id:
                bm25_rank[tid] = rank + 1

        # Semantic ranking (if enabled and vectors available)
        semantic_rank: dict[str, int] = {}
        if self._use_semantic and self._vectors is not None and self._embedder is not None:
            query_vec = self._embedder.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
            # Cosine similarity: normalise then dot product
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            query_norm = np.linalg.norm(query_vec)
            safe_norms = np.where(norms == 0, 1, norms)
            safe_query_norm = query_norm if query_norm != 0 else 1
            similarities = (self._vectors / safe_norms) @ (query_vec / safe_query_norm).T
            sim_order = sorted(
                range(len(self._ticket_ids)),
                key=lambda i: float(similarities[i]),
                reverse=True,
            )
            for rank, idx in enumerate(sim_order):
                tid = self._ticket_ids[idx]
                if tid != query_ticket.ticket_id:
                    semantic_rank[tid] = rank + 1

        # RRF fusion
        all_tids = set(bm25_rank.keys()) | set(semantic_rank.keys())
        rrf_scores: dict[str, float] = {}
        for tid in all_tids:
            score = 0.0
            if tid in bm25_rank:
                score += self._rrf_score(bm25_rank[tid])
            if tid in semantic_rank:
                score += self._rrf_score(semantic_rank[tid])
            rrf_scores[tid] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]

        # Build result list
        results = [
            {
                "ticket_id": tid,
                "title": self._tickets[tid].title if tid in self._tickets else "",
                "rrf_score": score,
                "bm25_rank": bm25_rank.get(tid),
                "semantic_rank": semantic_rank.get(tid),
                "match_reason": self._match_reason(tid, query_text),
            }
            for tid, score in ranked
            if tid in self._tickets
        ][:top_k]

        # Optional LLM reranking pass
        if self._rerank_provider is not None and results:
            results = self._llm_rerank(query_ticket, results, top_k)

        return results

    def _rrf_score(self, rank: int, k: int = 60) -> float:
        """Reciprocal Rank Fusion score: 1 / (k + rank). k=60 is standard."""
        return 1.0 / (k + rank)

    def _llm_rerank(
        self,
        query_ticket: Ticket,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Use rerank_provider.complete_simple() to reorder candidates by operational similarity."""
        import asyncio

        candidate_items = [
            {
                "ticket_id": r["ticket_id"],
                "title": r["title"],
                "description": self._tickets[r["ticket_id"]].description[:200]
                if r["ticket_id"] in self._tickets
                else "",
            }
            for r in candidates
        ]

        prompt = (
            "You are an ITSM analyst. Rerank the following incident tickets by operational "
            "similarity to the query ticket — meaning they likely share the same root cause.\n\n"
            f"QUERY: {query_ticket.ticket_id} — {query_ticket.title}\n"
            f"Description: {query_ticket.description[:300]}\n\n"
            "CANDIDATES (JSON):\n"
            f"{json.dumps(candidate_items, indent=2)}\n\n"
            "Return ONLY a JSON array ordered from most to least similar. "
            'Format: [{"ticket_id": "INC-...", "match_reason": "..."}, ...] '
            f"Maximum {top_k} results."
        )

        try:
            loop = asyncio.get_event_loop()
            raw = loop.run_until_complete(
                self._rerank_provider.complete_simple(prompt, max_tokens=500)
            )
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON array in reranking response")
            reranked = json.loads(match.group())

            id_to_candidate = {r["ticket_id"]: r for r in candidates}
            results = []
            for rank, item in enumerate(reranked[:top_k]):
                tid = item.get("ticket_id", "")
                if tid in id_to_candidate:
                    c = id_to_candidate[tid]
                    results.append({**c, "match_reason": item.get("match_reason", c["match_reason"])})
            return results
        except Exception as exc:
            logger.warning("LLM reranking failed (%s); using RRF results", exc)
            return candidates[:top_k]

    def _match_reason(self, ticket_id: str, query_text: str) -> str:
        """Generate a short match reason based on shared terms."""
        if ticket_id not in self._tickets:
            return "keyword similarity"
        ticket_text = self._build_ticket_text(self._tickets[ticket_id])
        query_tokens = set(self._tokenize(query_text))
        ticket_tokens = set(self._tokenize(ticket_text))
        shared = query_tokens & ticket_tokens
        # Filter out very common short words
        meaningful = [t for t in shared if len(t) > 3][:5]
        if meaningful:
            return f"Shared terms: {', '.join(meaningful)}"
        return "BM25 + semantic similarity"

    def _build_ticket_text(self, ticket: Ticket) -> str:
        """Combine ticket fields into a single string for indexing."""
        parts = [
            ticket.title,
            ticket.description,
            ticket.resolution_notes,
            ticket.service,
            ticket.category,
            " ".join(ticket.tags),
        ]
        return " ".join(p for p in parts if p)

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _load_embedder(self) -> None:
        """Lazily load SentenceTransformer. Shows status message on first load."""
        try:
            from sentence_transformers import SentenceTransformer

            with Status(
                "Loading embedding model (first run only)...",
                console=_console,
                spinner="dots",
            ):
                self._embedder = SentenceTransformer(self._embedding_model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to BM25 only. "
                "Run: pip install sentence-transformers"
            )
            self._use_semantic = False
        except Exception as exc:
            logger.warning("Failed to load embedding model (%s); falling back to BM25 only", exc)
            self._use_semantic = False
