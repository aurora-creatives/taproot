# Changelog

All notable changes to taproot are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.2.0] — 2026-06-24

### Added

- **Multi-provider LLM support** — Anthropic, OpenAI, Azure OpenAI, and AWS Bedrock via a uniform `LLMProvider` protocol; switch providers entirely through environment variables
- **LLMRouter** — `LLM_MODE=single` routes all tasks to one provider; `LLM_MODE=multi` routes `RERANK`, `ANALYSIS`, and `DRAFT` tasks to independently configured models
- **Hybrid PageIndex** — BM25 + `sentence-transformers` (`BAAI/bge-small-en-v1.5`) + Reciprocal Rank Fusion (k=60); degrades gracefully to BM25-only when `sentence-transformers` is not installed
- **Local PII scrubbing** — `ENABLE_SCRUBBING=true` anonymises emails, IPs, URLs, hostnames, and names before any LLM call; shared placeholder mapping prevents collision across multi-message contexts; runs entirely locally
- **87 tests** — suite expanded with `test_router.py`, `test_providers.py`, `test_scrubber.py`; zero real API calls

### Changed

- Agent loop migrated to Anthropic-native message format (`tool_use` / `tool_result` content blocks)
- `PageIndex` now accepts a `rerank_provider` argument (any `LLMProvider`) instead of a raw API client
- Tool definitions use Anthropic `input_schema` format internally; OpenAI/Azure providers translate to `parameters` on the fly

---

## [0.1.0] — 2026-06-12

### Added

- **Agent loop** — async OpenAI function-calling agent that reasons over incident ticket history and drafts ITIL problem records autonomously (`agent.py`)
- **PageIndex** — BM25 similarity search with optional LLM reranking; no vector database required (`pageindex.py`)
- **Six tools** — `fetch_tickets`, `get_ticket_details`, `search_similar_tickets`, `get_existing_problems`, `analyze_ticket_cluster`, `draft_problem_record`
- **CLI** — four commands: `taproot run`, `taproot review`, `taproot list-tickets`, `taproot list-problems`
- **Mock data** — 35 fixture tickets with 3 hidden recurring patterns (auth token expiry, report generation slowness, email notification delays) and 8 noise incidents
- **Human-in-the-loop review** — interactive approval/rejection flow; nothing auto-publishes
- **Pydantic v2 models** — `Ticket`, `ProblemRecord`, `AnalysisSummary` with full validation
- **45 tests** — full offline test suite, zero real API calls
- **GitHub Actions CI** — lint (ruff) + pytest matrix (Python 3.11, 3.12, 3.13)

[Unreleased]: https://github.com/aurora-creatives/taproot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/aurora-creatives/taproot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aurora-creatives/taproot/releases/tag/v0.1.0
