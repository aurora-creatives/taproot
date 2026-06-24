# taproot — Architecture

## System overview

taproot is a batch analysis tool that reads historical incident ticket data, identifies recurring patterns using an AI reasoning agent, and produces ITIL-compliant problem records for human review. It is designed to accelerate ITSM Problem Management in organisations where problem records are rarely created — not because incidents aren't recurring, but because no one has the bandwidth to find the patterns.

taproot is not an incident response tool. It does not receive webhooks, react to live alerts, or integrate into an incident management workflow. It is run on-demand or on a schedule over a rolling window of historical tickets. It is not an AIOps or observability platform — it does not read metrics, traces, or logs. Its input is ticket text: titles, descriptions, and resolution notes. It is also not a replacement for your ITSM platform — it outputs to JSON files and is designed to feed back into your existing problem management workflow via a human review step.

---

## Component breakdown

### Agent (`agent.py`)

The agent is an asynchronous loop that drives analysis by calling Claude with a set of tools. It receives a system prompt that defines its role as an ITSM problem manager, a user message with the analysis directive and parameters, and a set of tool definitions. The loop runs until Claude signals `end_turn` or a maximum iteration count is reached.

The agent is stateless between runs. All state — which tickets have been analysed, which problem records have been drafted — is managed through tool calls and Claude's context window. The agent decides which tickets to investigate, when a cluster is worth pursuing, whether a draft would duplicate an existing record, and when analysis is complete.

### PageIndex (`pageindex.py`)

PageIndex is the knowledge retrieval engine. It provides ticket similarity search using a three-stage hybrid pipeline: BM25 keyword retrieval, sentence-transformer semantic retrieval, and Reciprocal Rank Fusion (RRF) to combine both signals. On startup it builds a BM25 index and, when `PAGEINDEX_USE_SEMANTIC=true`, loads a local sentence-transformers model (`BAAI/bge-small-en-v1.5` by default) to encode the ticket corpus. When the agent calls `search_similar_tickets`, PageIndex:

1. Runs a BM25 query using the full text of the query ticket.
2. Runs a semantic cosine-similarity search over the local sentence-transformer embeddings.
3. Combines both ranked lists using Reciprocal Rank Fusion (`score = 1 / (k + rank)`, k=60) to produce a unified `rrf_score`.
4. If a `rerank_provider` is configured, sends the top candidates to the LLM for a final reranking pass.
5. Returns the ranked list with `rrf_score`, `bm25_rank`, `semantic_rank`, and `match_reason` per result.

If `sentence-transformers` is not installed, PageIndex degrades gracefully to BM25-only. All embeddings are computed locally — no ticket data is sent to an external service during indexing or retrieval.

### Tools (`tools/`)

Tools are the agent's hands. There are six:

- `fetch_tickets` — loads tickets from the corpus with optional filters.
- `get_ticket_details` — returns full details of a single ticket by ID.
- `search_similar_tickets` — runs PageIndex search for a given ticket.
- `get_existing_problems` — returns all existing problem records (to prevent duplicates).
- `analyze_ticket_cluster` — performs structural analysis of a group of tickets.
- `draft_problem_record` — creates and persists a draft problem record.

Each tool is a plain Python function. Tool definitions in `TOOL_DEFINITIONS` use the Anthropic message format; each provider adapter translates them internally before calling the API. The descriptions are operationally precise — they guide the agent's decision-making, not just describe parameters.

### Mock layer (`mock/`)

The mock layer provides fixture-based data for development and testing. `MockDataLoader` reads from `data/fixtures/tickets.json` and `data/fixtures/problems.json`, applies in-memory filters, and returns typed Pydantic objects. When `USE_MOCK_DATA=false`, tool functions raise `NotImplementedError` with guidance on implementing a real provider.

The fixture data contains 35 tickets with three hidden recurring patterns (auth service token expiry, report generation slowness, email notification delays) plus 8 noise tickets representing genuine one-off incidents.

### CLI (`cli.py`)

Four commands:

- `taproot run` — runs the agent and saves draft problem records to JSON.
- `taproot review` — interactive human review of draft records.
- `taproot list-tickets` — shows a table of tickets matching filters.
- `taproot list-problems` — shows all existing problem records.

The CLI uses `typer` for argument parsing and `rich` for terminal output.

---

## Why local hybrid search (BM25 + sentence-transformers)

The decision to use local models instead of external embedding APIs was deliberate:

**No infrastructure dependency.** External vector search requires an embedding API and a vector database (Pinecone, Weaviate, pgvector, etc.). taproot's hybrid index lives entirely in memory and is rebuilt in seconds — no additional services to run or manage.

**No embedding costs.** Every ticket processed through an external embedding API incurs a cost. `sentence-transformers` runs locally and is free at inference time. For a tool designed to run over a rolling window of hundreds or thousands of tickets, this matters.

**GDPR and data residency.** Sending ticket text to an external embedding API means your incident data — which may contain sensitive operational information — leaves your environment. Both BM25 and the sentence-transformer model run fully locally. LLM reranking sends only brief candidate excerpts to the configured provider, and only when a `rerank_provider` is set.

**Operational text has strong keyword signals.** ITSM tickets are structured operational reports containing error codes, service names, and specific technical phrases — `401 Unauthorized`, `token_expiry`, `connection pool exhausted`, `reporting-service`. These are exact-match signals where BM25 excels. Semantic search fills the gap when the same underlying issue is described with different vocabulary across tickets.

**RRF over a learned fusion.** Reciprocal Rank Fusion (`1 / (k + rank)`, k=60) requires no training data and no hyperparameter tuning. It consistently outperforms either signal alone across varying corpus sizes and query styles, making it a robust default for a tool that operators deploy against their own ticket data.

---

## The agent loop in detail

Each iteration of the agent loop:

1. The current message history (user directive + all prior assistant/tool turns) is sent to the configured LLM with `tools=TOOL_DEFINITIONS` and `max_tokens=4096`.
2. The LLM responds with either:
   - `stop_reason="tool_use"` — one or more tool call blocks, each with `id`, `name`, and `input`.
   - `stop_reason="end_turn"` — a final text response (analysis summary).
3. For each `tool_use` block, the corresponding Python function is called with the provided `input` dict. Results are serialised to JSON.
4. A `tool_result` block is created for each call, associating the result with the tool call's `id`.
5. All tool results are appended as a user turn and the loop continues.
6. The loop terminates when the LLM returns `end_turn`, `max_iterations=20` is reached, or `max_problem_records` drafts have been created.

All tool calls in a single turn are executed before sending the next request. Results are never skipped — skipping a tool result would corrupt the model's context and produce incorrect reasoning.

**Stopping conditions:**
- Claude returns `end_turn` (normal completion).
- `max_iterations=20` is reached (safety guard against runaway loops).
- `max_problem_records` drafts created (user-configurable cap, default 10).

---

## Human-in-the-loop design

taproot never auto-approves or auto-publishes problem records. The `draft_problem_record` tool writes to an in-session store. The `taproot run` command saves drafts to a JSON file. The `taproot review` command presents each draft interactively.

This design is intentional. An AI agent identifying recurring patterns is useful; an AI agent unilaterally creating problem records in your ITSM system is not. Problem records have organisational weight — they trigger investigations, resource allocation, and change management processes. A human must own that decision.

The confidence field (`HIGH` / `MEDIUM` / `LOW`) is designed to guide review prioritisation. A `HIGH` confidence record backed by 8 tickets with consistent symptoms and resolution notes needs less scrutiny than a `MEDIUM` record from 3 loosely related tickets. Reviewers can use this to triage their review time.

---

## Extending taproot

### Adding a new ITSM provider

1. Create `src/taproot/providers/<name>.py` implementing a class with the same interface as `MockDataLoader`: `get_tickets()`, `get_ticket_by_id()`, `get_all_tickets()`, `get_problems()`.
2. Add configuration variables to `Settings` in `config.py`.
3. In `tools/tickets.py` and `tools/problems.py`, add a branch that instantiates your provider when `use_mock_data=False` and the relevant credentials are set.

### Adding a new tool

1. Write the function in the appropriate `tools/` module.
2. Add a tool definition dict to `TOOL_DEFINITIONS` in `tools/__init__.py`.
3. Add the function to `_TOOL_DISPATCH` in `agent.py`.
4. Write tests in `tests/test_tools.py`.

### Swapping the LLM

taproot supports four LLM providers out of the box: **Anthropic**, **OpenAI**, **Azure OpenAI**, and **AWS Bedrock**. Provider selection is controlled entirely through environment variables — no code changes required.

**Single-provider mode** (default, `LLM_MODE=single`): one provider handles all tasks.

```env
LLM_MODE=single
LLM_PROVIDER=anthropic        # anthropic | openai | azure | bedrock
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
```

**Multi-provider mode** (`LLM_MODE=multi`): route different tasks to different models — use a fast, cheap model for candidate reranking and a more capable model for analysis and drafting.

```env
LLM_MODE=multi
LLM_RERANK_PROVIDER=openai
LLM_RERANK_MODEL=gpt-4o-mini
LLM_ANALYSIS_PROVIDER=anthropic
LLM_ANALYSIS_MODEL=claude-sonnet-4-20250514
LLM_DRAFT_PROVIDER=anthropic
LLM_DRAFT_MODEL=claude-sonnet-4-20250514
```

The `LLMRouter` in `providers/router.py` handles provider selection and exposes a uniform `LLMProvider` interface. Adding a new provider requires only implementing two async methods (`complete()` and `complete_simple()`) and registering it in `_build_provider()`.
