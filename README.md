# taproot

**Finds the root beneath the noise.**  
Surfaces ITSM problem records that should exist but don't.

[![CI](https://github.com/aurora-creatives/taproot/actions/workflows/ci.yml/badge.svg)](https://github.com/aurora-creatives/taproot/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Multi-provider](https://img.shields.io/badge/LLM-Anthropic%20%7C%20OpenAI%20%7C%20Azure%20%7C%20Bedrock-6B48FF)](https://github.com/aurora-creatives/taproot)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## The problem

Your support team is probably resolving the same five problems every week without realising it. A login issue gets fixed on Monday. Another one appears Thursday. A third one the following Tuesday. Each ticket is closed individually, each resolution is a workaround, and the root cause is never formally documented. The same engineer spends an hour diagnosing something they diagnosed three weeks ago.

Most ITSM tools capture incidents. Very few help you notice that incidents keep coming back. ITIL calls this **Problem Management** — the practice of finding and eliminating root causes, not just symptoms. In practice, it almost never happens because nobody has the time to read through hundreds of tickets looking for patterns.

**taproot does it for you.** It reads your ticket history, reasons across it using an AI agent, and drafts ITIL-compliant problem records with root cause analysis — ready for a human to review and approve. Your team resolves the root cause once instead of the symptom repeatedly.

---

## What taproot does

1. **Fetches** your incident ticket history (from mock fixtures by default, or a real ITSM via env vars).
2. **Searches** for tickets that are operationally similar using BM25 keyword matching with LLM reranking.
3. **Clusters** tickets that share symptoms, resolution patterns, or affected services.
4. **Analyses** each cluster to identify the probable root cause and contributing factors.
5. **Drafts** ITIL-compliant problem records with a suggested permanent fix and workaround.
6. **Presents** each draft for human review — approve, reject, or annotate before anything is written out.

---

## Demo

```
$ taproot run --days 90

  taproot — agent running
  ┌─────────────────────────────────────────┐
  │ Tool            search_similar_tickets  │
  │ Tickets seen    35                      │
  │ Records drafted 2                       │
  │ Iteration       8 / 20                  │
  │ Elapsed         14.3s                   │
  └─────────────────────────────────────────┘

  Analysis Complete
  ┌──────────────────────────────┬───────┐
  │ Tickets analysed             │ 35    │
  │ Clusters found               │ 3     │
  │ Problem records drafted      │ 3     │
  │ Duration                     │ 28.1s │
  └──────────────────────────────┴───────┘

  Drafts saved to: output/drafts_20260612_143201.json
  Run `taproot review` to approve or reject draft records.

$ taproot review

  Reviewing 3 draft problem records

  PRB1749732721 — Auth service token expiry misconfiguration
  ┌──────────────────────┬──────────────────────────────────────────────────┐
  │ Priority             │ HIGH                                             │
  │ Confidence           │ HIGH                                             │
  │ Affected Services    │ user-auth-service, portal-frontend               │
  │ Related Incidents    │ INC-2026-0001, INC-2026-0002, INC-2026-0003 ...  │
  │ Root Cause           │ JWT token TTL is misconfigured...                │
  │ Suggested Fix        │ Implement configurable token TTL with sliding... │
  └──────────────────────┴──────────────────────────────────────────────────┘
  [A]pprove / [R]eject / [E]dit notes / [S]kip: A
```

---

## How it works

**The agent loop.** taproot runs an AI agent that reasons over your ticket corpus autonomously using a configurable LLM. It is given six tools — fetch tickets, search for similar tickets, get ticket details, analyse a cluster, check for existing problems, draft a problem record — and decides how to use them. The loop continues until the agent has reviewed the full corpus or reached the configured maximum draft count.

**PageIndex.** taproot retrieves similar tickets using a hybrid of BM25 (Best Match 25) keyword ranking and local semantic embeddings (`sentence-transformers`), combined via Reciprocal Rank Fusion (RRF). This runs entirely on-device — no ticket data leaves the machine for retrieval. Optional LLM reranking is layered on top for the final ordering pass.

**Human-in-the-loop.** Nothing in taproot auto-approves or auto-publishes. Every draft problem record goes through `taproot review`, where a human reads the full analysis and decides to approve, reject, or annotate it. Confidence scores (`HIGH` / `MEDIUM` / `LOW`) are shown prominently so reviewers can prioritise.

See [docs/architecture.md](docs/architecture.md) for the full system design.

---

## Providers and Compliance

taproot supports four LLM providers. One env var switches between them — no code changes.

### Provider comparison

| Provider | Data location | Best for | Setup |
|---|---|---|---|
| `anthropic` | Anthropic cloud | Development, non-regulated | API key |
| `openai` | OpenAI cloud | Development, non-regulated | API key |
| `azure_openai` | Your Azure tenant | GDPR, SOC 2, HIPAA, enterprise | Azure subscription |
| `aws_bedrock` | Your AWS account | GDPR, SOC 2, HIPAA, enterprise | AWS account |

### Privacy architecture

**Retrieval is always local.** The hybrid PageIndex (BM25 + `sentence-transformers` embeddings) runs entirely on-device. No ticket data leaves the machine for indexing or search. Embedding models are downloaded once to `~/.cache/huggingface/`.

**LLM reasoning calls** go to the configured provider. For regulated environments, use `azure_openai` or `aws_bedrock` to keep data within your cloud perimeter. Both satisfy GDPR, SOC 2, and HIPAA BAA requirements.

**`ENABLE_SCRUBBING=true`** adds a local PII anonymisation layer on top of any provider. Emails, IPs, hostnames, and URLs are replaced with placeholders (`<<EMAIL_1>>`, `<<IP_2>>`) before any content leaves the machine, and restored in the response. Runs entirely on-device.

### Switching providers

One line in `.env`:

```bash
# Switch from Anthropic to Azure OpenAI:
LLM_PROVIDER=azure_openai
```

### Single vs multi mode

```bash
# Single mode (default): one model for everything
LLM_MODE=single
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514

# Multi mode: optimise cost and capability per task
LLM_MODE=multi
LLM_RERANK_PROVIDER=openai          # fast, cheap — high volume
LLM_RERANK_MODEL=gpt-4o-mini
LLM_ANALYSIS_PROVIDER=anthropic     # most capable — pattern reasoning
LLM_ANALYSIS_MODEL=claude-sonnet-4-20250514
LLM_DRAFT_PROVIDER=azure_openai     # structured output — stays in tenant
LLM_DRAFT_MODEL=gpt-4o
```

---

## Quick start

```bash
git clone https://github.com/aurora-creatives/taproot
cd taproot
pip install -e .
cp .env.example .env
# Add your ANTHROPIC_API_KEY (or OPENAI_API_KEY) to .env
taproot list-tickets      # verify 35 mock tickets load
taproot run               # run the agent
taproot review            # review draft problem records
```

> **No ITSM credentials needed to try it.** `USE_MOCK_DATA=true` (the default) runs entirely on the 35 built-in fixture tickets — 3 hidden recurring patterns + 8 noise incidents.

---

## Configuration

See `.env.example` for the full set of variables. Key settings:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `single` | `single` = one model for all tasks. `multi` = route each task independently. |
| `LLM_PROVIDER` | `anthropic` | Active provider in single mode: `anthropic`, `openai`, `azure_openai`, `aws_bedrock`. |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model name for single mode. |
| `ANTHROPIC_API_KEY` | — | Required when using `anthropic` provider. |
| `OPENAI_API_KEY` | — | Required when using `openai` provider. |
| `AZURE_OPENAI_ENDPOINT` | — | Required when using `azure_openai` provider. |
| `AZURE_OPENAI_API_KEY` | — | Required when using `azure_openai` provider. |
| `AZURE_OPENAI_DEPLOYMENT` | — | Required when using `azure_openai` provider. |
| `AWS_ACCESS_KEY_ID` | — | Required when using `aws_bedrock` provider. |
| `AWS_SECRET_ACCESS_KEY` | — | Required when using `aws_bedrock` provider. |
| `ENABLE_SCRUBBING` | `false` | Anonymise PII locally before any LLM call. |
| `PAGEINDEX_USE_SEMANTIC` | `true` | Enable local semantic embeddings for hybrid search. |
| `PAGEINDEX_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model (~130MB, cached locally). |
| `USE_MOCK_DATA` | `true` | `true` = built-in fixture data. `false` = real ITSM. |
| `OUTPUT_DIR` | `./output` | Where draft problem records are saved as JSON. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

---

## Connecting real ITSM

taproot ships with mock data enabled. To connect a real ITSM system, set `USE_MOCK_DATA=false` and provide credentials — then implement the provider adapter (see [docs/architecture.md](docs/architecture.md#extending-taproot)).

### ServiceNow

```bash
USE_MOCK_DATA=false
SERVICENOW_INSTANCE_URL=https://your-instance.service-now.com
SERVICENOW_USERNAME=your_username
SERVICENOW_PASSWORD=your_password
```

### Jira Service Management

```bash
USE_MOCK_DATA=false
JIRA_BASE_URL=https://your-org.atlassian.net
JIRA_EMAIL=your@email.com
JIRA_API_TOKEN=your_api_token
```

---

## Project structure

```
taproot/
├── src/taproot/
│   ├── agent.py          # Agentic loop — LLMRouter + Anthropic message format
│   ├── pageindex.py      # Hybrid BM25 + semantic search + RRF fusion
│   ├── config.py         # Settings (pydantic-settings, .env)
│   ├── cli.py            # CLI: run / review / list-tickets / list-problems
│   ├── providers/        # LLMProvider abstraction + Anthropic, OpenAI, Azure, Bedrock
│   ├── scrubbing/        # Local PII scrubber (DataScrubber)
│   ├── tools/            # 6 tool functions + TOOL_DEFINITIONS (Anthropic format)
│   ├── models/           # Ticket, ProblemRecord, AnalysisSummary (Pydantic v2)
│   └── mock/             # Fixture data loader
├── data/fixtures/
│   ├── tickets.json      # 35 mock tickets (3 hidden patterns + 8 noise)
│   └── problems.json     # Existing problem records (empty by default)
├── tests/                # Test suite — zero real API calls
└── docs/
    └── architecture.md   # Component design, PageIndex rationale
```

---

## Running tests

```bash
pytest                    # full suite (45 tests, ~0.5s)
pytest --cov=taproot      # with coverage report
pytest -v                 # verbose output
```

All tests run offline — no real LLM API calls anywhere in the test suite.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and ITSM provider adapters (ServiceNow, Jira, Freshservice) are especially welcome.

---

## License

MIT — see [LICENSE](LICENSE).
