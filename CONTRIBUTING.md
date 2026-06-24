# Contributing to taproot

Thank you for your interest in contributing. taproot is a focused tool — contributions that add scope creep will be declined, but improvements to correctness, performance, test coverage, and ITSM integrations are very welcome.

## Development setup

```bash
git clone https://github.com/aurora-creatives/taproot
cd taproot
pip install -e ".[dev]"
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env for integration testing
```

## Running tests

```bash
pytest                        # full test suite
pytest --cov=taproot          # with coverage report
pytest tests/test_pageindex.py  # single module
```

All tests must pass with zero failures before submitting a pull request. Tests must not make real API calls — mock `taproot.agent.LLMRouter` and any provider class in tests that exercise the agent or LLM reranking.

## Code standards

- All functions must have docstrings.
- All parameters and return types must have type hints.
- Use `rich.console.Console` for terminal output — no bare `print()`.
- No hardcoded strings for service names, priorities, or config values.
- No bare `except:` — catch specific exceptions.
- Imports grouped: stdlib → third-party → local, blank line between each group.
- `config.py` `Settings` is the single source of truth for all configuration.
- Run `ruff check src/ tests/` before submitting.

## What to contribute

**Welcome:**
- Bug fixes
- New ITSM provider adapters (ServiceNow, Jira, Freshservice, etc.)
- Improved test coverage
- Performance improvements to PageIndex
- Better fixture data (more realistic tickets, more patterns)
- Documentation improvements

**Not in scope:**
- Live incident response features
- Metrics / trace / log ingestion
- Automatic approval or publishing of problem records
- Vector database integration (by design — see architecture.md)

## Submitting changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Make your changes with tests.
4. Run `pytest` to verify everything passes.
5. Open a pull request with a clear description of what and why.

## Questions

Open an issue with the `question` label.
