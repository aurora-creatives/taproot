## What does this PR do?

<!-- One paragraph. Link the related issue if there is one: Closes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] ITSM provider adapter (ServiceNow, Jira, etc.)
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] CI / tooling

## Checklist

- [ ] `pytest` passes locally with zero failures
- [ ] `ruff check src/ tests/` passes with no errors
- [ ] New public functions have docstrings and type hints
- [ ] No bare `except:` — specific exceptions only
- [ ] No `print()` — used `rich.Console` or `logging`
- [ ] `.env` is not committed (no secrets anywhere)
- [ ] Tests added or updated for the change
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`

## Testing notes

<!-- How did you test this? What edge cases did you consider? -->
