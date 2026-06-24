import re
from unittest.mock import patch

from typer.testing import CliRunner

from taproot.cli import app

runner = CliRunner()


def _plain(text: str) -> str:
    """Strip ANSI escape codes so help-text assertions work in CI and locally."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def test_list_tickets_exits_zero():
    """taproot list-tickets should exit with code 0."""
    with patch("taproot.cli.get_settings") as mock_settings, \
         patch("taproot.tools.tickets.get_settings") as mock_tool_settings:
        mock_settings.return_value.configure_logging = lambda: None
        mock_tool_settings.return_value.use_mock_data = True
        result = runner.invoke(app, ["list-tickets", "--days", "90"])
    assert result.exit_code == 0, f"Unexpected output: {result.output}"


def test_list_problems_exits_zero():
    """taproot list-problems should exit with code 0."""
    with patch("taproot.cli.get_settings") as mock_settings, \
         patch("taproot.tools.problems.get_settings") as mock_tool_settings:
        mock_settings.return_value.configure_logging = lambda: None
        mock_tool_settings.return_value.use_mock_data = True
        result = runner.invoke(app, ["list-problems"])
    assert result.exit_code == 0, f"Unexpected output: {result.output}"


def test_run_help_shows_options():
    """taproot run --help should show the expected options."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--days" in out
    assert "--service" in out
    assert "--max-records" in out


def test_list_tickets_help():
    """taproot list-tickets --help should show expected options."""
    result = runner.invoke(app, ["list-tickets", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--days" in out
    assert "--service" in out


def test_list_tickets_service_filter():
    """taproot list-tickets --service should filter to that service."""
    with patch("taproot.cli.get_settings") as mock_settings, \
         patch("taproot.tools.tickets.get_settings") as mock_tool_settings:
        mock_settings.return_value.configure_logging = lambda: None
        mock_tool_settings.return_value.use_mock_data = True
        result = runner.invoke(app, ["list-tickets", "--days", "90", "--service", "reporting-service"])
    assert result.exit_code == 0
    # Should show reporting-service tickets, not auth tickets
    assert "reporting-service" in result.output
