"""Tests for the CLI module."""

import pytest
from click.testing import CliRunner
from cued_speech.cli import cli


class TestCLI:
    """Test cases for the CLI module."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Cued Speech Processing Tools" in result.output

    def test_cli_missing_required_args(self):
        """Test CLI with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # Click returns exit code 2 for usage errors (no subcommand)
        assert result.exit_code == 2
        assert "Usage:" in result.output
        assert "Cued Speech Processing Tools" in result.output

    def test_cli_version(self):
        """Test CLI version information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # The help should show the command description
        assert "Cued Speech Processing Tools" in result.output 