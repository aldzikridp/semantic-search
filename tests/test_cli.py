"""CLI integration tests (C-1 through C-6)."""

import json
import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from semsearch.cli import app

runner = CliRunner()


class TestVersionCommand:
    def test_version(self):
        """semsearch version shows version string."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "semsearch" in result.output


class TestDeleteCommand:
    def test_delete_all_without_yes_blocked(self):
        """C-5: semsearch delete --all without --yes exits non-zero."""
        result = runner.invoke(app, ["delete", "--all"])
        assert result.exit_code != 0

    def test_delete_requires_filter_or_all(self):
        """semsearch delete without --filter or --all exits non-zero."""
        result = runner.invoke(app, ["delete"])
        assert result.exit_code != 0


class TestHelpCommand:
    def test_help_shows_commands(self):
        """semsearch --help shows all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "ingest" in result.output
        assert "search" in result.output
        assert "delete" in result.output
        assert "stats" in result.output
