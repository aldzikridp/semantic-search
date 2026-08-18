"""CLI integration tests (C-1 through C-6)."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch

from semsearch.cli import app
from semsearch.config import get_settings

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


class TestConfigOption:
    def test_config_flag(self, tmp_path):
        """--config flag works with custom config file."""
        config = tmp_path / "test.env"
        config.write_text("SEMSEARCH_COLLECTION_NAME=cli_test_table\n")
        result = runner.invoke(app, ["--config", str(config), "version"])
        assert result.exit_code == 0
        assert "semsearch" in result.output

    def test_config_short_flag(self, tmp_path):
        """-c short flag works with custom config file."""
        config = tmp_path / "test.env"
        config.write_text("SEMSEARCH_COLLECTION_NAME=cli_test_table\n")
        result = runner.invoke(app, ["-c", str(config), "version"])
        assert result.exit_code == 0
        assert "semsearch" in result.output

    def test_config_missing_file(self):
        """--config with missing file exits non-zero."""
        result = runner.invoke(app, ["--config", "missing.env", "version"])
        assert result.exit_code != 0

    def test_config_default_behavior(self):
        """Without --config, uses default .env (unchanged behavior)."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0


class TestGetSettings:
    def test_get_settings_custom_file(self, tmp_path):
        """get_settings loads from custom config file."""
        config = tmp_path / "custom.env"
        config.write_text("SEMSEARCH_COLLECTION_NAME=custom_table\n")
        settings = get_settings(str(config))
        assert settings.collection_name == "custom_table"

    def test_get_settings_default(self):
        """get_settings without path uses default .env."""
        settings = get_settings()
        assert settings.collection_name == "semsearch_chunks"

    def test_get_settings_none(self):
        """get_settings with None uses default .env."""
        settings = get_settings(None)
        assert settings.collection_name == "semsearch_chunks"
