# TASK-021: CLI Tests

> **Phase**: 10.8 | **Priority**: Medium | **Status**: Not Started
> **Depends on**: TASK-014
> **Blocks**: TASK-022

## Objective

Write CLI integration tests using `typer.testing.CliRunner`.

## File to Create

### `tests/test_cli.py`

```python
import pytest
from typer.testing import CliRunner
from semsearch.cli import app

runner = CliRunner()

class TestInitCommand:
    def test_init_idempotent(self, settings):  # C-1
        """semsearch init x2 — both succeed"""
        # Run init twice
        # Both should succeed
        pass

class TestIngestCommand:
    def test_ingest_prints_json(self, settings, sample_txt):  # C-2
        """semsearch ingest prints valid JSON with chunks_added"""
        result = runner.invoke(app, ["ingest", str(sample_txt)])
        assert result.exit_code == 0
        import json
        output = json.loads(result.output)
        assert "chunks_added" in output

class TestSearchCommand:
    def test_search_returns_results(self, settings):  # C-3
        """semsearch search returns results array"""
        # Ingest first, then search
        result = runner.invoke(app, ["search", "test query", "--k", "3"])
        assert result.exit_code == 0
        import json
        output = json.loads(result.output)
        assert "results" in output
        assert isinstance(output["results"], list)

class TestDeleteCommand:
    def test_delete_filter_works(self, settings):  # C-4
        """semsearch delete --filter works"""
        # Ingest first
        result = runner.invoke(app, [
            "delete",
            "--filter", '{"source": "./data/sample.txt"}'
        ])
        assert result.exit_code == 0
        import json
        output = json.loads(result.output)
        assert "deleted_count" in output

    def test_delete_all_without_yes_blocked(self):  # C-5
        """semsearch delete --all without --yes exits non-zero"""
        result = runner.invoke(app, ["delete", "--all"])
        assert result.exit_code != 0

class TestStatsCommand:
    def test_stats_shows_counts(self, settings):  # C-6
        """semsearch stats shows expected structure"""
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        import json
        output = json.loads(result.output)
        assert "table" in output
        assert "chunk_count" in output
        assert "source_count" in output
```

## Critical Notes

1. **Each CLI test needs fresh DB state** — May need to run `init --recreate --yes` before tests
2. **Environment setup** — Tests need correct `.env` or env vars for Settings
3. **JSON output validation** — Parse and validate structure
4. **Exit code validation** — 0 for success, non-zero for errors

## Verification

- [ ] All CLI tests pass
- [ ] JSON output matches expected schemas
- [ ] Error cases exit with non-zero code
- [ ] --all requires --yes confirmation
