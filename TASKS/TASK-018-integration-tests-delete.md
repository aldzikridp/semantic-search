# TASK-018: Integration Tests — Delete

> **Phase**: 10.5 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-008, TASK-011
> **Blocks**: TASK-022

## Objective

Write integration tests for the delete method covering various filter types and the special empty filter case.

## File to Create

### `tests/test_service_delete.py`

```python
import pytest

class TestDeleteBySource:
    def test_delete_by_exact_source(self, service):  # I-6
        """Delete by exact source removes only that file's chunks"""
        # Ingest a.pdf and b.pdf
        # Delete {"source": "a.pdf"}
        # Assert deleted_count == chunks_of_a
        # Assert only a.pdf chunks gone, b.pdf intact
        pass

class TestDeleteByMetadata:
    def test_delete_by_metadata_filter(self, service):  # I-7
        """Delete by metadata filter removes matching chunks"""
        # Ingest 2 invoices (doc_type=invoice) + 3 contracts (doc_type=contract)
        # Delete {"doc_type": "invoice"}
        # Assert deleted_count == 2
        # Assert only invoices gone
        pass

class TestDeleteAll:
    def test_delete_all_with_empty_filter(self, service):  # I-8
        """Delete with empty filter {} removes everything"""
        # Ingest 10 chunks
        # Delete {}
        # Assert deleted_count == 10
        # Assert table empty
        stats = service.stats()
        assert stats["chunk_count"] == 0

class TestCLIConfirmation:
    def test_delete_all_without_yes_blocked(self):  # I-12
        """delete --all without --yes exits non-zero"""
        # Use CliRunner to test
        # Assert exit code != 0
        pass

    def test_delete_all_with_yes_works(self, service):  # I-13
        """delete --all --yes wipes table"""
        # Use CliRunner with --all --yes
        # Assert deleted_count > 0
        # Assert table empty
        pass

class TestDeleteEdgeCases:
    def test_delete_nonexistent_source(self, service):
        """Delete for source that doesn't exist returns deleted_count=0"""
        result = service.delete({"source": "nonexistent.txt"})
        assert result.deleted_count == 0

    def test_delete_returns_filter(self, service):
        """DeleteResult includes the filter that was applied"""
        filter_dict = {"source": "test.txt"}
        result = service.delete(filter_dict)
        assert result.filter == filter_dict
```

## Critical Notes

1. **Empty filter test (I-8)** — Verifies the bypass of `PGVectorStore.delete()`
2. **deleted_count computation** — Verify it's accurate for the before/after approach
3. **CLI tests** — Use `typer.testing.CliRunner`
4. **Test ordering** — Each test gets fresh table (fixture handles this)

## Verification

- [ ] All delete tests pass
- [ ] `deleted_count` is accurate
- [ ] Empty filter `{}` deletes everything
- [ ] Non-empty filter only deletes matching rows
- [ ] CLI `--all` requires `--yes`
