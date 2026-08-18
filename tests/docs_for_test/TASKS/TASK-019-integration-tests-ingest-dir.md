# TASK-019: Integration Tests — Ingest Dir

> **Phase**: 10.6 | **Priority**: High | **Status**: ✅ Done
> **Depends on**: TASK-008, TASK-012
> **Blocks**: TASK-022

## Objective

Write integration tests for the `ingest_dir` method covering file discovery, pruning, and edge cases.

## File to Create

### `tests/test_service_ingest_dir.py`

```python
import pytest
from pathlib import Path
import fnmatch

class TestFileDiscovery:
    def test_mixed_files(self, service, tmp_path):  # I-29
        """ingest_dir with mixed file types counts correctly"""
        # Create: 2 txt + 1 pdf + 1 .docx (unsupported)
        # ...
        result = service.ingest_dir(tmp_path)
        assert result.files_discovered == 4
        assert result.files_skipped_unsupported == 1
        assert result.files_attempted == 3
        assert result.files_succeeded == 3

    def test_glob_filter(self, service, tmp_path):  # I-30
        """--glob filter limits to matching files"""
        # Create: 2 pdf + 3 txt
        # Ingest with --glob "**/*.pdf"
        result = service.ingest_dir(tmp_path, glob="**/*.pdf")
        assert result.files_attempted == 2

    def test_exclude_filter(self, service, tmp_path):  # I-31
        """--exclude skips matching files"""
        # Create: draft/ subdir with files + root files
        # Ingest with --exclude "*/draft/*"
        result = service.ingest_dir(tmp_path, exclude=["*/draft/*"])
        # Assert draft files not ingested
        pass

class TestErrorHandling:
    def test_continue_on_error(self, service, tmp_path):  # I-32
        """Failing file + continue_on_error=True continues"""
        # Create: 1 corrupt pdf + 2 good files
        result = service.ingest_dir(tmp_path, continue_on_error=True)
        assert result.files_failed == 1
        assert result.files_succeeded == 2
        assert len(result.failed_files) == 1

    def test_no_continue_on_error(self, service, tmp_path):  # I-33
        """Failing file + no_continue_on_error raises"""
        # Create: 1 corrupt pdf + 2 good files
        with pytest.raises(Exception):
            service.ingest_dir(tmp_path, continue_on_error=False)

class TestHiddenAndSymlinks:
    def test_skips_hidden_files(self, service, tmp_path):  # I-34
        """Hidden files (starting with .) are skipped"""
        # Create: .secret.txt + notes.md
        # Only notes.md should be ingested
        pass

    def test_skips_symlinks_by_default(self, service, tmp_path):  # I-35
        """Symlinks skipped by default"""
        # Create: real file + symlink to another file
        # Default: symlink skipped
        pass

    def test_follows_symlinks_with_flag(self, service, tmp_path):
        """--follow-symlinks follows symlinks"""
        # Same setup as above but with follow_symlinks=True
        pass

class TestIdempotency:
    def test_idempotent_rerun(self, service, tmp_path):  # I-36
        """Re-running on unchanged dir makes zero embed calls"""
        service.ingest_dir(tmp_path)
        result = service.ingest_dir(tmp_path)
        assert result.aggregate.chunks_reused > 0
        assert result.aggregate.chunks_added == 0

    def test_new_file_added(self, service, tmp_path):  # I-37
        """New file added between runs"""
        service.ingest_dir(tmp_path)
        # Add new file
        (tmp_path / "new.txt").write_text("new content " * 100)
        result = service.ingest_dir(tmp_path)
        assert result.aggregate.chunks_added > 0

class TestPrune:
    def test_prune_deletes_orphans(self, service, tmp_path):  # I-38
        """--prune deletes chunks for deleted files"""
        service.ingest_dir(tmp_path)
        # Delete a file from disk
        deleted_file = list(tmp_path.glob("*.txt"))[0]
        deleted_file.unlink()
        result = service.ingest_dir(tmp_path, prune=True)
        assert str(deleted_file) in result.pruned_sources
        assert result.pruned_chunks > 0

    def test_no_prune_leaves_orphans(self, service, tmp_path):  # I-39
        """Without --prune, orphans remain"""
        service.ingest_dir(tmp_path)
        deleted_file = list(tmp_path.glob("*.txt"))[0]
        deleted_file.unlink()
        result = service.ingest_dir(tmp_path, prune=False)
        assert result.pruned_sources == []
        assert result.pruned_chunks == 0

    def test_prune_dry_run(self, service, tmp_path):  # I-40
        """--prune --dry-run lists but doesn't delete"""
        service.ingest_dir(tmp_path)
        deleted_file = list(tmp_path.glob("*.txt"))[0]
        deleted_file.unlink()
        result = service.ingest_dir(tmp_path, prune=True, prune_dry_run=True)
        assert str(deleted_file) in result.pruned_sources
        assert result.pruned_chunks == 0  # Nothing actually deleted

    def test_prune_file_rename(self, service, tmp_path):  # I-41
        """Prune handles file rename correctly"""
        # Old file becomes orphan, new file is ingested
        pass

    def test_prune_file_move(self, service, tmp_path):  # I-42
        """Prune handles file move correctly"""
        # Same as rename
        pass

    def test_prune_only_affects_dir_prefix(self, service, tmp_path):  # I-43
        """Prune only deletes sources under dir_path/ prefix"""
        # Ingest from data/ and other/
        # Delete file from other/
        # Run ingest_dir on data/ with --prune
        # other/'s chunks should NOT be pruned
        pass
```

## Critical Notes

1. **Test data setup** — Each test needs specific directory structures
2. **Prune prefix matching** — Verify `str(dir_path) + "/"` prefix logic
3. **Symlink tests** — May need platform-specific handling
4. **Hidden file test** — Verify `.` prefix detection

## Verification

- [ ] All ingest_dir tests pass
- [ ] File discovery rules work (hidden, symlinks, extensions)
- [ ] Prune correctly identifies and deletes orphans
- [ ] Dry run lists without deleting
- [ ] Error handling works with continue_on_error
