"""Integration tests for ingest_dir (I-29 through I-43)."""

import pytest
from pathlib import Path


class TestFileDiscovery:
    def test_mixed_files(self, service, tmp_path):
        """I-29: Mixed file types counted correctly."""
        (tmp_path / "a.txt").write_text("hello " * 100)
        (tmp_path / "b.txt").write_text("world " * 100)
        (tmp_path / "c.csv").write_text("name,val\nfoo,1\nbar,2\n")
        (tmp_path / "d.docx").write_text("unsupported")

        result = service.ingest_dir(tmp_path)
        assert result.files_discovered == 4
        assert result.files_skipped_unsupported == 1
        assert result.files_attempted == 3
        assert result.files_succeeded == 3
        assert result.files_failed == 0

    def test_glob_filter(self, service, tmp_path):
        """I-30: Glob filter limits to matching files."""
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
        (tmp_path / "c.txt").write_text("hello " * 100)

        result = service.ingest_dir(tmp_path, glob="**/*.txt")
        assert result.files_attempted == 1
        assert result.files_succeeded == 1

    def test_exclude_filter(self, service, tmp_path):
        """I-31: Exclude skips matching files."""
        (tmp_path / "keep.txt").write_text("keep " * 100)
        draft = tmp_path / "draft"
        draft.mkdir()
        (draft / "skip.txt").write_text("skip " * 100)

        result = service.ingest_dir(tmp_path, exclude=["*/draft/*"])
        assert result.files_attempted == 1
        assert result.files_succeeded == 1

    def test_skips_hidden_files(self, service, tmp_path):
        """I-34: Hidden files (starting with .) are skipped."""
        (tmp_path / "visible.txt").write_text("visible " * 100)
        (tmp_path / ".hidden.txt").write_text("hidden " * 100)

        result = service.ingest_dir(tmp_path)
        assert result.files_attempted == 1
        assert result.files_succeeded == 1


class TestErrorHandling:
    def test_continue_on_error(self, service, tmp_path):
        """I-32: Failing file + continue_on_error=True continues."""
        (tmp_path / "good.txt").write_text("good " * 100)
        (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

        result = service.ingest_dir(tmp_path, continue_on_error=True)
        assert result.files_failed == 1
        assert result.files_succeeded == 1
        assert len(result.failed_files) == 1

    def test_no_continue_on_error(self, service, tmp_path):
        """I-33: Failing file + no_continue_on_error raises."""
        (tmp_path / "good.txt").write_text("good " * 100)
        (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

        with pytest.raises(Exception):
            service.ingest_dir(tmp_path, continue_on_error=False)


class TestIdempotency:
    def test_idempotent_rerun(self, service, tmp_path):
        """I-36: Re-running on unchanged dir reuses all chunks."""
        (tmp_path / "a.txt").write_text("content " * 100)
        (tmp_path / "b.txt").write_text("data " * 100)

        service.ingest_dir(tmp_path)
        result = service.ingest_dir(tmp_path)
        assert result.aggregate.chunks_reused > 0
        assert result.aggregate.chunks_added == 0

    def test_new_file_added(self, service, tmp_path):
        """I-37: New file added between runs is ingested."""
        (tmp_path / "a.txt").write_text("content " * 100)
        service.ingest_dir(tmp_path)

        (tmp_path / "b.txt").write_text("new content " * 100)
        result = service.ingest_dir(tmp_path)
        assert result.aggregate.chunks_added > 0


class TestPrune:
    def test_prune_deletes_orphans(self, service, tmp_path):
        """I-38: Prune deletes chunks for deleted files."""
        (tmp_path / "a.txt").write_text("aaa " * 100)
        (tmp_path / "b.txt").write_text("bbb " * 100)

        service.ingest_dir(tmp_path)
        (tmp_path / "b.txt").unlink()

        result = service.ingest_dir(tmp_path, prune=True)
        assert any("b.txt" in s for s in result.pruned_sources)
        assert result.pruned_chunks > 0

    def test_no_prune_leaves_orphans(self, service, tmp_path):
        """I-39: Without prune, orphans remain."""
        (tmp_path / "a.txt").write_text("aaa " * 100)
        (tmp_path / "b.txt").write_text("bbb " * 100)

        service.ingest_dir(tmp_path)
        (tmp_path / "b.txt").unlink()

        result = service.ingest_dir(tmp_path, prune=False)
        assert result.pruned_sources == []
        assert result.pruned_chunks == 0

    def test_prune_dry_run(self, service, tmp_path):
        """I-40: Prune dry-run lists but doesn't delete."""
        (tmp_path / "a.txt").write_text("aaa " * 100)
        (tmp_path / "b.txt").write_text("bbb " * 100)

        service.ingest_dir(tmp_path)
        (tmp_path / "b.txt").unlink()

        result = service.ingest_dir(tmp_path, prune=True, prune_dry_run=True)
        assert any("b.txt" in s for s in result.pruned_sources)
        assert result.pruned_chunks == 0


class TestConnectionReuse:
    def test_ingest_dir_reuses_connection(self, service, tmp_path):
        """ingest_dir opens one connection for all files in the batch."""
        (tmp_path / "a.txt").write_text("content " * 100)
        (tmp_path / "b.txt").write_text("data " * 100)

        call_count = 0
        original_get_conn = service._get_conn

        def counting_get_conn():
            nonlocal call_count
            call_count += 1
            return original_get_conn()

        service._get_conn = counting_get_conn
        service.ingest_dir(tmp_path)

        # 1 connection for the ingest loop (all files share it)
        assert call_count == 1

    def test_ingest_dir_reuses_connection_with_prune(self, service, tmp_path):
        """ingest_dir opens separate connection for prune SELECT."""
        (tmp_path / "a.txt").write_text("aaa " * 100)
        (tmp_path / "b.txt").write_text("bbb " * 100)

        service.ingest_dir(tmp_path)
        (tmp_path / "b.txt").unlink()

        call_count = 0
        original_get_conn = service._get_conn

        def counting_get_conn():
            nonlocal call_count
            call_count += 1
            return original_get_conn()

        service._get_conn = counting_get_conn
        service.ingest_dir(tmp_path, prune=True)

        # 1 for ingest loop + 1 for prune SELECT/delete
        assert call_count == 2

    def test_reingest_reuses_connection(self, service, sample_txt):
        """reingest opens one connection for delete + ingest."""
        service.ingest(sample_txt)

        call_count = 0
        original_get_conn = service._get_conn

        def counting_get_conn():
            nonlocal call_count
            call_count += 1
            return original_get_conn()

        service._get_conn = counting_get_conn
        service.reingest(sample_txt)

        # 1 connection shared by delete + ingest
        assert call_count == 1

    def test_standalone_ingest_creates_own_connection(self, service, sample_txt):
        """Standalone ingest() with conn=None creates its own connection."""
        call_count = 0
        original_get_conn = service._get_conn

        def counting_get_conn():
            nonlocal call_count
            call_count += 1
            return original_get_conn()

        service._get_conn = counting_get_conn
        service.ingest(sample_txt)

        # 1 connection (merged read+write, not 2 like before)
        assert call_count == 1
