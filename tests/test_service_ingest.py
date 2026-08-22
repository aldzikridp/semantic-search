"""Integration tests for the ingest method (I-1, I-2, I-3, I-9, I-10, I-11)."""

import pytest
from unittest.mock import patch


class TestIngestText:
    def test_ingest_text_file(self, service, sample_txt):
        """I-1: Ingesting a text file produces chunks with correct counts."""
        result = service.ingest(sample_txt)
        assert result.chunks_added >= 1
        assert result.chunks_reused == 0
        assert result.source == str(sample_txt)

    def test_ingest_creates_rows_in_db(self, service, sample_txt):
        """Ingested chunks appear in the database."""
        result = service.ingest(sample_txt)
        stats = service.stats()
        assert stats["chunk_count"] == result.chunks_added


class TestIngestCSV:
    def test_ingest_csv(self, service, sample_csv):
        """I-3: Ingesting a CSV produces one chunk per row."""
        result = service.ingest(sample_csv)
        assert result.chunks_added == 10


class TestIdempotency:
    def test_reingest_unchanged_reuses(self, service, sample_txt):
        """I-9: Re-ingesting unchanged file reuses all chunks."""
        service.ingest(sample_txt)
        result = service.ingest(sample_txt)
        assert result.chunks_added == 0
        assert result.chunks_reused > 0
        assert result.chunks_updated == 0

    def test_reingest_zero_embed_calls(self, service, sample_txt):
        """I-9b: Re-ingesting unchanged file makes zero embedding API calls."""
        service.ingest(sample_txt)
        initial_count = service.embedder._call_count
        service.ingest(sample_txt)
        assert service.embedder._call_count == initial_count

    def test_reingest_partial_change(self, service, tmp_path):
        """I-10: Re-ingesting with 1 changed chunk re-embeds only that chunk."""
        file = tmp_path / "test.txt"
        file.write_text(("chunk content " * 100 + "\n") * 5)
        result1 = service.ingest(file)
        total_chunks = result1.chunks_added

        # Modify one chunk
        content = file.read_text()
        lines = content.split("\n")
        lines[2] = "MODIFIED CHUNK " * 100
        file.write_text("\n".join(lines))

        result = service.ingest(file)
        assert result.chunks_updated >= 1
        assert result.chunks_added == 0
        assert result.chunks_reused + result.chunks_updated == total_chunks

    def test_case_d_stale_tail_pruned(self, service, tmp_path):
        """I-10d: File shortening deletes stale tail chunks."""
        file = tmp_path / "test.txt"
        file.write_text(("chunk " * 100 + "\n") * 5)
        service.ingest(file)

        # Shorten to 3 chunks
        content = file.read_text()
        lines = content.split("\n")
        file.write_text("\n".join(lines[:3]))

        result = service.ingest(file)
        assert result.chunks_pruned == 2
        stats = service.stats()
        assert stats["chunk_count"] == 3

    def test_case_d_no_prune_when_growing(self, service, tmp_path):
        """I-10e: File growing does not prune anything."""
        file = tmp_path / "test.txt"
        file.write_text(("chunk " * 100 + "\n") * 3)
        service.ingest(file)

        file.write_text(("chunk " * 100 + "\n") * 5)
        result = service.ingest(file)
        assert result.chunks_pruned == 0


class TestForceReembed:
    def test_force_reembeds_everything(self, service, sample_txt):
        """I-11: --force re-embeds all chunks even if unchanged."""
        service.ingest(sample_txt)
        initial_count = service.embedder._call_count
        result = service.ingest(sample_txt, reembed_unchanged=True)
        assert result.chunks_reused == 0
        assert result.chunks_updated > 0
        assert service.embedder._call_count > initial_count


class TestBatchedIngest:
    """Phase A: batched multi-row INSERT for CASE B/C."""

    def _count_rows_for_source(self, service, source):
        conn = service._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {service.settings.collection_name} "
                    f"WHERE source = %s",
                    (source,),
                )
                return int(cur.fetchone()[0])
        finally:
            conn.close()

    def test_ingest_many_chunks_single_transaction(self, service, tmp_path):
        """Phase A: ingesting a 300+ chunk file stores every row with correct counts."""
        file = tmp_path / "big.txt"
        file.write_text(("chunk content " * 100 + "\n") * 400)
        result = service.ingest(file)
        assert result.chunks_added >= 300
        assert result.chunks_reused == 0
        assert result.chunks_updated == 0
        assert self._count_rows_for_source(service, str(file)) == result.chunks_added

    def test_ingest_case_bc_preserves_upsert_semantics(self, service, tmp_path):
        """Phase A: re-ingesting a modified file updates CASE B rows, adds CASE C
        rows, and leaves CASE A rows untouched (ON CONFLICT upsert semantics)."""
        file = tmp_path / "upsert.txt"
        file.write_text(("original content " * 100 + "\n") * 5)
        first = service.ingest(file)
        total = first.chunks_added

        # Modify lines 1-2 (CASE B) and append 2 new lines (CASE C);
        # remaining original chunks stay untouched (CASE A).
        lines = file.read_text().split("\n")
        for i in range(1, 3):
            lines[i] = f"MODIFIED {i} " * 100
        file.write_text("\n".join(lines) + ("brand new tail\n") * 2)

        second = service.ingest(file)
        assert second.chunks_updated >= 1  # CASE B
        assert second.chunks_reused >= 1  # CASE A untouched
        assert second.chunks_added >= 1  # CASE C new tail
        total_after = (
            second.chunks_reused + second.chunks_updated + second.chunks_added
        )
        # Overlap makes chunk boundaries shift when lines change, so exact
        # per-case counts aren't fixed — but every chunk must be accounted for.
        assert self._count_rows_for_source(service, str(file)) == total_after
        assert total_after >= total
