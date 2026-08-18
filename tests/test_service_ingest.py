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
