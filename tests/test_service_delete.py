"""Integration tests for the delete method."""

import pytest


class TestDelete:
    def test_delete_by_source(self, service, sample_txt):
        """I-6: Delete by exact source removes only that file's chunks."""
        service.ingest(sample_txt)
        stats_before = service.stats()
        assert stats_before["chunk_count"] > 0

        result = service.delete({"source": str(sample_txt)})
        assert result.deleted_count > 0

        stats_after = service.stats()
        assert stats_after["chunk_count"] == 0

    def test_delete_empty_filter(self, service, sample_txt):
        """I-8: Delete with empty filter {} deletes all rows."""
        service.ingest(sample_txt)
        stats_before = service.stats()
        assert stats_before["chunk_count"] > 0

        result = service.delete({})
        assert result.deleted_count == stats_before["chunk_count"]

        stats_after = service.stats()
        assert stats_after["chunk_count"] == 0

    def test_delete_nonexistent_source(self, service):
        """Delete with non-matching filter returns deleted_count=0."""
        result = service.delete({"source": "nonexistent.txt"})
        assert result.deleted_count == 0

    def test_delete_with_filter_raw_sql(self, service, sample_txt):
        """Filtered delete uses raw SQL (no PGVectorStore dependency)."""
        service.ingest(sample_txt)
        stats_before = service.stats()
        assert stats_before["chunk_count"] > 0

        result = service.delete({"source": str(sample_txt)})
        assert result.deleted_count > 0
        assert result.deleted_count == stats_before["chunk_count"]

        stats_after = service.stats()
        assert stats_after["chunk_count"] == 0

    def test_delete_returns_correct_filter(self, service, sample_txt):
        """DeleteResult preserves the filter dict."""
        service.ingest(sample_txt)
        filter_dict = {"source": str(sample_txt)}
        result = service.delete(filter_dict)
        assert result.filter == filter_dict
