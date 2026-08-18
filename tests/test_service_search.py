"""Integration tests for the search method."""

import pytest


class TestSearch:
    def test_search_returns_results(self, service, sample_txt):
        """I-4: Search returns results sorted by score DESC."""
        service.ingest(sample_txt)
        results = service.search("test paragraph", k=3)
        assert len(results) > 0
        assert len(results) <= 3
        # Verify sorted by score DESC
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_empty_table(self, service):
        """I-5: Search empty table returns empty list."""
        results = service.search("anything", k=5)
        assert results == []

    def test_search_score_range(self, service, sample_txt):
        """Scores are in range [-1, 1] and higher = more similar."""
        service.ingest(sample_txt)
        results = service.search("test", k=5)
        for r in results:
            assert -1 <= r.score <= 1

    def test_search_k_validation(self, service):
        """k out of range raises ValueError."""
        with pytest.raises(ValueError, match="k must be"):
            service.search("test", k=0)
        with pytest.raises(ValueError, match="k must be"):
            service.search("test", k=51)

    def test_search_by_doc_type(self, service, sample_txt, sample_csv):
        """I-17: doc_type filter works."""
        service.ingest(sample_txt)
        service.ingest(sample_csv)
        results = service.search("test", k=20, filter={"doc_type": "csv"})
        for r in results:
            assert r.doc_type == "csv"
