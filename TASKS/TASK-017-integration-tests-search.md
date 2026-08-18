# TASK-017: Integration Tests — Search

> **Phase**: 10.4 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-008, TASK-010
> **Blocks**: TASK-022

## Objective

Write integration tests for the search method covering similarity search, filters, and edge cases.

## File to Create

### `tests/test_service_search.py`

```python
import pytest

class TestBasicSearch:
    def test_search_returns_ranked_results(self, service):  # I-4
        """Search returns ≤ k results sorted by score DESC"""
        # Ingest test docs first
        # ...
        results = service.search("password reset", k=3)
        assert len(results) <= 3
        # Verify sorted by score DESC
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # Verify score is similarity (not distance)
        for r in results:
            assert -1 <= r.score <= 1

    def test_search_empty_table(self, service):  # I-5
        """Search on empty table returns empty list"""
        results = service.search("anything", k=5)
        assert results == []

class TestFilterSearch:
    def test_ilike_prefix_filter(self, service):  # I-15
        """$ilike prefix filter returns only matching sources"""
        # Ingest docs/a.txt, docs/sub/b.txt, other/c.txt
        # Search with filter={"source": {"$ilike": "docs/%"}}
        # Assert only a.txt and b.txt chunks returned
        pass

    def test_exact_source_filter(self, service):  # I-16
        """Exact source filter returns only that file's chunks"""
        # Ingest 2 files
        # Search with filter={"source": "docs/a.txt"}
        # Assert only a.txt chunks returned
        pass

    def test_doc_type_filter(self, service):  # I-17
        """doc_type filter works"""
        # Ingest mix of pdf/txt/csv
        # Search with filter={"doc_type": "pdf"}
        # Assert only PDF chunks returned
        pass

    def test_combined_and_filter(self, service):  # I-18
        """Combined $and filter works"""
        # Ingest docs/a.pdf, docs/b.txt, other/c.pdf
        # Search with filter={"$and": [{"source": {"$ilike": "docs/%"}}, {"doc_type": "pdf"}]}
        # Assert only docs/a.pdf chunks returned
        pass

    def test_filter_matches_nothing(self, service):  # I-19
        """Filter matching nothing returns empty list, no error"""
        # Ingest docs/a.txt
        # Search with filter={"source": {"$ilike": "nonexistent/%"}}
        results = service.search("test", filter={"source": {"$ilike": "nonexistent/%"}})
        assert results == []

    def test_numeric_comparison_filter(self, service):  # I-20
        """Numeric comparison filter works"""
        # Ingest invoice chunks with year=2024 in langchain_metadata
        # Search with filter={"year": {"$gte": 2024}}
        # Assert only 2024 invoices returned
        pass

class TestScoreConversion:
    def test_score_is_similarity_not_distance(self, service):
        """Score = 1.0 - distance (higher = more similar)"""
        # Ingest known content
        # Search for exact match
        # Score should be close to 1.0 (not 0.0)
        results = service.search("exact content from ingested doc", k=1)
        assert results[0].score > 0.5  # Should be high similarity

    def test_k_validation(self, service):
        """k out of range raises ValueError"""
        with pytest.raises(ValueError):
            service.search("test", k=0)
        with pytest.raises(ValueError):
            service.search("test", k=51)
```

## Critical Notes

1. **Score conversion test is critical** — Must verify `score = 1.0 - distance`
2. **Filter tests require specific test data** — Set up docs with known metadata
3. **Empty table test** — Must run before any data is ingested
4. **$ilike prefix test** — Verify grep-style matching works

## Verification

- [ ] All search tests pass
- [ ] Scores are in range [-1, 1] and sorted DESC
- [ ] All filter types work (exact, $ilike, $and, numeric)
- [ ] Empty filter returns empty list
- [ ] k validation works
