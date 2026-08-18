# TASK-010: Core Service — Search Method

> **Phase**: 8.4 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-008
> **Blocks**: TASK-014, TASK-017

## Objective

Implement the `search()` method — the read path that delegates to `PGVectorStore.similarity_search_with_score()` and converts distance to similarity.

## File to Modify

### `src/semsearch/service.py` (add `search` method)

## Implementation

### Method Signature

```python
def search(
    self,
    query: str,
    k: int | None = None,
    filter: dict | None = None,
) -> list[SearchResult]:
```

### Step-by-Step Flow

```python
def search(
    self,
    query: str,
    k: int | None = None,
    filter: dict | None = None,
) -> list[SearchResult]:
    # Step 1: Validate k
    if k is None:
        k = self.settings.default_k
    if not (1 <= k <= 50):
        raise ValueError(f"k must be between 1 and 50, got {k}")

    # Step 2: Delegate to PGVectorStore
    try:
        results_with_scores = self.store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
        )
    except Exception as e:
        raise SearchError(f"Search failed: {e}") from e

    # Step 3: Convert distance → similarity and wrap in SearchResult
    results = []
    for doc, distance in results_with_scores:
        # LangChain returns cosine DISTANCE, not similarity
        # Higher distance = less similar
        # Convert: score = 1.0 - distance (so higher score = more similar)
        score = 1.0 - distance

        results.append(SearchResult(
            id=doc.metadata.get("langchain_id", ""),
            content=doc.page_content,
            score=score,
            source=doc.metadata.get("source"),
            chunk_index=doc.metadata.get("chunk_index"),
            page=doc.metadata.get("page"),
            row=doc.metadata.get("row"),
            doc_type=doc.metadata.get("doc_type"),
            metadata=doc.metadata,
        ))

    # Already sorted by distance ASC (best first), which = score DESC
    return results
```

## Critical Implementation Details

### 1. Score Conversion

**LangChain returns distance, NOT similarity.**

```python
# distance = cosine distance (0 = identical, 2 = opposite)
# score = 1.0 - distance (range: [-1, 1], higher = more similar)
score = 1.0 - distance
```

This must be done in the service. Tests must assert on the converted score, not the raw distance.

### 2. Filter Syntax

The `filter` parameter is passed straight through to `PGVectorStore.similarity_search_with_score(filter=...)`. Supported operators (from SPEC §8.4):

| Operator | Meaning | Example |
|----------|---------|---------|
| (direct value) | Equality | `{"source": "docs/x.pdf"}` |
| `$eq` | Equality | `{"source": {"$eq": "docs/x.pdf"}}` |
| `$ne` | Inequality | `{"doc_type": {"$ne": "pdf"}}` |
| `$lt`, `$lte` | Less than / <= | `{"year": {"$lt": 2024}}` |
| `$gt`, `$gte` | Greater than / >= | `{"year": {"$gte": 2024}}` |
| `$in` | In list | `{"source": {"$in": ["a.pdf", "b.pdf"]}}` |
| `$nin` | Not in list | `{"doc_type": {"$nin": ["csv", "json"]}}` |
| `$between` | Between | `{"page": {"$between": [1, 10]}}` |
| `$like` | SQL LIKE | `{"source": {"$like": "docs/%"}}` |
| `$ilike` | Case-insensitive LIKE | `{"source": {"$ilike": "DOCS/%"}}` |
| `$and` | Logical AND | `{"$and": [{...}, {...}]}` |
| `$or` | Logical OR | `{"$or": [{...}, {...}]}` |
| `$exists` | Field present | `{"source": {"$exists": true}}` |
| `$not` | Logical NOT | `{"$not": {"doc_type": "pdf"}}` |

**⚠️ Verify before implementing**: Check installed `langchain-postgres==0.0.17` filter translator supports these operators. `$exists` and `$not` may not be in PGVectorStore (v2).

### 3. Result Ordering

`similarity_search_with_score` returns results ordered by distance ASC (best matches first). After conversion to similarity (`1.0 - distance`), this becomes score DESC — which matches user intuition.

### 4. Metadata Extraction

`PGVectorStore` returns Documents where metadata includes both the JSONB blob AND the top-level columns. The service extracts:
- `langchain_id` → `id`
- `source` → `source` (top-level column)
- `chunk_index` → `chunk_index` (top-level column)
- `page`, `row`, `doc_type` → from langchain_metadata JSONB
- Full metadata blob → `metadata`

## Verification (Integration Tests)

- [ ] I-4: Search returns ≤ k results sorted by score DESC
- [ ] I-5: Search empty table returns `[]`
- [ ] I-15: `$ilike` prefix filter returns only matching sources
- [ ] I-16: Exact source filter returns only that file's chunks
- [ ] I-17: `doc_type` filter works
- [ ] I-18: Combined `$and` filter works
- [ ] I-19: Filter matching nothing returns `[]` (no error)
- [ ] I-20: Numeric comparison filter works
- [ ] Score is in range [-1, 1] and higher = more similar
- [ ] `k` out of range raises `ValueError`
