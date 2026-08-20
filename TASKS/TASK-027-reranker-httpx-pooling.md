# TASK-027: Persistent httpx2.Client in Reranker with Retry

> **Status**: Complete ✅  
> **Phase**: Performance Phase 1  
> **Depends on**: TASK-024 (benchmark harness)  
> **Blocks**: TASK-029 (before/after benchmarks)  

## Objective

Replace per-call `httpx2.post()` with a persistent `httpx2.Client` in the Reranker, and add retry logic for 429 rate limits.

## Problem

`Reranker.rerank()` creates a new HTTP request each call — no connection reuse, no retry. Each call pays TCP+TLS handshake overhead (~150ms).

**Location**: `reranker.py` lines 67–82 — `httpx2.post()` call  
**Location**: No retry logic for transient failures

## Solution

1. Create a persistent `httpx2.Client` in `Reranker.__init__()` with connection pooling and timeouts.
2. Add exponential backoff retry (3 attempts) for 429 rate limits.
3. Use `self._client.post()` instead of `httpx2.post()`.

## Files to Modify

### `src/semsearch/reranker.py`

**`__init__`** — add persistent client:

```python
def __init__(
    self,
    config: RerankerProviderConfig,
    api_key: str,
) -> None:
    self.base_url = config.base_url
    self.model = config.model
    self.api_key = api_key
    self.default_top_n = config.top_n

    # Persistent client — connection pool survives across rerank() calls
    self._client = httpx2.Client(
        timeout=httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=2.0),
        limits=httpx2.Limits(max_connections=10, max_keepalive_connections=5),
        headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        },
    )
```

**`rerank()`** — use persistent client with retry:

```python
def rerank(
    self,
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    if not documents:
        return []

    n = top_n or self.default_top_n
    texts = [doc.page_content for doc in documents]
    payload = {
        "model": self.model,
        "query": query,
        "documents": texts,
        "top_n": n,
    }

    # Retry with exponential backoff for 429s
    for attempt in range(3):
        try:
            response = self._client.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            break
        except httpx2.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 2:
                import time
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise SearchError(f"Rerank failed: {e}") from e
        except httpx2.HTTPError as e:
            raise SearchError(f"Rerank failed: {e}") from e

    results = data.get("results", [])

    # Map indices back to original documents (preserving metadata)
    reranked: list[Document] = []
    for r in results:
        idx = r["index"]
        score = r["relevance_score"]
        doc = documents[idx]
        doc.metadata["rerank_score"] = score
        reranked.append(doc)

    return reranked
```

## Design Decisions

### Connection Pool Settings

- `max_connections=10`: Reasonable limit for a single-service process
- `max_keepalive_connections=5`: Keep 5 connections warm between calls
- `timeout=30.0s`: Total timeout per request (connect=5s, read=30s)

### Retry Strategy

- 3 attempts max
- Exponential backoff: 0.5s, 1s, 2s (only for 429s)
- Non-429 errors raise immediately
- Only retries on `httpx2.HTTPStatusError` (429), not connection errors

### Backward Compatibility

This is a drop-in replacement. The `rerank()` method signature and return type are unchanged. External callers see no difference.

## Acceptance Criteria

- [x] `Reranker` uses persistent `httpx2.Client` instead of per-call `httpx2.post()`
- [x] Connection pool settings: `max_connections=10`, `max_keepalive_connections=5`
- [x] Retry logic: 3 attempts with exponential backoff for 429s
- [x] Non-429 errors raise immediately
- [x] `close()` method releases HTTP client
- [x] `SemanticSearchService.close()` closes reranker
- [x] All existing tests pass: `pytest tests/test_service_search.py -v`
- [x] New tests validate retry behavior (mock 429 responses)
- [x] New tests validate `build_reranker()` factory
