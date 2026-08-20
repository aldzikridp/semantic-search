# TASK-025: Cache Vector Size + DB Read in stats()

> **Status**: Complete ✅  
> **Phase**: Performance Phase 1  
> **Depends on**: TASK-024 (benchmark harness)  
> **Blocks**: TASK-029 (before/after benchmarks)  

## Objective

Eliminate the real embedding API call in `stats()` and `init_schema()` by caching the vector dimension and reading it from the database.

## Problem

`stats()` calls `_get_vector_size()` on every invocation, which embeds the literal string `"dimension probe"` via a real API call (~150–800ms + API cost). `init_schema()` also calls it. For a metadata command, this is wasteful.

**Location**: `service.py` line ~163 — `_get_vector_size()`  
**Location**: `service.py` line ~601 — `"embedding_dim": self._get_vector_size()`

## Solution

1. Cache the vector dimension in `self._cached_vector_size` after first probe.
2. Add `_get_vector_size_from_db()` that reads dimension from `pg_attribute`.
3. In `stats()`, prefer DB read; fall back to cached probe only when table doesn't exist.

## Files to Modify

### `src/semsearch/service.py`

**`__init__`** — add cache field:
```python
self._cached_vector_size: int | None = None
```

**`_get_vector_size()`** — add caching:
```python
def _get_vector_size(self, *, force_probe: bool = False) -> int:
    """Return the embedding dimension. Cached after first probe.

    Args:
        force_probe: If True, ignore the cache and re-embed the probe string.
    """
    if not force_probe and self._cached_vector_size is not None:
        return self._cached_vector_size
    size = len(self.embedder.embed_query("dimension probe"))
    self._cached_vector_size = size
    return size
```

**New method `_get_vector_size_from_db()`**:
```python
def _get_vector_size_from_db(
    self, conn: psycopg.Connection | None = None
) -> int | None:
    """Read the vector dimension straight from pg_attribute — no API call.

    Args:
        conn: Optional psycopg connection to reuse. If None, a new
            connection is created and closed internally.
    """
    table = self.settings.collection_name
    owns_conn = conn is None
    if owns_conn:
        conn = self._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (regexp_match(format_type(atttypid, atttypmod), "
                "'\\((\\d+)\\)'))[1]::int "
                "FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        # Table doesn't exist or other DB error — fall back to probe.
        logger.debug("_get_vector_size_from_db failed: %s", e)
        return None
    finally:
        if owns_conn:
            conn.close()
```

**`stats()`** — use DB read first, reusing existing connection:
```python
"embedding_dim": self._get_vector_size_from_db(conn) or self._get_vector_size(),
```

## Acceptance Criteria

- [x] `semsearch stats` returns instantly when table exists (no network call)
- [x] `semsearch init` only probes once (first time), then uses cache
- [x] `_get_vector_size_from_db()` returns correct dimension from DB
- [x] All existing tests pass: `pytest tests/test_service_search.py -v`
- [x] New tests in `tests/test_vector_size_cache.py` pass
