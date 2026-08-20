# TASK-026: Expose HNSW Tuning in Config

> **Status**: Complete ✅  
> **Phase**: Performance Phase 1  
> **Depends on**: TASK-024 (benchmark harness)  
> **Blocks**: TASK-029 (before/after benchmarks)  

## Objective

Expose HNSW index parameters (`m`, `ef_construction`, `ef_search`) in configuration with tuned defaults for better recall.

## Problem

pgvector's HNSW defaults to `ef_search = 40` at query time and `ef_construction = 64` at index build time. These are never tuned and there's no way to configure them. The pgvector docs recommend:
- `ef_construction = 200` for high-recall indexes
- `ef_search = 100+` for good recall at query time

**Location**: `store.py` lines 230–234 — HNSW index creation  
**Location**: No config model for HNSW parameters

## Solution

1. Add `HnswConfig` model in `config.py`.
2. Use it in `store.py` when creating HNSW index.
3. Set table-level `ef_search` default in `init_schema()`.
4. Tune defaults: `ef_construction = 200` (up from 64), `ef_search = 80` (up from 40).

## Files to Modify

### `src/semsearch/config.py`

Add new config model after `DiskANNConfig`:

```python
class HnswConfig(BaseModel):
    """HNSW index tuning parameters.

    Env var mapping:
        SEMSEARCH_HNSW__M=16
        SEMSEARCH_HNSW__EF_CONSTRUCTION=200
        SEMSEARCH_HNSW__EF_SEARCH=80
    """
    m: int = Field(default=16, ge=2, le=100)
    ef_construction: int = Field(default=200, ge=4, le=1000)
    ef_search: int = Field(default=80, ge=10, le=1000)
```

Add to `Settings` class:

```python
hnsw: HnswConfig = Field(default_factory=HnswConfig)
```

### `src/semsearch/store.py`

Update HNSW index creation in `init_schema()` to use config values:

```python
hnsw = settings.hnsw
cur.execute(
    f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
    f"WITH (m = {hnsw.m}, ef_construction = {hnsw.ef_construction})"
)
```

Add table-level `ef_search` default after index creation:

```python
# Set table-level ef_search default (idempotent, works for existing indexes too)
if not has_vectorscale:
    hnsw = settings.hnsw
    try:
        cur.execute(
            f"ALTER TABLE {table} SET (hnsw.ef_search = {hnsw.ef_search})"
        )
    except Exception as e:
        # ef_search not supported by this pgvector version — safe to ignore
        logger.debug("Could not set hnsw.ef_search on %s: %s", table, e)
```

## Design Decisions

### Table-level vs Session-level ef_search

Using `ALTER TABLE ... SET (hnsw.ef_search = N)` sets the default for all sessions querying the table. Users can still override per-session via `SET hnsw.ef_search = 200;` in psql.

Benefits:
- No code changes needed in the search path
- Works with `PGVectorStore.similarity_search_with_score()` without injection
- Easy to verify: `SHOW hnsw.ef_search;` after connecting

### Config Precedence

1. `SEMSEARCH_HNSW__EF_SEARCH=100` (env var)
2. `HnswConfig(ef_search=80)` (default)
3. pgvector default `40` (only if table created before this feature)

## Acceptance Criteria

- [x] `HnswConfig` model exists in `config.py`
- [x] `semsearch init` creates index with configured `ef_construction`
- [x] `semsearch init` sets table-level `ef_search` default
- [x] Configurable via env vars: `SEMSEARCH_HNSW__M`, `SEMSEARCH_HNSW__EF_CONSTRUCTION`, `SEMSEARCH_HNSW__EF_SEARCH`
- [x] All existing tests pass: `pytest tests/test_service_search.py -v`
- [x] New tests validate HNSW config is applied
- [x] Env var override test added
