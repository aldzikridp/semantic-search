# TASK-008: Core Service — Skeleton & Stats

> **Phase**: 8.1-8.2 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-002, TASK-003, TASK-004, TASK-005, TASK-006, TASK-007
> **Blocks**: TASK-009, TASK-010, TASK-011, TASK-012, TASK-013, TASK-014

## Objective

Implement the `SemanticSearchService` class skeleton with lifecycle management (`from_settings`, context manager, `close`) and the `stats()` method to validate the full pipeline works end-to-end.

## File to Create

### `src/semsearch/service.py`

## Implementation

### 1. Class Skeleton

```python
class SemanticSearchService:
    """
    High-level facade over loader + splitter + embedder + PGVectorStore.

    Lifecycle:
      with SemanticSearchService.from_settings(settings) as svc:
          svc.ingest(...)
          svc.search(...)
          svc.delete(filter={...})
    """

    def __init__(self, settings, engine, embedder, store):
        self.settings = settings
        self.engine = engine
        self.embedder = embedder
        self.store = store

    @classmethod
    def from_settings(cls, settings: Settings) -> "SemanticSearchService":
        """Build all internal components from settings.
        Does NOT call init_schema — call svc.init_schema() explicitly."""
        engine = build_engine(settings)
        embedder = build_embedder(settings)
        store = build_store(settings, engine, embedder)
        return cls(settings, engine, embedder, store)

    def __enter__(self) -> "SemanticSearchService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying PGEngine connection pool.
        PGEngine.close() is async — submit to engine's background loop."""
        # Use asyncio.run_coroutine_threadsafe or asyncio.run
        pass
```

### 2. `init_schema` method

```python
def init_schema(self, *, recreate: bool = False) -> None:
    """Idempotently create the chunks table with the active provider's vector dim."""
    # Get vector size from embedder
    # For OpenAI: known from model name
    # For Ollama: depends on model
    vector_size = self._get_vector_size()
    init_schema(self.settings, self.engine, vector_size, recreate=recreate)
```

Helper to determine vector size:
```python
def _get_vector_size(self) -> int:
    """Embed a dummy query to determine the dimension."""
    return len(self.embedder.embed_query("dimension probe"))
```

### 3. `stats()` method

Simplest method — validates the store connection works:

```python
def stats(self) -> dict:
    """
    Return:
      {
        "table": str,
        "embedding_provider": str,
        "embedding_dim": int,
        "chunk_count": int,
        "source_count": int,
        "sources_by_count": list[tuple[str, int]]
      }
    """
```

Implementation:
1. Query `SELECT COUNT(*) FROM {table}` for `chunk_count`
2. Query `SELECT COUNT(DISTINCT source) FROM {table}` for `source_count`
3. Query `SELECT source, COUNT(*) as cnt FROM {table} GROUP BY source ORDER BY cnt DESC LIMIT 20` for `sources_by_count`
4. Return dict with all fields

Use SQLAlchemy for parameterized queries:
```python
with self.engine.begin() as conn:
    result = conn.execute(text(f"SELECT COUNT(*) FROM {self.settings.collection_name}"))
    chunk_count = result.scalar()
    # ... etc
```

## Critical Notes

1. **`from_settings` does NOT call `init_schema`** — Caller must explicitly call `svc.init_schema()` if needed.

2. **`PGEngine.close()` is async** — For sync service, use:
   ```python
   import asyncio
   loop = self.engine._async_engine.sync_engine.pool._creator
   # OR: just call asyncio.run(self.engine._async_engine.dispose())
   ```

3. **`_get_vector_size()` embeds a dummy query** — This is a one-time cost on service initialization.

4. **Table name from settings** — Always use `self.settings.collection_name`, never hardcode.

## Verification

- [ ] `SemanticSearchService.from_settings(settings)` creates a valid service
- [ ] Context manager (`with ... as svc`) calls `close()` on exit
- [ ] `svc.stats()` returns correct dict structure
- [ ] `svc.stats()` on empty table returns `chunk_count=0`, `source_count=0`
- [ ] `svc.init_schema()` creates the table
- [ ] `svc.init_schema()` called twice doesn't error (idempotent)
