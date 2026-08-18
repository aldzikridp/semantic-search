# TASK-007: Database Store

> **Phase**: 7 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-002 (config), TASK-001 (langchain-postgres)
> **Blocks**: TASK-008

## Objective

Implement PGEngine/PGVectorStore construction and idempotent schema initialization with proper column types, indexes, and constraints.

## File to Create

### `src/semsearch/store.py`

## Implementation

### 1. Column Constants

```python
from langchain_postgres import PGEngine, PGVectorStore, Column

# Custom top-level columns (REAL Postgres columns, not JSONB keys)
CUSTOM_COLUMNS = [
    Column("source", "TEXT", nullable=False),
    Column("chunk_index", "INTEGER", nullable=False),
    Column("document_hash", "CHAR(64)"),
]

# Column names as strings — PGVectorStore.create_sync expects names, not Column objects
CUSTOM_COLUMN_NAMES = [c.name for c in CUSTOM_COLUMNS]

# Explicit column name overrides
LC_CONTENT_COLUMN = "content"              # Override; PGVectorStore default is "page_content"
LC_METADATA_COLUMN = "langchain_metadata"  # This IS LangChain's default
LC_ID_COLUMN = "langchain_id"             # This IS the default name; we override TYPE to TEXT
```

### 2. `build_engine(settings: Settings) -> PGEngine`

```python
def build_engine(settings: Settings) -> PGEngine:
    return PGEngine.from_connection_string(url=settings.database_url)
```

### 3. `build_store(settings: Settings, engine: PGEngine, embedder: Embeddings) -> PGVectorStore`

```python
def build_store(settings: Settings, engine: PGEngine, embedder: Embeddings) -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.collection_name,
        embedding_service=embedder,
        id_column=LC_ID_COLUMN,
        content_column=LC_CONTENT_COLUMN,
        metadata_json_column=LC_METADATA_COLUMN,
        metadata_columns=CUSTOM_COLUMN_NAMES,  # list of strings, NOT Column objects
    )
```

### 4. `init_schema(settings: Settings, engine: PGEngine, vector_size: int, *, recreate: bool = False) -> None`

This is the most complex function in this module. Steps:

1. **Check if table exists** (idempotency):
   ```sql
   SELECT EXISTS (
       SELECT 1 FROM information_schema.tables
       WHERE table_name = :table_name
   )
   ```

2. **If table exists and `recreate=False`**: Check vector dimension matches. If mismatch, raise `SchemaMismatchError`.

3. **If `recreate=True`**: `DROP TABLE IF EXISTS {table_name} CASCADE`

4. **Create table** via `engine.init_vectorstore_table()`:
   ```python
   engine.init_vectorstore_table(
       table_name=settings.collection_name,
       vector_size=vector_size,
       id_column=Column("langchain_id", "TEXT"),  # TEXT, not UUID
       content_column=LC_CONTENT_COLUMN,
       metadata_json_column=LC_METADATA_COLUMN,
       metadata_columns=CUSTOM_COLUMNS,  # Column objects here, not strings
   )
   ```

5. **Create indexes** (separately — `init_vectorstore_table` does NOT create them):
   ```sql
   -- HNSW cosine similarity index
   CREATE INDEX IF NOT EXISTS {table}_hnsw_idx
       ON {table} USING hnsw (embedding vector_cosine_ops)
       WITH (m = 16, ef_construction = 64);

   -- JSONB GIN index for filter performance
   CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx
       ON {table} USING gin (langchain_metadata jsonb_path_ops);

   -- Composite index for re-ingest lookup
   CREATE INDEX IF NOT EXISTS {table}_source_chunk_idx
       ON {table} (source, chunk_index);

   -- UNIQUE constraint for chunk identity
   ALTER TABLE {table} ADD CONSTRAINT {table}_source_chunk_unique
       UNIQUE (source, chunk_index);
   ```

## Critical Verification Needed

### `Column("langchain_id", "TEXT", primary_key=True)`

The spec notes this is **unverified** against `langchain-postgres==0.0.17`. Before implementing:

1. Read the installed library's `Column` definition
2. If `primary_key` isn't a real parameter, drop it — the id column is implicitly primary key by role
3. The call should be: `Column("langchain_id", "TEXT")` without `primary_key=True`

### `init_vectorstore_table` is NOT idempotent

It uses `CREATE TABLE`, not `CREATE TABLE IF NOT EXISTS`. The idempotency check in step 1 prevents double-creation errors.

## Critical Notes

1. **`CUSTOM_COLUMNS` vs `CUSTOM_COLUMN_NAMES`**:
   - `CUSTOM_COLUMNS` (Column objects) → used by `init_vectorstore_table()`
   - `CUSTOM_COLUMN_NAMES` (strings) → used by `PGVectorStore.create_sync()`

2. **Indexes created separately** — `init_vectorstore_table` does NOT create HNSW, GIN, or composite indexes. Must use `CREATE INDEX IF NOT EXISTS`.

3. **`UNIQUE (source, chunk_index)`** — Enforced at DB level. Prevents duplicate chunks.

4. **Vector dimension must match provider** — OpenAI `text-embedding-3-small` = 1536, Ollama `nomic-embed-text` = 768, etc.

5. **`PGEngine.close()` is async** — For sync API, use `asyncio.run_coroutine_threadsafe(...).result()` or similar.

## Verification

- [ ] Table created with correct column types (TEXT id, vector(N) embedding, TEXT content, JSONB metadata)
- [ ] Idempotent: calling `init_schema` twice doesn't error
- [ ] `recreate=True` drops and recreates table
- [ ] All three indexes created (HNSW, GIN, composite)
- [ ] UNIQUE constraint on (source, chunk_index) enforced
- [ ] `SchemaMismatchError` raised when vector_size doesn't match existing table
