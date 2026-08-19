# Architecture

This document describes the code structure and design decisions behind semsearch.

## Project Structure

```
semantic-search/
├── src/
│   └── semsearch/
│       ├── __init__.py       # Package marker + version
│       ├── cli.py            # Typer CLI entry point
│       ├── config.py         # Pydantic Settings + EmbeddingProviderConfig
│       ├── embeddings.py     # Provider dispatch factory
│       ├── errors.py         # Exception hierarchy
│       ├── loaders.py        # File type dispatch (pick_loader)
│       ├── models.py         # Pydantic models (SearchResult, IngestResult, etc.)
│       ├── service.py        # Main SemanticSearchService class
│       ├── splitter.py       # Text chunking (RecursiveCharacterTextSplitter)
│       └── store.py          # PostgreSQL + pgvector (PGEngine, PGVectorStore)
├── tests/
│   ├── conftest.py           # Shared fixtures (MockEmbeddings, service)
│   ├── test_loaders.py       # Loader unit tests
│   ├── test_embeddings.py    # Embeddings factory tests
│   ├── test_service_ingest.py    # Ingest integration tests
│   ├── test_service_search.py    # Search integration tests
│   ├── test_service_delete.py    # Delete integration tests
│   ├── test_service_ingest_dir.py # Directory ingest tests
│   └── test_cli.py           # CLI tests
├── docs/                     # This documentation
├── pyproject.toml            # Project metadata + dependencies
├── requirements.txt          # Pinned dependencies
├── .env.example              # Environment template
└── .env                      # Local configuration (gitignored)
```

## Design Decisions

### 1. Service-Owned Write Path

**Problem:** `PGVectorStore.add_documents()` does row-by-row commits and re-embeds all documents.

**Solution:** The service owns the write path using raw `psycopg` connections with parameterized SQL:

```python
conn = psycopg.connect(db_url)
with conn.cursor() as cur:
    # CASE A: cheap UPDATE (no embedding)
    # CASE B + C: INSERT ... ON CONFLICT DO UPDATE
    # CASE D: DELETE stale tail chunks
    conn.commit()  # Atomic
```

**Benefits:**
- Atomic transactions (all-or-nothing)
- Pre-computed embeddings (no double-embedding)
- UPSERT via `ON CONFLICT`
- CASE D cleanup in same transaction

**Note:** All writes (INSERT, UPDATE, DELETE) go through raw `psycopg`. `PGVectorStore` is used read-only for `similarity_search_with_score()`.

### 2. Content-Hash Caching

**Problem:** Re-ingesting unchanged files wastes API calls.

**Solution:** SHA-256 hash of each chunk's content:

```python
hash = sha256(chunk.page_content.encode()).hexdigest()
```

**Cases:**
- **CASE A:** Hash matches → reuse embedding, bump `ingested_at`
- **CASE B:** Hash differs → re-embed, UPSERT
- **CASE C:** No existing row → embed, INSERT
- **CASE D:** Stale tail → DELETE

### 3. Deterministic IDs

**Problem:** UUIDs are opaque and don't match natural keys.

**Solution:** `langchain_id = "{source}::{chunk_index}"`

```python
chunk_id = f"{source}::{chunk_index}"
# e.g., "docs/readme.md::0"
```

**Benefits:**
- Human-readable in `psql`
- Matches natural key `(source, chunk_index)`
- Enables UPSERT by natural key

### 4. Double Underscore Env Vars

**Problem:** Nested Pydantic models can't be set from flat env vars.

**Solution:** `env_nested_delimiter="__"` in pydantic-settings:

```python
model_config = SettingsConfigDict(
    env_prefix="SEMSEARCH_",
    env_nested_delimiter="__",
)
```

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai  # → settings.embedding_provider.type
```

### 5. Lazy Store Initialization

**Problem:** `PGVectorStore.create_sync()` requires the table to exist.

**Solution:** Lazy `store` property:

```python
@property
def store(self) -> PGVectorStore:
    if self._store is None:
        self._store = build_store(self.settings, self.engine, self.embedder)
    return self._store
```

**Flow:**
1. `from_settings()` builds engine + embedder (no store)
2. User calls `init_schema()` (creates table)
3. User calls `search()` → `store` property initializes PGVectorStore

## Data Flow

### Ingest Flow

```
File → pick_loader → loader.load() → with_doc_type
     → split_documents → SHA-256 hashes
     → Fetch existing rows (SELECT)
     → Classify: CASE A/B/C
     → Embed CASE B + C (batch)
     → Execute in ONE transaction:
         - CASE A: UPDATE ingested_at
         - CASE B + C: INSERT ... ON CONFLICT
         - CASE D: DELETE stale tail
     → Return IngestResult
```

### Search Flow

```
Query → PGVectorStore.similarity_search_with_score()
      → Convert: score = 1.0 - distance
      → Wrap in SearchResult
      → Return sorted by score DESC
```

### Delete Flow

```
Filter → Build WHERE clause from filter dict
       → Count matching (SELECT COUNT ... WHERE ...)
       → Delete matching (DELETE ... WHERE ...)
       → Commit
       → Return DeleteResult
```

**Filter handling:**
- `source` → top-level column: `WHERE source = %s`
- Other keys → JSONB path: `WHERE langchain_metadata->>'key' = %s`

## Module Responsibilities

### `config.py`

- `EmbeddingProviderConfig`: Provider settings (type, model, api_key, routing)
- `Settings`: Top-level settings (database_url, collection_name, chunking)

### `errors.py`

- `SemSearchError`: Base exception
- `FileIngestError`: Loader/embedder/insert failure
- `SearchError`: Search query failure
- `DeleteError`: Delete operation failure
- `SchemaMismatchError`: Vector dimension mismatch
- `ProviderConfigError`: Missing credentials

### `loaders.py`

- `pick_loader(path)`: Returns lazy loader callable
- `with_doc_type(docs, path)`: Injects `source` + `doc_type` metadata

### `splitter.py`

- `build_splitter(chunk_size, chunk_overlap)`: Creates text splitter
- `split_documents(docs, chunk_size, chunk_overlap)`: Splits documents

### `embeddings.py`

- `build_embedder(settings)`: Dispatches to provider-specific class
- `_build_openrouter_routing(cfg)`: Builds OpenRouter routing dict

### `store.py`

- `build_engine(settings)`: Creates PGEngine
- `build_store(settings, engine, embedder)`: Creates PGVectorStore
- `init_schema(settings, engine, vector_size)`: Creates table + indexes

### `models.py`

- `SearchResult`: Search hit with score
- `IngestResult`: Per-file ingest stats
- `BatchIngestResult`: Directory ingest stats
- `DeleteResult`: Delete operation result

### `service.py`

- `SemanticSearchService`: Main orchestration class
  - `init_schema()`: Create table
  - `ingest(path, *, conn=None)`: Single file ingest
  - `ingest_dir(dir_path)`: Batch ingest (reuses one connection for all files)
  - `search(query)`: Similarity search
  - `delete(filter, conn=None)`: Delete by filter (raw SQL)
  - `stats(conn=None)`: Table statistics
  - `reingest(path)`: Delete + ingest (reuses one connection)

**Connection reuse:** Methods accept an optional `conn` parameter. When `None`, they create and close their own connection. When provided, the caller owns the lifecycle. `ingest_dir()` and `reingest()` open one connection and pass it through to sub-operations.

### `cli.py`

- Typer app with 8 commands
- JSON output to stdout
- Error handling with SemSearchError

## Dependencies

### Core

- `langchain`, `langchain-core`: Document abstractions
- `langchain-postgres`: PGVectorStore
- `langchain-community`: Document loaders
- `langchain-text-splitters`: Text chunking
- `psycopg`: PostgreSQL driver
- `pgvector`: Vector column type
- `sqlalchemy`: Database toolkit
- `pydantic`, `pydantic-settings`: Configuration
- `typer`: CLI framework
- `pymupdf`: PDF loader
- `jq`: JSON loader

### Providers (lazy-loaded)

- `langchain-openai`: OpenAI, OpenRouter, OpenAI-compatible
- `langchain-ollama`: Ollama

### Test

- `pytest`, `pytest-cov`: Test framework
- `testcontainers`: PostgreSQL container for tests

## Future Improvements

- **Async support**: Native async/await for all operations
- **Streaming ingest**: Process files as they're written
- **Vector quantization**: Reduce storage for large corpora
- **Hybrid search**: Combine vector + full-text search
- **Multi-tenancy**: Separate collections per user
- **Connection pooling**: Add `psycopg_pool` for concurrent access patterns
