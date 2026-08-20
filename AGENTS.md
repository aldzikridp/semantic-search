# AGENTS.md — Semantic Search Project

> This file provides instructions for AI coding agents working on this project.
> Read it **before** making any changes.

---

## Project Overview

A Python CLI tool (`semsearch`) that provides **semantic search over local documents** using LangChain + PostgreSQL + pgvector. It ingests files (PDF, CSV, JSON, TXT, MD), splits them into chunks, generates embeddings via API providers, and stores them in PostgreSQL with pgvector for similarity search.

**Key package**: `pg-semantic-search` (PyPI) / `semsearch` (import name)

---

## Environment

### NixOS

This project runs on **NixOS**. The development environment is defined in `flake.nix`.

```bash
# Enter the dev shell (sets up venv, installs deps automatically)
nix develop

# Run any command inside the dev shell
nix develop --command bash -c "python -c 'print(42)'"
nix develop --command bash -c "pytest --collect-only"
nix develop --command bash -c "semsearch --help"
```

**Always use `nix develop --command`** for running Python commands. Direct `pip install` or `python` calls outside the nix shell will fail due to missing system libraries (libstdc++, libz, etc.).

### Missing Libraries

If you encounter `ImportError` for `.so` files, add the missing Nix package to `flake.nix`:
1. Add to `buildInputs` (e.g., `pkgs.zlib`)
2. Add to `LD_LIBRARY_PATH` in `shellHook`

---

## Architecture

### Source Layout

```
src/semsearch/
├── __init__.py       # Package marker + __version__
├── cli.py            # Typer CLI entry point (9 commands)
├── config.py         # Pydantic Settings + EmbeddingProviderConfig + RerankerProviderConfig + HnswConfig
├── embeddings.py     # Provider-dispatch factory (openai, ollama, openrouter, openai_compatible)
├── errors.py         # Exception hierarchy (SemSearchError base)
├── loaders.py        # File-type dispatch (pick_loader) + metadata injection (with_doc_type)
├── models.py         # Pydantic models (SearchResult, IngestResult, BatchIngestResult, DeleteResult)
├── reranker.py       # Generic reranker (OpenRouter, Jina, any compatible endpoint) — persistent httpx2.Client
├── server.py         # FastAPI HTTP server (POST /search, /ingest, /ingest-dir, DELETE /delete, GET /stats, /health)
├── service.py        # SemanticSearchService — main orchestration layer (cached vector size)
├── splitter.py       # RecursiveCharacterTextSplitter wrapper
└── store.py          # PGEngine/PGVectorStore construction + schema init (raw psycopg, HNSW config)

tests/
├── conftest.py       # Shared fixtures (MockEmbeddings, service, pg_url)
├── test_loaders.py   # Loader unit tests (7 tests)
├── test_embeddings.py # Embeddings factory tests (6 tests)
├── test_service_ingest.py    # Ingest integration tests (9 tests)
├── test_service_search.py    # Search integration tests (5 tests)
├── test_service_delete.py    # Delete integration tests (3 tests)
├── test_service_ingest_dir.py # Directory ingest tests (11 tests)
├── test_service_provider.py  # Provider routing tests (13 tests)
├── test_cli.py       # CLI tests (4 tests)
├── test_vector_size_cache.py # Vector size caching tests (7 tests)
├── test_hnsw_tuning.py      # HNSW config tests (7 tests)
├── test_reranker_pooling.py  # Reranker httpx tests (15 tests)
└── test_server.py            # FastAPI server tests (17 tests)

docs/
├── index.md          # Documentation index
├── getting-started.md # Installation, first search
├── configuration.md  # Environment variables
├── cli-reference.md  # All commands with examples
├── architecture.md   # Code structure, design decisions
├── api-reference.md  # Python API reference
├── database.md       # Schema, indexes, maintenance
├── providers.md      # Embedding provider details
└── development.md    # Testing, contributing
```

---

## Design Decisions (Critical)

These are non-negotiable. Violating them will cause spec non-compliance.

### 1. Write path is service-owned SQL

**NEVER** use `PGVectorStore.add_documents()`. All writes go through raw `psycopg` connections with parameterized SQL. This is for atomicity and precomputed embeddings.

```python
# CORRECT ✅
conn = psycopg.connect(db_url)
with conn.cursor() as cur:
    cur.execute("INSERT INTO ... ON CONFLICT ... DO UPDATE ...", params)
    conn.commit()

# WRONG ❌
store.add_documents(docs)  # Row-by-row commits, re-embeds everything
```

### 2. PGVectorStore is read-only

Use `PGVectorStore` ONLY for:
- `similarity_search_with_score()` — search

All writes (INSERT, UPDATE, DELETE) go through raw `psycopg` connections. This includes filtered deletes — build a `WHERE` clause from the filter dict instead of calling `PGVectorStore.delete()`.

### 3. Deterministic TEXT IDs

`langchain_id` is TEXT (not UUID) with format `"{source}::{chunk_index}"`. This enables UPSERT by natural key.

### 4. OpenRouter routing goes through `model_kwargs`

```python
# CORRECT ✅
OpenAIEmbeddings(
    model_kwargs={"extra_body": {"provider": {"order": ["openai", "together"]}}}
)

# WRONG ❌
OpenAIEmbeddings(
    extra_body={"provider": {"order": ["openai", "together"]}}  # DOES NOT EXIST
)
```

### 5. Score conversion

`score = 1.0 - cosine_distance`. LangChain returns distance, not similarity.

### 6. Double underscore for nested env vars

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai    # ✅ Correct
SEMSEARCH_EMBEDDING_PROVIDER_TYPE=openai      # ❌ Won't work
```

### 7. `allow_fallbacks=None` is OMITTED

When the user doesn't set it, the field must be absent from the dict — NOT `{"allow_fallbacks": true}`.

### 8. API-only embeddings

No local embeddings (HuggingFace removed). Providers: `openai`, `ollama`, `openrouter`, `openai_compatible`.

### 9. Lazy store initialization

`PGVectorStore.create_sync()` requires the table to exist. The store is lazily initialized via a property:

```python
@property
def store(self) -> PGVectorStore:
    if self._store is None:
        self._store = build_store(self.settings, self.engine, self.embedder)
    return self._store
```

### 10. Async close()

`PGEngine.close()` is async. Use `asyncio.new_event_loop()` to run it synchronously:

```python
def close(self) -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(self.engine.close())
    finally:
        loop.close()
```

### 11. URL normalization

Database URLs may use `postgres://` (not `postgresql://`). Normalize before use:

```python
if url.startswith("postgres://"):
    url = url.replace("postgres://", "postgresql://", 1)
```

### 12. Vector index strategy (DiskANN vs HNSW)

The project uses the best available vector index:

| pgvectorscale installed? | Vector dims | Index |
|-------------------------|-------------|-------|
| Yes | Any | DiskANN (SBQ compressed) |
| No | ≤2000 | HNSW |
| No | >2000 | None (sequential scan) |

DiskANN works for all dimensions (no 2000-dim limit like HNSW). When
vectorscale is detected, HNSW indexes are automatically upgraded to
DiskANN on `init_schema()`.

**DiskANN defaults:**
- `storage_layout = memory_optimized` (SBQ compression)
- `num_bits_per_dimension = 2` (16× compression; auto-downgrades to 1 for >900 dims)
- `num_neighbors = 50`
- `search_list_size = 100`

Override via `SEMSEARCH_DISKANN__*` env vars or `DiskANNConfig` in code.

```bash
SEMSEARCH_DISKANN__STORAGE_LAYOUT=memory_optimized
SEMSEARCH_DISKANN__NUM_BITS_PER_DIMENSION=2
```

### 13. Reranker (optional)

Reranking improves search precision by re-scoring results with a cross-encoder. Uses generic HTTP API (OpenRouter, Jina, etc.).

```bash
# Config
SEMSEARCH_RERANKER__BASE_URL=https://openrouter.ai/api/v1/rerank
SEMSEARCH_RERANKER__MODEL=cohere/rerank-v3.5

# Usage
semsearch search "query" --rerank --k 5
```

**Flow:** Vector search (k*4 candidates) → Reranker → Top-k results with `rerank_score` in metadata.

### 14. Custom config files

Use `--config` (`-c`) to specify a custom config file:

```bash
semsearch --config prod.env search "query"
semsearch -c staging.env stats
```

**Implementation:** Global callback stores path in `_config_path`, all commands use `get_settings(_config_path)`.

### 15. Connection reuse in batch operations

Methods that touch the database accept an optional `conn: psycopg.Connection | None = None` parameter. When `None`, the method creates and closes its own connection. When provided, the caller owns the connection lifecycle.

**Ownership rule:**
- `conn=None` → method owns it (create → use → close)
- `conn` provided → caller owns it (method uses it, never closes)

**Batch pattern (`ingest_dir`):**
```python
conn = self._get_conn()
try:
    for file_path in files:
        self.ingest(file_path, conn=conn)  # Reuse one connection
finally:
    conn.close()
```

**Connection counts:**
| Method | Connections |
|--------|-------------|
| `ingest()` standalone | 1 (merged read+write) |
| `ingest_dir(N files)` | 1 (batch) + 1 (prune) |
| `delete()` standalone | 1 |
| `reingest()` | 1 (shared by delete + ingest) |
| `stats()` | 1 |

---

## Provider Configuration

| Type | Class | Required | Default |
|------|-------|----------|---------|
| `openai` | `OpenAIEmbeddings` | `api_key` | `text-embedding-3-small` |
| `ollama` | `OllamaEmbeddings` | daemon running | `http://localhost:11434` |
| `openrouter` | `OpenAIEmbeddings` | `api_key` | `https://openrouter.ai/api/v1` |
| `openai_compatible` | `OpenAIEmbeddings` | `base_url` | `http://localhost:1234/v1` |

**Important**: For OpenRouter, use `check_embedding_ctx_length=False` in `OpenAIEmbeddings` to avoid context length validation errors.

---

## Database Schema

```sql
CREATE TABLE semsearch_chunks (
    langchain_id        TEXT PRIMARY KEY,
    embedding           vector(N),             -- N = provider dimension
    content             TEXT,
    langchain_metadata  JSONB,                 -- stored as JSON, cast to jsonb for GIN index
    source              TEXT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    document_hash       CHAR(64),
    UNIQUE (source, chunk_index)
);

-- Indexes (created separately, NOT by init_vectorstore_table)
-- HNSW only for vectors ≤2000 dim
CREATE INDEX {table}_hnsw_idx ON {table} USING hnsw (embedding vector_cosine_ops);
-- GIN index requires jsonb cast
CREATE INDEX {table}_metadata_gin_idx ON {table} USING gin ((langchain_metadata::jsonb) jsonb_path_ops);
-- Composite index for re-ingest lookups
CREATE INDEX {table}_source_chunk_idx ON {table} (source, chunk_index);
```

**Note**: `langchain_metadata` is stored as `json` type by langchain-postgres, but must be cast to `::jsonb` for GIN index and `jsonb_set()` operations.

---

## File Operations

### Loaders

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.txt`, `.md` | `TextLoader` | Single document |
| `.pdf` | `PyMuPDFLoader` | One doc per page |
| `.csv` | `CSVLoader` | One doc per row |
| `.json` | `JSONLoader` | `jq_schema=".[].content"` |

### Ingest Cases

- **CASE A**: Content unchanged → reuse embedding (no API call)
- **CASE B**: Content changed → re-embed
- **CASE C**: New chunk → embed
- **CASE D**: Stale tail → delete

---

## Running Commands

```bash
# Enter dev shell first
nix develop

# Then inside the shell:
semsearch --help
semsearch version
pytest --collect-only
pytest -v
```

### Outside dev shell (using --command)

```bash
nix develop --command bash -c "semsearch --help"
nix develop --command bash -c "pytest -v --tb=short"
```

### With test database

```bash
# Local PostgreSQL
export PGHOST="$HOME/Project/semantic-search/.pgsocket"
export TEST_DATABASE_URL="postgresql+psycopg://semsearch:test@/semsearch?host=$PGHOST"

nix develop --command bash -c "TEST_DATABASE_URL='$TEST_DATABASE_URL' pytest -v"
```

---

## Testing

### Test Structure

| File | Tests | Coverage |
|------|-------|----------|
| `test_loaders.py` | 7 | Loader dispatch, metadata injection |
| `test_embeddings.py` | 6 | Provider factory, OpenRouter routing |
| `test_service_ingest.py` | 9 | CASE A/B/C/D, idempotency |
| `test_service_search.py` | 5 | Search, filters, score range |
| `test_service_delete.py` | 5 | Delete by source, empty filter, raw SQL, filter result |
| `test_service_ingest_dir.py` | 15 | File discovery, prune, error handling, connection reuse |
| `test_service_provider.py` | 13 | OpenRouter routing, error propagation |
| `test_cli.py` | 4 | CLI commands, help text |
| `test_vector_size_cache.py` | 7 | Vector size caching, DB read |
| `test_hnsw_tuning.py` | 7 | HNSW config, env vars, index creation |
| `test_reranker_pooling.py` | 15 | httpx2.Client, retry, build_reranker |
| `test_server.py` | 11 | FastAPI endpoints, OpenAPI docs, reranker |

**Total: 115 tests**

### MockEmbeddings

For tests that don't need real embeddings:

```python
from tests.conftest import MockEmbeddings

embeddings = MockEmbeddings(dim=128)
result = embeddings.embed_query("test")  # Deterministic hash-based
```

### Testcontainers

Integration tests use testcontainers for PostgreSQL:

```python
@pytest.fixture(scope="session")
def pg_container():
    from testcontainers.community.postgres import PostgresContainer
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg
```

---

## Dependencies

### Runtime (core)

- langchain, langchain-core, langchain-postgres, langchain-community, langchain-text-splitters
- psycopg, pgvector, sqlalchemy
- pydantic, pydantic-settings
- typer, pymupdf, python-dotenv, jq

### Providers (lazy-loaded)

- `langchain-openai` — for openai, openrouter, openai_compatible
- `langchain-ollama` — for ollama

### HTTP Server (optional)

- `fastapi>=0.115.0` — HTTP API for AI agent integration
- `uvicorn[standard]>=0.30.0` — ASGI server

### Test

- pytest, pytest-asyncio, pytest-cov, testcontainers

---

## Task Tracking

See `TODO.md` for the full task list. Each task has a detailed spec in `TASKS/TASK-XXX-*.md`.

**Status**: 29/29 tasks complete ✅

### Phase 1: Core Implementation (Complete)

| Phase | Task | Status |
|-------|------|--------|
| 1 | Project Scaffolding | ✅ |
| 2 | Configuration Layer | ✅ |
| 3 | Data Models | ✅ |
| 4 | Document Loaders | ✅ |
| 5 | Text Splitter | ✅ |
| 6 | Embeddings Factory | ✅ |
| 7 | Database Store | ✅ |
| 8.1-2 | Service Skeleton & Stats | ✅ |
| 8.3 | Service Ingest | ✅ |
| 8.4 | Service Search | ✅ |
| 8.5 | Service Delete | ✅ |
| 8.6 | Service Ingest Dir | ✅ |
| 8.7 | Service Reingest | ✅ |
| 9 | CLI | ✅ |
| 10.1-2 | Unit Tests | ✅ |
| 10.3 | Integration Tests — Ingest | ✅ |
| 10.4 | Integration Tests — Search | ✅ |
| 10.5 | Integration Tests — Delete | ✅ |
| 10.6 | Integration Tests — Ingest Dir | ✅ |
| 10.7 | Integration Tests — Provider | ✅ |
| 10.8 | CLI Tests | ✅ |
| 11 | Documentation | ✅ |
| 12 | Verification | ✅ |

### Phase 1 Performance Optimization ✅

| Task | Status | File |
|------|--------|------|
| Benchmark Harness | ✅ | `TASKS/TASK-024-benchmark-harness.md` |
| Cache Vector Size + DB Read in stats() | ✅ | `TASKS/TASK-025-cache-vector-size.md` |
| Expose HNSW Tuning in Config | ✅ | `TASKS/TASK-026-hnsw-tuning.md` |
| Persistent httpx2.Client in Reranker | ✅ | `TASKS/TASK-027-reranker-httpx-pooling.md` |
| FastAPI HTTP Server (`semsearch serve`) | ✅ | `TASKS/TASK-028-fastapi-server.md` |
| Before/After Performance Benchmarks | ✅ | `TASKS/TASK-029-before-after-benchmarks.md` |

---

## Common Pitfalls

1. **`init_vectorstore_table` is NOT idempotent** — Check `information_schema.tables` first
2. **`Column` has no `primary_key` param** — Just use `Column("langchain_id", "TEXT")`
3. **`PGEngine.close()` is async** — Use `asyncio.new_event_loop()` for sync cleanup
4. **`collection_name` becomes a table name** — Validated against `/^[a-z_][a-z0-9_]{0,62}$/`
5. **NixOS requires `nix develop`** — Don't run Python directly outside the shell
6. **`postgres://` URLs** — Normalize to `postgresql://` for SQLAlchemy
7. **`jsonb_set` on json column** — Cast to `::jsonb` first, back to `::json`
8. **GIN index on json column** — Cast to `(langchain_metadata::jsonb) jsonb_path_ops`
9. **HNSW 2000 dim limit** — Skip index for vectors >2000 dimensions
10. **OpenRouter context length** — Use `check_embedding_ctx_length=False`
11. **`build_store` needs table** — Use lazy `store` property, call `init_schema` first
12. **Reranker must be closed** — `Reranker` has `httpx2.Client`; call `close()` or use context manager
13. **FastAPI lifespan owns service** — `create_app()` closes service on shutdown; pre-built services via `service=` param are caller's responsibility
14. **HNSW `ef_search` is table-level** — Set via `ALTER TABLE ... SET (hnsw.ef_search = N)`; users can override per-session with `SET hnsw.ef_search = N`
15. **`_get_vector_size_from_db` needs table** — Returns `None` if table doesn't exist; falls back to API probe

---

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | User-facing documentation |
| `AGENTS.md` | This file — agent instructions |
| `SPEC.md` | Full technical specification |
| `PLAN.md` | Performance optimization plan (Phase 1) |
| `TODO.md` | Task tracking with dependencies |
| `TASKS/` | Individual task specs (29 files) |
| `docs/` | Comprehensive documentation |
| `docs/api-reference.md` | HTTP API reference for serve mode |
| `BENCHMARKS.md` | Performance benchmark results and methodology |
| `semantic-search-performance-audit.md` | Performance audit findings |
