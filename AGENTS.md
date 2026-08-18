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
├── cli.py            # Typer CLI entry point
├── config.py         # Pydantic Settings + EmbeddingProviderConfig
├── embeddings.py     # Provider-dispatch factory (openai, ollama, openrouter, openai_compatible)
├── errors.py         # Exception hierarchy (SemSearchError base)
├── loaders.py        # File-type dispatch (pick_loader) + metadata injection (with_doc_type)
├── models.py         # Pydantic models (SearchResult, IngestResult, etc.)
├── splitter.py       # RecursiveCharacterTextSplitter wrapper
└── store.py          # PGEngine/PGVectorStore construction + schema init
```

### Planned (not yet implemented)

```
src/semsearch/
└── service.py        # SemanticSearchService — main orchestration layer (TASK-008+)

tests/
├── conftest.py       # Shared fixtures (PGVectorStore container, service)
├── test_loaders.py
├── test_embeddings.py
├── test_service_ingest.py
├── test_service_search.py
├── test_service_delete.py
├── test_service_ingest_dir.py
├── test_service_provider.py
└── test_cli.py
```

---

## Design Decisions (Critical)

These are non-negotiable. Violating them will cause spec non-compliance.

### 1. Write path is service-owned SQL

**NEVER** use `PGVectorStore.add_documents()`. All writes go through `engine.begin()` + parameterized SQLAlchemy statements. This is for atomicity and precomputed embeddings.

### 2. PGVectorStore is read-only

Use `PGVectorStore` ONLY for:
- `similarity_search_with_score()` — search
- `delete(filter=...)` — deletion

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

---

## Provider Configuration

| Type | Class | Required | Default |
|------|-------|----------|---------|
| `openai` | `OpenAIEmbeddings` | `api_key` | `text-embedding-3-small` |
| `ollama` | `OllamaEmbeddings` | daemon running | `http://localhost:11434` |
| `openrouter` | `OpenAIEmbeddings` | `api_key` | `https://openrouter.ai/api/v1` |
| `openai_compatible` | `OpenAIEmbeddings` | `base_url` | `http://localhost:1234/v1` |

---

## Database Schema

```sql
CREATE TABLE semsearch_chunks (
    langchain_id        TEXT PRIMARY KEY,
    embedding           vector(1536),          -- dim depends on provider
    content             TEXT,
    langchain_metadata  JSONB,
    source              TEXT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    document_hash       CHAR(64),
    UNIQUE (source, chunk_index)
);

-- Indexes (created separately, NOT by init_vectorstore_table)
CREATE INDEX {table}_hnsw_idx ON {table} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX {table}_metadata_gin_idx ON {table} USING gin (langchain_metadata jsonb_path_ops);
CREATE INDEX {table}_source_chunk_idx ON {table} (source, chunk_index);
```

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

### Test

- pytest, pytest-asyncio, pytest-cov, testcontainers

---

## Task Tracking

See `TODO.md` for the full task list. Each task has a detailed spec in `TASKS/TASK-XXX-*.md`.

**Status as of last update**: 7/23 tasks complete (Phase 1-7 done)

| Phase | Task | Status |
|-------|------|--------|
| 1 | Project Scaffolding | ✅ |
| 2 | Configuration Layer | ✅ |
| 3 | Data Models | ✅ |
| 4 | Document Loaders | ✅ |
| 5 | Text Splitter | ✅ |
| 6 | Embeddings Factory | ✅ |
| 7 | Database Store | ✅ |
| 8 | Core Service | ⏳ Next |
| 9 | CLI | ⏳ |
| 10 | Tests | ⏳ |

---

## Common Pitfalls

1. **`init_vectorstore_table` is NOT idempotent** — Check `information_schema.tables` first
2. **`Column` has no `primary_key` param** — Just use `Column("langchain_id", "TEXT")`
3. **`PGEngine.close()` is async** — Use `asyncio.run()` for sync cleanup
4. **`collection_name` becomes a table name** — Validated against `/^[a-z_][a-z0-9_]{0,62}$/`
5. **NixOS requires `nix develop`** — Don't run Python directly outside the shell

---

## Documentation Files

| File | Purpose |
|------|---------|
| `SPEC.md` | Full technical specification (1983 lines) |
| `PLAN.md` | Implementation plan with phases |
| `TODO.md` | Task tracking with dependencies |
| `TASKS/` | Individual task specs |
| `AGENTS.md` | This file — agent instructions |
