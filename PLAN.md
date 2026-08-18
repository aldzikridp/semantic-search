# Implementation Plan — Semantic Search Service

> **Spec**: `SPEC.md` (1983 lines, 16 sections)
> **Status**: Ready for implementation
> **Estimated effort**: 5–7 working days for a solo engineer

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Phase 1: Project Scaffolding](#3-phase-1-project-scaffolding)
4. [Phase 2: Configuration Layer](#4-phase-2-configuration-layer)
5. [Phase 3: Data Models](#5-phase-3-data-models)
6. [Phase 4: Document Loaders](#6-phase-4-document-loaders)
7. [Phase 5: Text Splitter](#7-phase-5-text-splitter)
8. [Phase 6: Embeddings Factory](#8-phase-6-embeddings-factory)
9. [Phase 7: Database Store](#9-phase-7-database-store)
10. [Phase 8: Core Service](#10-phase-8-core-service)
11. [Phase 9: CLI](#11-phase-9-cli)
12. [Phase 10: Tests](#12-phase-10-tests)
13. [Phase 11: Documentation & Polish](#13-phase-11-documentation--polish)
14. [Phase 12: Verification & Acceptance](#14-phase-12-verification--acceptance)
15. [Risk Register](#15-risk-register)
16. [File Creation Order (Dependency Graph)](#16-file-creation-order-dependency-graph)

---

## 1. Overview

This plan implements a Python CLI library (`semsearch`) that provides semantic search over local documents using LangChain + PostgreSQL + pgvector. The implementation follows the spec's architecture exactly: service-owned write path (SQLAlchemy transactions), `PGVectorStore` for search/delete, configurable embedding providers (OpenAI, HuggingFace, Ollama, OpenRouter, OpenAI-compatible), and a `typer` CLI.

**Key architectural decisions from the spec that must be honored:**
- Write path is service-owned SQL (NOT `PGVectorStore.add_documents`) for atomicity and precomputed embeddings.
- `PGVectorStore` is used ONLY for `similarity_search_with_score()` and `delete(filter=...)`.
- `langchain_id` is TEXT with deterministic IDs `"{source}::{chunk_index}"`.
- Content-hash caching (CASE A/B/C/D) avoids re-embedding unchanged chunks.
- Search scores are converted: `score = 1.0 - cosine_distance`.
- OpenRouter routing uses `model_kwargs={"extra_body": {...}}` (NOT a direct `extra_body=` kwarg).

---

## 2. Prerequisites & Environment Setup

**Before writing any code**, verify:

- [ ] Python 3.11+ available (`python3 --version`)
- [ ] Docker available for testcontainers (`docker --version`)
- [ ] Project directory exists at `/home/master-x/Project/semantic-search`
- [ ] Git initialized (already done)

**No external services needed for development** — HuggingFace (default provider) runs locally, and testcontainers spins up Postgres automatically for tests.

---

## 3. Phase 1: Project Scaffolding

**Goal**: Create the directory structure, dependency files, and package boilerplate.

**Files to create:**

### 3.1 `requirements.txt`

Pin all dependencies per SPEC §2:

```text
langchain==0.2.16
langchain-core==0.2.41
langchain-postgres==0.0.17
langchain-community==0.2.16
langchain-text-splitters==0.2.4
langchain-openai==0.1.22
langchain-huggingface==0.0.3
langchain-ollama==0.1.0
psycopg[binary]==3.2.1
pgvector==0.3.5
sqlalchemy==2.0.32
pydantic==2.8.2
pydantic-settings==2.4.0
typer==0.12.3
pymupdf==1.24.10
python-dotenv==1.0.1
sentence-transformers==3.0.1
jq==1.7.0
pytest==8.3.2
pytest-asyncio==0.23.8
testcontainers==4.8.1
pytest-cov  # for coverage reporting
```

### 3.2 `pyproject.toml` (update existing)

Add the `[project]` section with:
- `name = "semsearch"`
- `version = "0.1.0"`
- `requires-python = ">=3.11"`
- `dependencies` list matching `requirements.txt` (minus test deps)
- `[project.scripts]` → `semsearch = "semsearch.cli:app"`
- `[tool.pytest.ini_options]` → `testpaths = ["tests"]`

### 3.3 `.env.example`

Copy from SPEC §6.2 verbatim. This is the canonical reference for env var names.

### 3.4 `scripts/init_db.sql`

Copy from SPEC §4.1. The SQL for creating extensions, role, and database.

### 3.5 Directory structure

```
src/semsearch/__init__.py          # empty, marks as package
src/semsearch/errors.py            # exception hierarchy
tests/__init__.py                  # empty
tests/conftest.py                  # placeholder (filled in Phase 10)
```

### 3.6 `.gitignore` (update existing)

Ensure it includes: `.venv/`, `__pycache__/`, `*.egg-info/`, `.env`, `.pytest_cache/`, `htmlcov/`, `.coverage`

**Verification**: `pip install -e .` succeeds, `semsearch --help` shows placeholder.

---

## 4. Phase 2: Configuration Layer

**Goal**: Implement `config.py` with typed env-var loading.

**File**: `src/semsearch/config.py`

**Implementation steps:**

1. Define `_TABLE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")` for SQL identifier validation.
2. Define `EmbeddingProviderConfig(BaseModel)` with all fields from SPEC §6.1:
   - `type: Literal["openai", "huggingface", "ollama", "openai_compatible", "openrouter"]`
   - `model: str`
   - `api_key: SecretStr | None = None`
   - `base_url: str | None = None`
   - `device: str = "cpu"`
   - OpenRouter routing fields: `provider_order`, `provider_allow_fallbacks`, `provider_ignore`, `provider_only`, `provider_require_parameters`, `provider_data_collection`, `provider_max_price`
3. Define `Settings(BaseSettings)` with:
   - `model_config = SettingsConfigDict(env_prefix="SEMSEARCH_", env_file=".env", extra="ignore", env_nested_delimiter="__")`
   - `database_url`, `collection_name` (with validator), `embedding_provider` (default: HuggingFace), `chunk_size`, `chunk_overlap`, `default_k`, `recreate_collection_on_init`
4. Validate `collection_name` against `_TABLE_NAME_RE`.

**Critical gotcha to implement correctly:**
- `env_nested_delimiter="__"` is what allows `SEMSEARCH_EMBEDDING_PROVIDER__TYPE` to map to `settings.embedding_provider.type`. A single underscore (`SEMSEARCH_EMBEDDING_PROVIDER_TYPE`) will NOT work.

**Verification**: Unit test that env vars parse correctly into nested config.

---

## 5. Phase 3: Data Models

**Goal**: Define all Pydantic models for the API contract.

**File**: `src/semsearch/models.py`

**Models to implement (per SPEC §7.1):**

1. `SearchResult` — with `id`, `content`, `score`, `source`, `chunk_index`, `page`, `row`, `doc_type`, `metadata`
2. `DeleteResult` — with `deleted_count`, `filter`
3. `IngestResult` — with `source`, `chunks_added`, `chunks_reused`, `chunks_updated`, `chunks_pruned`, `ingested_at`
4. `BatchAggregate` — with `chunks_added`, `chunks_reused`, `chunks_updated`, `chunks_pruned`
5. `BatchIngestResult` — with `dir`, `files_discovered`, `files_skipped_unsupported`, `files_attempted`, `files_succeeded`, `files_failed`, `failed_files`, `aggregate`, `elapsed_seconds`, `pruned_sources`, `pruned_chunks`

**Verification**: Models instantiate correctly, JSON serialization works.

---

## 6. Phase 4: Document Loaders

**Goal**: Implement file-type dispatch and metadata injection.

**File**: `src/semsearch/loaders.py`

**Implementation steps:**

1. Define `_DOC_TYPE_BY_EXT` mapping: `{".txt": "text", ".md": "text", ".pdf": "pdf", ".csv": "csv", ".json": "json"}`.
2. Implement `pick_loader(path: Path) -> Callable[[], list[Document]]`:
   - `.txt`, `.md` → `TextLoader(path, encoding="utf-8")`
   - `.pdf` → `PyMuPDFLoader(str(path))`
   - `.csv` → `CSVLoader(str(path))`
   - `.json` → `JSONLoader(str(path), jq_schema=".[].content", text_content=False)`
   - Raise `ValueError` for unsupported extensions
   - Raise `FileNotFoundError` if path doesn't exist
3. Implement `with_doc_type(docs, path)` to inject `source` and `doc_type` into each doc's metadata.

**Key insight**: Loaders do NOT set `doc_type` natively — the service wraps loader output to inject it based on file extension.

**Verification**: Unit tests U-1 through U-4.

---

## 7. Phase 5: Text Splitter

**Goal**: Thin wrapper around `RecursiveCharacterTextSplitter`.

**File**: `src/semsearch/splitter.py`

**Implementation steps:**

1. `build_splitter(chunk_size, chunk_overlap)` — returns `RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)`
2. `split_documents(docs, chunk_size, chunk_overlap)` — calls `build_splitter(...).split_documents(docs)`

**Verification**: Splitter produces ≥ input docs, metadata preserved.

---

## 8. Phase 6: Embeddings Factory

**Goal**: Provider-dispatch factory with OpenRouter routing translation.

**File**: `src/semsearch/embeddings.py`

**Implementation steps:**

1. Implement `build_embedder(settings: Settings) -> Embeddings`:
   - `openai` → `OpenAIEmbeddings(api_key=..., model=...)`
   - `huggingface` → `HuggingFaceEmbeddings(model_name=..., model_kwargs={"device": ...})`
   - `ollama` → `OllamaEmbeddings(base_url=..., model=...)`
   - `openrouter` → `OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1", model=..., model_kwargs={"extra_body": {"provider": {...}}})`
   - `openai_compatible` → `OpenAIEmbeddings(base_url=..., model=...)`
2. Implement `_build_openrouter_routing(cfg)`:
   - Maps spec fields to OpenRouter `provider.*` keys
   - **Only emits `allow_fallbacks` if explicitly set** (None = omit, NOT emit True)
   - Normalizes all provider slugs to lowercase
   - Returns `{"provider": {...}}` or `{}` if no routing fields set
3. Implement `_require(secret, name)` for credential validation.

**Critical implementation detail**: OpenRouter routing goes through `model_kwargs={"extra_body": {...}}`, NOT a direct `extra_body=` kwarg. This is because `OpenAIEmbeddings` doesn't accept `extra_body` as a direct parameter — it accepts `model_kwargs` which gets unpacked into `client.create(**model_kwargs)`.

**Verification**: Unit tests U-5 through U-9.

---

## 9. Phase 7: Database Store

**Goal**: PGEngine/PGVectorStore construction and schema initialization.

**File**: `src/semsearch/store.py`

**Implementation steps:**

1. Define column constants:
   - `CUSTOM_COLUMNS = [Column("source", "TEXT", nullable=False), Column("chunk_index", "INTEGER", nullable=False), Column("document_hash", "CHAR(64)")]`
   - `CUSTOM_COLUMN_NAMES = [c.name for c in CUSTOM_COLUMNS]`
   - `LC_CONTENT_COLUMN = "content"`, `LC_METADATA_COLUMN = "langchain_metadata"`, `LC_ID_COLUMN = "langchain_id"`
2. `build_engine(settings) → PGEngine.from_connection_string(url=settings.database_url)`
3. `build_store(settings, engine, embedder) → PGVectorStore.create_sync(...)` with explicit column overrides
4. `init_schema(settings, engine, vector_size, *, recreate)`:
   - Check `information_schema.tables` for idempotency
   - If `recreate=True`, `DROP TABLE IF EXISTS ... CASCADE`
   - Call `engine.init_vectorstore_table(table_name=, vector_size=, id_column=Column("langchain_id", "TEXT"), content_column="content", metadata_json_column="langchain_metadata", metadata_columns=CUSTOM_COLUMNS)`
   - Create indexes separately (HNSW, GIN, composite) with `CREATE INDEX IF NOT EXISTS`

**Critical verification needed**: Check if `Column("langchain_id", "TEXT", primary_key=True)` works in `langchain-postgres==0.0.17`. If `primary_key` isn't a real kwarg, drop it — the id column is implicitly primary key by role.

**Verification**: Schema creation works, idempotent on second call, `--recreate` drops and recreates.

---

## 10. Phase 8: Core Service

**Goal**: Implement `SemanticSearchService` — the main orchestration layer.

**File**: `src/semsearch/service.py`

This is the largest and most complex module. Implement in this order:

### 10.1 Service skeleton

```python
class SemanticSearchService:
    @classmethod
    def from_settings(cls, settings) -> "SemanticSearchService": ...
    def __enter__(self): ...
    def __exit__(self, ...): ...
    def close(self): ...
    def init_schema(self, *, recreate=False): ...
```

### 10.2 `stats()` method

Simplest method — implement first to validate the store connection:
- Query `SELECT COUNT(*)` and `SELECT COUNT(DISTINCT source)` from the chunks table
- Query top 20 sources by chunk count
- Return the stats dict per SPEC §7.6

### 10.3 `ingest()` method — The Write Path

This is the most complex method. Steps:

1. Load file: `loader = pick_loader(path); docs = loader(); docs = with_doc_type(docs, path)`
2. Split: `chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)`
3. Compute hashes: `hashes = [sha256(chunk.page_content) for chunk in chunks]`
4. Fetch existing rows: `SELECT langchain_id, chunk_index, document_hash FROM {table} WHERE source = :source ORDER BY chunk_index`
5. Classify each chunk into CASE A/B/C:
   - CASE A: existing row found AND hash matches AND `not reembed_unchanged` → reuse
   - CASE B: existing row found AND (hash differs OR `reembed_unchanged`) → re-embed
   - CASE C: no existing row → embed new
6. Compute CASE D: if `len(chunks) < max_existing_chunk_index + 1`, stale tail exists
7. Batch-embed only CASE B + CASE C texts: `vectors = embedder.embed_documents(texts_to_embed)`
8. Execute ALL writes in ONE `engine.begin()` transaction:
   - CASE A: `UPDATE ... SET langchain_metadata = jsonb_set(..., '{ingested_at}', :now)`
   - CASE B/C: `INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE SET ...`
   - CASE D: `DELETE WHERE source = :source AND chunk_index >= :new_len`
9. Return `IngestResult`

**Critical**: Deterministic IDs are `f"{source}::{chunk_index}"`.

### 10.4 `search()` method — The Read Path

1. Default `k` to `settings.default_k`; validate `1 <= k <= 50`
2. Call `store.similarity_search_with_score(query, k=k, filter=filter)`
3. Convert: `score = 1.0 - distance` (LangChain returns distance, not similarity)
4. Wrap each `(Document, score)` in `SearchResult`
5. Return sorted by score DESC

### 10.5 `delete()` method

1. Handle empty filter (`{}`) special case:
   - `SELECT COUNT(*) FROM {table}` to get count
   - Execute `DELETE FROM {table}` directly via SQLAlchemy (bypass `PGVectorStore.delete` which may not handle empty filter)
2. For non-empty filter:
   - Open transaction
   - `SELECT COUNT(*) FROM {table} WHERE <filter predicates>` — this requires translating the filter dict to SQL WHERE clause. **Alternative**: use the same filter translation that `PGVectorStore` uses, OR accept that `deleted_count` may not be perfectly accurate for complex filters and rely on `PGVectorStore.delete()` returning success.
   - Call `store.delete(filter=filter)`
   - Return `DeleteResult(deleted_count=count, filter=filter)`

**Design decision**: For the `deleted_count` computation, the spec says to do `SELECT COUNT(*) ... WHERE <filter>` in the same transaction. The challenge is that translating the rich filter dict (`$ilike`, `$and`, etc.) to SQL is non-trivial and is exactly what `PGVectorStore` does internally. Two approaches:
- **Approach A**: Implement our own filter-to-SQL translator (complex, error-prone, duplicates langchain-postgres logic)
- **Approach B**: For non-empty filters, execute a `SELECT COUNT(*)` using `PGVectorStore.similarity_search_with_score` with a very high `k` and count results. Wasteful but correct.
- **Approach C (recommended)**: Accept that for non-empty filters, `deleted_count` is best-effort. Call `store.delete(filter=filter)` and if it succeeds, query the table count after deletion to compute the delta. Or, keep a "before" count and an "after" count.

**Recommendation**: Use Approach C — count before and after in the same transaction. This is simple, correct, and doesn't require reimplementing filter translation.

### 10.6 `ingest_dir()` method

1. Validate `dir_path` exists and is a directory
2. Walk `dir_path.glob(glob)`, filter by:
   - `SUPPORTED_EXTENSIONS` (`.txt`, `.md`, `.pdf`, `.csv`, `.json`)
   - Skip hidden files (name starts with `.`)
   - Skip symlinks unless `follow_symlinks=True`
   - Skip files matching `exclude` patterns (fnmatch)
3. Sort for deterministic order
4. Loop: call `self.ingest(file, reembed_unchanged=reembed_unchanged)` per file
   - If `continue_on_error=True`: catch exceptions, log, collect in `failed_files`
   - If `continue_on_error=False`: let first exception propagate
5. If `prune=True`:
   - Query DB for all distinct `source` values starting with `str(dir_path) + "/"`
   - Compute set difference with files just ingested
   - For each orphan source: if `prune_dry_run=False`, call `self.delete(filter={"source": s})`; always add to `pruned_sources`
6. Return `BatchIngestResult`

### 10.7 `reingest()` method

Per SPEC §8.7 gap note, implement as:
- Delete all chunks for the source: `self.delete(filter={"source": str(path)})`
- Ingest fresh: `self.ingest(path, reembed_unchanged=True)`
- NOT atomic across delete+ingest (different transaction paths). Document this clearly.

### 10.8 Exception hierarchy (`errors.py`)

```python
class SemSearchError(Exception): ...
class FileIngestError(SemSearchError): ...
class SearchError(SemSearchError): ...
class DeleteError(SemSearchError): ...
class SchemaMismatchError(SemSearchError): ...
class ProviderConfigError(SemSearchError): ...
```

**Verification**: All integration tests I-1 through I-43 pass.

---

## 11. Phase 9: CLI

**Goal**: Implement all `typer` subcommands.

**File**: `src/semsearch/cli.py`

**Implementation steps:**

1. Create `app = typer.Typer(help="Semantic search over local documents")`
2. `init` command:
   - `--recreate` flag, `--yes` flag
   - If `--recreate` and not `--yes`: prompt for confirmation
   - Call `svc.init_schema(recreate=recreate)`
3. `ingest` command:
   - Args: `path: Path`
   - Flags: `--force`, `--provider`, `--provider-model`, `--provider-order`, `--provider-allow-fallbacks`, `--provider-ignore`, `--provider-base-url`, `--provider-api-key`
   - Build settings, apply CLI overrides to `embedding_provider`
   - Call `svc.ingest(path, reembed_unchanged=force)`
   - Print JSON result to stdout
4. `ingest-dir` command:
   - Args: `dir_path: Path`
   - Flags: `--glob`, `--exclude` (repeatable), `--prune`, `--dry-run`, `--no-continue-on-error`, `--follow-symlinks`, `--force`
   - Call `svc.ingest_dir(...)`
   - Print JSON result to stdout
5. `search` command:
   - Args: `query: str`
   - Flags: `--k`, `--filter`
   - Call `svc.search(query, k=k, filter=filter)`
   - Print JSON result to stdout
6. `delete` command:
   - Flags: `--filter`, `--all`, `--yes`
   - If `--all` and not `--yes`: error + exit
   - Call `svc.delete(filter=filter)`
   - Print JSON result to stdout
7. `stats` command:
   - No args
   - Call `svc.stats()`
   - Print JSON result to stdout
8. `reingest` command:
   - Args: `path: Path`
   - Call `svc.reingest(path)`
   - Print JSON result to stdout

**Error handling**: Wrap all commands in try/except for `SemSearchError` → print `{"error": "...", "type": "..."}` to stderr, exit code 1.

**Verification**: CLI tests C-1 through C-6.

---

## 12. Phase 10: Tests

**Goal**: Achieve ≥85% coverage with unit + integration tests.

### 12.1 Test fixtures (`tests/conftest.py`)

```python
@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg

@pytest.fixture(scope="session")
def settings(pg_container):
    return Settings(
        database_url=pg_container.get_connection_url(driver="psycopg"),
        embedding_provider=EmbeddingProviderConfig(
            type="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

@pytest.fixture
def service(settings):
    with SemanticSearchService.from_settings(settings) as svc:
        svc.init_schema(recreate=True)  # fresh table per test
        yield svc
```

### 12.2 Test files

| File | Tests | Priority |
|------|-------|----------|
| `tests/test_loaders.py` | U-1 through U-4 | High |
| `tests/test_embeddings.py` | U-5 through U-9 | High |
| `tests/test_service_ingest.py` | I-1, I-2, I-3, I-9, I-9b, I-10, I-10b, I-10c, I-10d, I-10e, I-11 | Critical |
| `tests/test_service_search.py` | I-4, I-5, I-15 through I-20 | Critical |
| `tests/test_service_delete.py` | I-6, I-7, I-8, I-12, I-13 | Critical |
| `tests/test_service_ingest_dir.py` | I-29 through I-43 | High |
| `tests/test_service_provider.py` | I-14, I-21 through I-28c | Medium |
| `tests/test_cli.py` | C-1 through C-6 | Medium |

### 12.3 Test data fixtures

Create `tests/fixtures/` with:
- `sample.txt` — multi-paragraph text file (~3 chunks worth)
- `sample.pdf` — 5-page PDF (generate or use a known small PDF)
- `sample.csv` — 10-row CSV with headers
- `sample.json` — JSON array with 5+ elements, each having `content` field

### 12.4 Key test patterns

**For idempotency tests (I-9, I-36)**:
```python
def test_reingest_no_api_calls(service, sample_txt, mock_embedder):
    service.ingest(sample_txt)
    service.ingest(sample_txt)  # second ingest
    # Assert mock_embedder.embed_documents.call_count == 0
```

**For CASE D tests (I-10d)**:
```python
def test_stale_tail_pruned(service, tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("chunk0 " * 200 + "chunk1 " * 200 + "chunk2 " * 200)
    service.ingest(file)
    # Shorten file
    file.write_text("chunk0 " * 200)
    result = service.ingest(file)
    assert result.chunks_pruned == 2
```

**Verification**: `pytest -v` passes all tests, `pytest --cov=src/semsearch` shows ≥85%.

---

## 13. Phase 11: Documentation & Polish

### 13.1 `README.md`

≤50 lines covering:
1. What it does (one sentence)
2. Quickstart (install, start Postgres, init, ingest, search)
3. Supported file types
4. Configuration (link to `.env.example`)
5. CLI reference (one-liner per command)

### 13.2 Final code review checklist

- [ ] All `PGVectorStore.add_documents()` calls are ABSENT (write path is service-owned)
- [ ] `model_kwargs={"extra_body": {...}}` for OpenRouter (not direct `extra_body=`)
- [ ] `score = 1.0 - distance` conversion in search
- [ ] `langchain_id` is TEXT, not UUID
- [ ] `env_nested_delimiter="__"` in Settings
- [ ] Double underscore in all `SEMSEARCH_EMBEDDING_PROVIDER__*` env vars
- [ ] Provider slugs normalized to lowercase
- [ ] `allow_fallbacks=None` is OMITTED, not emitted as True
- [ ] `init_schema` is idempotent (checks `information_schema.tables`)
- [ ] `collection_name` validated against `_TABLE_NAME_RE`
- [ ] `SecretStr` for API keys (no leakage in repr/logs)
- [ ] All indexes created: HNSW, GIN, composite (source, chunk_index)
- [ ] `UNIQUE (source, chunk_index)` constraint on the table
- [ ] `ON CONFLICT (source, chunk_index) DO UPDATE` in ingest SQL

---

## 14. Phase 12: Verification & Acceptance

Run the full acceptance checklist from SPEC §16:

```bash
# 1. Run all tests
pytest -v --tb=short

# 2. Check coverage
pytest --cov=src/semsearch --cov-report=term-missing

# 3. Manual smoke test
semsearch init
semsearch ingest ./tests/fixtures/sample.txt
semsearch search "test query" --k 3
semsearch delete --filter '{"source": "./tests/fixtures/sample.txt"}'
semsearch stats

# 4. Provider swap test
# Edit .env to use OpenAI (if key available), then:
semsearch init --recreate --yes
semsearch ingest ./tests/fixtures/sample.txt
semsearch search "test" --k 3

# 5. Directory ingest + prune test
semsearch ingest-dir ./tests/fixtures/
# Delete a file, then:
semsearch ingest-dir ./tests/fixtures/ --prune --dry-run
semsearch ingest-dir ./tests/fixtures/ --prune
```

---

## 15. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `Column("langchain_id", "TEXT", primary_key=True)` doesn't work in `langchain-postgres==0.0.17` | Medium | High | Read the installed library source first. If `primary_key` isn't supported, drop it — the id column is implicitly PK by role. |
| `PGVectorStore` filter operators differ from deprecated `PGVector` | Medium | Medium | Read `langchain_postgres/v2/` filter translator source before writing filter tests. Adjust operator table in code comments. |
| `PGVectorStore.delete(filter={})` doesn't work (returns False) | High | Low | Already handled in spec — bypass with direct `DELETE FROM` for empty filter. |
| `init_vectorstore_table` is NOT idempotent | High | Medium | Already handled in spec — check `information_schema.tables` first. |
| `PGEngine.close()` is async but we need sync | Medium | Medium | Use `asyncio.run_coroutine_threadsafe(...).result()` or `asyncio.run()` in `close()`. |
| `langchain-postgres` version 0.0.17 has breaking API changes from examples | Low | High | Pin exact version. Read installed source, not just docs. |
| Testcontainers startup slow in CI | Low | Low | Use `scope="session"` for the container fixture. |

---

## 16. File Creation Order (Dependency Graph)

Files must be created in this order due to import dependencies:

```
Phase 1 (no dependencies):
  src/semsearch/__init__.py
  src/semsearch/errors.py
  requirements.txt
  pyproject.toml
  .env.example
  scripts/init_db.sql
  tests/__init__.py

Phase 2 (depends on: nothing):
  src/semsearch/config.py

Phase 3 (depends on: nothing):
  src/semsearch/models.py

Phase 4 (depends on: langchain-core):
  src/semsearch/loaders.py

Phase 5 (depends on: langchain-text-splitters):
  src/semsearch/splitter.py

Phase 6 (depends on: config, errors):
  src/semsearch/embeddings.py

Phase 7 (depends on: config, langchain-postgres):
  src/semsearch/store.py

Phase 8 (depends on: ALL above):
  src/semsearch/service.py

Phase 9 (depends on: service, config):
  src/semsearch/cli.py

Phase 10 (depends on: ALL above):
  tests/conftest.py
  tests/test_loaders.py
  tests/test_embeddings.py
  tests/test_service_ingest.py
  tests/test_service_search.py
  tests/test_service_delete.py
  tests/test_service_ingest_dir.py
  tests/test_service_provider.py
  tests/test_cli.py
  tests/fixtures/  (sample files)

Phase 11:
  README.md
```

---

## Summary of Critical Implementation Details

These are the "gotchas" that, if missed, will cause the implementation to fail spec compliance:

1. **Write path is service-owned SQL** — `PGVectorStore.add_documents()` is NEVER used. All writes go through `engine.begin()` + parameterized SQLAlchemy statements.

2. **OpenRouter routing** — `model_kwargs={"extra_body": {"provider": {...}}}`, NOT `extra_body=` as a direct kwarg.

3. **Score conversion** — `score = 1.0 - cosine_distance`. LangChain returns distance, not similarity.

4. **Deterministic TEXT IDs** — `langchain_id` is `f"{source}::{chunk_index}"` with TEXT type, not UUID.

5. **Idempotent schema init** — Check `information_schema.tables` before calling `init_vectorstore_table()` (which is NOT idempotent).

6. **Double underscore in env vars** — `SEMSEARCH_EMBEDDING_PROVIDER__TYPE` (not `_TYPE`).

7. **`allow_fallbacks=None` omits the field** — Do NOT emit `True` as default; only include if explicitly set.

8. **`init_vectorstore_table` creates NO indexes** — HNSW, GIN, and composite indexes must be created separately with `CREATE INDEX IF NOT EXISTS`.

9. **Empty filter bypass** — `delete({})` bypasses `PGVectorStore.delete()` and issues direct `DELETE FROM`.

10. **CASE D cleanup** — Stale tail chunks deleted in the SAME transaction as new chunks (atomic).
