# Semantic Search Service — Technical Specification

> **Document**: `spec.md`
> **Project**: `pg-semantic-search`
> **Version**: 1.0
> **Status**: Draft for implementation
> **Target Audience**: Backend / Python engineers implementing the service

---

## 1. Overview & Goals

### 1.1 Purpose

This specification describes a Python program that provides **semantic search** over a small corpus of local documents. It uses **LangChain** for orchestration and **PostgreSQL + pgvector** as the vector store. The program is a library plus a thin CLI — it is **not** a web service. It is intended for datasets under ~1,000 documents, where a single `documents` table is sufficient and no sharding or async ingestion pipeline is required.

### 1.2 Goals

1. **Ingest** heterogeneous local sources: plain text (`.txt`, `.md`), PDFs, and structured tabular records (`.csv`, `.json`).
2. **Embed** text chunks using a **configurable embedding provider** — OpenAI, HuggingFace, Ollama, OpenRouter, or any OpenAI-compatible endpoint — selected at runtime via configuration, with no code changes required to switch providers.
3. **Store** embeddings and metadata in PostgreSQL with the `pgvector` extension, using LangChain's `PGVectorStore` integration as the storage abstraction.
4. **Search** the corpus via cosine similarity search, returning the top-k matching chunks along with their scores and provenance metadata.
5. **Remove** entries from the store by source file or by arbitrary metadata filter, so the corpus can be maintained over time (e.g. re-ingest an updated PDF after deleting its previous chunks).
6. **Ingest directories recursively** — walk a directory tree and ingest every supported file in a single batch, with optional `--prune` to keep the store in sync when files are deleted or renamed on disk.

### 1.3 Non-Goals

The following are explicitly **out of scope** for v1:

- Multi-tenant / per-user collections (single shared collection only).
- RAG / RetrievalQA chains and LLM-generated answers — v1 returns raw matching chunks only.
- Async ingestion pipelines, message queues, or background workers.
- Hybrid (BM25 + vector) search.
- Production deployment, authentication, rate limiting, observability dashboards.
- Scale beyond ~1k documents (no HNSW tuning, no partitioning).

### 1.4 Success Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| SC-1 | Ingesting a 50-page PDF produces ≥1 chunk per page, all stored with correct `source` metadata. | Test `test_ingest_pdf` |
| SC-2 | Switching `EMBEDDING_PROVIDER` from `openai` to `huggingface` requires only a config change, no source edits. | Test `test_provider_swap` |
| SC-3 | `similarity_search("how do I reset my password", k=5)` returns ≤5 chunks ranked by cosine similarity. | Test `test_similarity_search` |
| SC-4 | `delete({"source": "handbook.pdf"})` removes all and only chunks whose `source == "handbook.pdf"`. | Test `test_delete_by_source` |
| SC-5 | Subsequent searches return zero matches for deleted content. | Test `test_search_after_delete` |

---

## 2. Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11+ | LangChain ecosystem baseline |
| Orchestration | `langchain >= 0.2`, `langchain-core >= 0.2` | Standard building blocks |
| Vector store | `langchain-postgres >= 0.0.17` | Provides `PGVectorStore` + `PGEngine` (successor to deprecated `PGVector`). v0.0.17+ is required for filtered `delete(filter=...)`. We use `PGVectorStore` for **search only**; the write path is owned by the service via explicit SQL (see §3.2, §10.2) because `PGVectorStore.add_documents()` is not transactional and would double-embed precomputed vectors. |
| Embeddings (optional) | `langchain-openai`, `langchain-huggingface`, `langchain-ollama` | Loaded lazily based on `EMBEDDING_PROVIDER_TYPE`. `langchain-openai` is used for `openai`, `openai_compatible`, and `openrouter` types — all three go through OpenAI's SDK against different `base_url`s. OpenRouter routing is forwarded via `model_kwargs={"extra_body": {...}}` (NOT a direct `extra_body=` kwarg, which does not exist on `OpenAIEmbeddings`). |
| Document loaders | `langchain-community` (`TextLoader`, `PyMuPDFLoader`, `CSVLoader`, `JSONLoader`) | Battle-tested. Note: loaders do NOT set `doc_type` natively — service wraps loader output to inject `doc_type` based on file extension. |
| Text splitter | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) | Default chunking strategy |
| DB driver | `psycopg[binary]` (psycopg3) | Required by `langchain-postgres` for the sync API. Also used directly by the service for write-path SQL. |
| SQL toolkit | `sqlalchemy >= 2.0` | Used for parameterized writes in the service-owned write path (the same connection pool `PGEngine` uses). |
| PostgreSQL | 14+ with `pgvector` extension >= 0.5.0 | Required for `vector` column type |
| Config | `pydantic-settings` | Typed env-var loading with `env_nested_delimiter="__"` for nested `EmbeddingProviderConfig` |
| CLI | `typer` | Ergonomic subcommands |
| Testing | `pytest`, `pytest-asyncio`, `testcontainers-python` (Postgres) | Real DB in CI |

### Pinned versions (recommended `requirements.txt` baseline)

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
sentence-transformers==3.0.1     # required by langchain-huggingface
jq==1.7.0                       # required by JSONLoader
# test
pytest==8.3.2
pytest-asyncio==0.23.8
testcontainers==4.8.1
```

> **Why own the write path?** Three reasons: (a) `PGVectorStore.add_documents()` performs row-by-row inserts with per-row commits — not atomic, so a mid-loop failure leaves partial state; (b) we precompute embeddings for CASE B/C chunks to skip CASE A — `add_documents` would re-embed them; (c) we need `INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE` for idempotent UPSERT, which `add_documents` does not expose. Search and delete still go through `PGVectorStore` — those APIs are well-suited and use the same `PGEngine` connection pool.

---

## 3. System Architecture & Data Flow

### 3.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                            CLI (typer)                          │
│   init  |  ingest  |  search  |  delete  |  stats  |  reingest  │
└──────────────────────┬──────────────────────────────────────────┘
                       │  calls
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SemanticSearchService (core)                   │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌─────────┐ │
│  │ Loaders  │ → │  Splitter    │ → │  Embedder  │ → │ Store   │ │
│  │ (Text/   │   │ (Recursive   │   │ (OpenAI/  │   │ (PG     │ │
│  │  PDF/    │   │  Character)  │   │  HF/Ollama/│   │  Vector │ │
│  │  CSV/    │   └──────────────┘   │  OpenRouter)│   │  Store) │ │
│  │  JSON)   │                      └────────────┘   └────┬────┘ │
│  └──────────┘                                            │      │
│                                                           ▼      │
│                                              ┌────────────────────┐ │
│                                              │ PGEngine           │ │
│                                              │ (connection pool)  │ │
│                                              └─────────┬──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                                              ┌────────────────────┐
│                                              │  PostgreSQL +      │
│                                              │  pgvector          │
│                                              │  table:            │
│                                              │  semsearch_chunks  │
│                                              └────────────────────┘
```

### 3.2 Ingest Pipeline (Write Path)

```
Source file
    │
    ▼
[1] Loader.pick(path)         →  LangChain Document list (1 doc / page or row)
    │       └─ service calls with_doc_type(docs, path) to inject `source` + `doc_type`
    │          (loaders don't set doc_type natively)
    ▼
[2] Splitter.split_documents()→  Document list (1 doc / chunk)
    │       │
    │       └─ chunk_size=1000, chunk_overlap=200
    │       └─ splitter assigns chunk_index (0-based) per source
    ▼
[3] SELECT existing (chunk_index, document_hash, langchain_id) for source
    │       └─ one query, returns at most len(new_chunks) rows
    ▼
[4] Classify chunks → CASE A / B / C  (see §10)
    │
    ▼
[5] Embedder.embed_documents(CASE B + C texts only)  →  List[List[float]]
    │       └─ CASE A chunks skip this step entirely (no API call)
    │       └─ precomputed vectors are passed to step [6], NOT to PGVectorStore.add_documents
    │          (which would re-embed them)
    ▼
[6] Service-owned SQLAlchemy transaction (single BEGIN…COMMIT):
        │       ├─ CASE A:  UPDATE ingested_at WHERE langchain_id = :id
        │       ├─ CASE B:  INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE
        │       │             SET embedding=:vec, content=:text, document_hash=:hash,
        │       │                 langchain_metadata=:meta
        │       ├─ CASE C:  INSERT (same SQL, just no conflict expected)
        │       └─ CASE D:  DELETE WHERE source = :source AND chunk_index >= :new_len
        │                  (cleanup stale tail chunks when file shortened)
        └─ COMMIT (or ROLLBACK on any error — atomic, unlike PGVectorStore.add_documents)
```

> **Why service-owned SQL instead of `PGVectorStore.add_documents`?** Three reasons:
> 1. `PGVectorStore.add_documents()` performs row-by-row inserts with per-row commits — not atomic, so a mid-loop failure leaves partial state.
> 2. We precompute embeddings for CASE B/C chunks (to skip CASE A); `add_documents` would re-embed them.
> 3. We need `INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE` for idempotent UPSERT and `DELETE WHERE chunk_index >= :new_len` for CASE D — `add_documents` doesn't expose either.

### 3.3 Search Pipeline (Read Path)

```
Query string (+ optional filter dict)
    │
    ▼
[1] PGVectorStore.similarity_search_with_score(query, k=k, filter=filter)
        │
        │  (internally: embeds the query, runs the SQL below)
        │
        ├─ SELECT langchain_id, content, langchain_metadata, source, chunk_index,
        │         embedding <=> :query_vec AS distance    -- ← DISTANCE, not similarity
        │   FROM semsearch_chunks
        │   WHERE <filter predicates translated by langchain-postgres>
        │   ORDER BY embedding <=> :query_vec             -- ascending distance = best matches first
        │   LIMIT :k
        │
        └─ Returns List[Tuple[Document, distance]]
    │
    ▼
[2] Service converts distance → similarity:
        score = 1.0 - distance
    (LangChain returns cosine DISTANCE for the cosine strategy, NOT similarity.
     Higher score = more similar, matching user intuition. Tests must assert on
     the converted score, not the raw distance.)
    │
    ▼
[3] Service wraps each (Document, score) as SearchResult(
        id=langchain_id, content=document.page_content,
        source=document.metadata['source'],
        chunk_index=document.metadata['chunk_index'],
        page=document.metadata.get('page'),
        row=document.metadata.get('row'),
        doc_type=document.metadata.get('doc_type'),
        score=score, metadata=document.metadata,
    )
```

> **Filter dict syntax** is documented in §8.4. Supported operators: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`, `$between`, `$like`, `$ilike`, `$and`, `$or`, `$exists`, `$not`.

### 3.4 Delete Pipeline (Write Path)

Deletion delegates to `PGVectorStore.delete(filter=...)` for the actual DELETE, but we compute `deleted_count` ourselves first via `SELECT COUNT(*)` in the same transaction — `PGVectorStore.delete()` returns `Optional[bool]`, not a rowcount.

```
svc.delete(filter={...})
    │
    ▼
[1] BEGIN transaction (SQLAlchemy)
    │
    ▼
[2] SELECT COUNT(*) FROM semsearch_chunks WHERE <filter predicates>
    │       └─ returns the number of rows that will be deleted
    ▼
[3] PGVectorStore.delete(filter=filter)
    │       │
    │       │  Filter syntax is the same rich dict used by similarity_search:
    │       │    {"source": "handbook.pdf"}                      # delete a single file
    │       │    {"doc_type": "invoice", "year": 2024}         # delete by metadata conjunction
    │       │    {"$and": [{"source": {"$ilike": "docs/%"}}, {"doc_type": "pdf"}]}
    │       │
    │       └─ generates: DELETE FROM semsearch_chunks WHERE <filter predicates>
    │
    ▼
[4] COMMIT
    │
    ▼
[5] Return DeleteResult(deleted_count=<count from step 2>, filter=filter)
```

**Special case: `delete({})` (empty filter / `--all` flag)**

`PGVectorStore.delete(filter=None)` or `filter={}` may return `False` without deleting anything (the underlying implementation requires non-empty filter criteria). For `--all`, the service bypasses `PGVectorStore` and issues `DELETE FROM <table>` directly via SQLAlchemy, then returns `DeleteResult(deleted_count=<count>, filter={})`.

---

## 4. Database Schema & SQL

The `langchain-postgres` library (v0.0.14+) uses `PGVectorStore` with an explicit `PGEngine` and a single user-named table per collection. The schema is created via `engine.init_vectorstore_table(table_name=, vector_size=)` — no auto-creation magic. This replaces the deprecated `PGVector` two-table layout (`langchain_pg_collection` + `langchain_pg_embedding`).

### 4.1 Extension & Database Setup

Run once as a superuser:

```sql
-- Run as superuser (e.g. postgres)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Application role (one-time)
CREATE ROLE semsearch_app
    LOGIN PASSWORD 'change_me_in_prod'
    NOSUPERUSER NOCREATEDB NOCREATEROLE;

-- Database
CREATE DATABASE semsearch
    OWNER semsearch_app
    ENCODING 'UTF8';

-- Inside semsearch database, as superuser:
\c semsearch
CREATE EXTENSION IF NOT EXISTS vector;
GRANT USAGE ON SCHEMA public TO semsearch_app;
GRANT CREATE ON SCHEMA public TO semsearch_app;
```

### 4.2 Table Schema (Created by `engine.init_vectorstore_table`)

The table is created explicitly by the program (not auto-created by `langchain-postgres`). The name is configurable via `settings.collection_name` (defaults to `semsearch_chunks`). The layout below is what `init_vectorstore_table` produces. `PGVectorStore`'s actual defaults are `page_content` (content column — **not** `content`), `embedding`, `langchain_id`, and `langchain_metadata` (see the correction note below the table for why an earlier draft of this spec got this wrong). We explicitly pass `content_column="content"` and `metadata_json_column="langchain_metadata"` at both table-creation and store-construction time, so this program's table uses `content` regardless of the library's own default.

```sql
-- Created by PGEngine.init_vectorstore_table(); shown here for reference.
-- Column names below are THIS PROGRAM'S chosen overrides, not PGVectorStore's
-- out-of-the-box defaults (default content column is "page_content" and the
-- default id column is UUID-typed — see the note below the table).

CREATE TABLE IF NOT EXISTS semsearch_chunks (
    langchain_id        TEXT PRIMARY KEY,          -- TEXT, not UUID — we use deterministic IDs
    embedding           vector(1536),              -- dim set via init_vectorstore_table(vector_size=N); see §4.3
    content             TEXT,                     -- chunk text (this program's override; see note below)
    langchain_metadata   JSONB,                    -- soft metadata: {page, row, doc_type, ingested_at, chunk_size, chunk_overlap}
    -- Custom columns added via metadata_columns= at init time:
    source              TEXT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    document_hash       CHAR(64),                  -- sha256(chunk_text); enables re-ingest to skip re-embedding
    -- Canonical identity of a chunk. Enforced at the DB level.
    UNIQUE (source, chunk_index)
);

-- HNSW cosine similarity index. For <1k docs, an exact-search IVFFLAT or no index
-- is also fine; HNSW is chosen so the schema is future-proof.
CREATE INDEX IF NOT EXISTS semsearch_chunks_hnsw_idx
    ON semsearch_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- JSONB containment + path-ops index for filter performance
-- (used by both similarity_search(filter=) and delete(filter=)).
CREATE INDEX IF NOT EXISTS semsearch_chunks_metadata_gin_idx
    ON semsearch_chunks USING gin (langchain_metadata jsonb_path_ops);

-- Composite index for re-ingest lookup:
--   WHERE source = :source ORDER BY chunk_index
-- Also serves `delete(filter={"source": ...})` and "show chunks 0..N of file X".
CREATE INDEX IF NOT EXISTS semsearch_chunks_source_chunk_idx
    ON semsearch_chunks (source, chunk_index);
```

> **Critical: `langchain_id` is TEXT, not UUID.** We use deterministic IDs of the form `"{source}::{chunk_index}"` so re-ingest can UPSERT by natural key. The default UUID PK conflicts with string IDs — `init_vectorstore_table(id_column=Column("langchain_id", "TEXT", primary_key=True))` overrides it. **⚠️ Verify before implementing**: published examples of `id_column=Column(...)` only show `Column(name, data_type)` (e.g. `Column("langchain_id", "VARCHAR")`); a `primary_key=` kwarg on `Column` has not been confirmed against the actual dataclass signature in `langchain_postgres.v2.engine`. Check the installed library's `Column` definition — if `primary_key` isn't a real parameter, drop it (the id column is implicitly the primary key by role) or this call will raise `TypeError`.

> **`init_vectorstore_table` is NOT idempotent.** It uses `CREATE TABLE`, not `CREATE TABLE IF NOT EXISTS`. The service's `init_schema()` method handles idempotency by checking `information_schema.tables` first (see §7.5). It also creates the `UNIQUE`, HNSW, and GIN constraints/indexes separately via explicit `CREATE [UNIQUE ] INDEX` statements (with `IF NOT EXISTS`) — `init_vectorstore_table` does not create them.

> **`UNIQUE (source, chunk_index)` constraint** is the canonical identity of a chunk. A buggy script, a manual `psql` edit, or a future migration cannot create duplicate rows — Postgres will reject the insert.

> **Note on column naming**: `PGVectorStore`/`init_vectorstore_table`'s actual defaults are `page_content` (content column), `embedding`, `langchain_id` (UUID), and `langchain_metadata` — **not** `content`. (The previous spec revision's claim that `content` was the default was itself wrong; it conflated `langchain-postgres`'s `PGVectorStore` with Google's separate `langchain-google-cloud-sql-pg` package, whose `PostgresVectorStore` does default to `content`.) This spec does **not** rely on the default — it explicitly passes `content_column="content"` and `metadata_json_column="langchain_metadata"` at both `init_vectorstore_table()` and store-construction time (§7.5), so the table's actual column is `content` regardless of what upstream calls it by default. Implementers must verify the exact defaults against the installed `langchain-postgres==0.0.17` source before relying on any default going forward — this library's schema helper API has changed across recent point releases. The previous spec revision incorrectly used `document`/`cmetadata` (those were the deprecated `PGVector` class's names, not `PGVectorStore`'s).

### 4.3 Embedding Dimension per Provider

The `embedding` column is typed `vector(N)` where N depends on the provider. The program creates the table with the right dimension via `engine.init_vectorstore_table(vector_size=N)` before the first insert. There is no auto-creation — the dimension must be passed explicitly.

| Provider type | Default model | Dim |
|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 |
| `openai` | `text-embedding-3-large` | 3072 |
| `huggingface` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `huggingface` | `BAAI/bge-small-en-v1.5` | 384 |
| `ollama` | `nomic-embed-text` | 768 |
| `openrouter` | `openai/text-embedding-3-small` | 1536 |
| `openrouter` | `text-embedding-3-small` (no provider prefix) | 1536 |
| `openai_compatible` | varies — depends on what the endpoint serves | n/a |

> **Constraint**: there is only one active provider at a time (§6.1). The DB column `vector(N)` matches whatever the active provider returns. If you switch providers with a different dim, run `semsearch init --recreate` (which drops the table and re-creates it with the new `vector_size`). There is no across-provider runtime fallback — when OpenRouter (or any provider) is down, edit `.env` and switch to a backup manually.

### 4.4 Connection String

The program reads `DATABASE_URL` from environment and passes it to `PGEngine.from_connection_string(url=...)`:

```text
postgresql+psycopg://semsearch_app:change_me_in_prod@localhost:5432/semsearch
```

> Note: `langchain-postgres` also supports `postgresql+asyncpg://...` for the async API. For v1 (sync only), use the `+psycopg` form. The `PGEngine` is a connection-pool holder shared across all `PGVectorStore` calls on the same engine.

---

## 5. Project Structure

```
pg-semantic-search/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env.example
├── spec.md                  # ← this file
├── src/
│   └── semsearch/
│       ├── __init__.py
│       ├── config.py        # pydantic-settings: Settings
│       ├── loaders.py       # pick_loader() + per-type loaders
│       ├── splitter.py      # chunking wrapper
│       ├── embeddings.py    # build_embedder() factory (includes OpenRouter routing translation)
│       ├── store.py         # PGEngine + PGVectorStore construction; init_schema()
│       ├── service.py       # SemanticSearchService (orchestration, ingest/search/delete)
│       ├── cli.py           # typer entrypoint
│       └── models.py        # pydantic: SearchResult, IngestResult, DeleteResult
├── tests/
│   ├── conftest.py          # testcontainers Postgres fixture + PGEngine fixture
│   ├── test_loaders.py
│   ├── test_embeddings.py
│   ├── test_service_ingest.py
│   ├── test_service_search.py
│   ├── test_service_delete.py
│   └── test_cli.py
└── scripts/
    └── init_db.sql          # §4.1 commands
```

> **`delete.py` removed**: deletion is now a single method on the service (`delete(filter=)`) delegating to `PGVectorStore.delete(filter=)`. No raw SQL helper module needed.

---

## 6. Configuration (`config.py`)

All runtime knobs are loaded from environment variables via `pydantic-settings`. There is **no** YAML config file by design — env vars compose better with Docker and CI.

### 6.1 `Settings` class

```python
import re
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


# Validates that collection_name is a safe SQL identifier — it becomes a table name
# in `init_vectorstore_table(table_name=...)` and is interpolated into raw SQL
# statements (the service-owned write path). We use a strict regex to prevent
# SQL injection and ensure compatibility with Postgres identifier rules.
_TABLE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class EmbeddingProviderConfig(BaseModel):
    """Single embedding provider configuration.

    One provider is active at a time. To switch providers (e.g. when OpenRouter
    is down), edit SEMSEARCH_EMBEDDING_PROVIDER__TYPE in .env — the program does
    NOT cascade across providers at runtime. Fallback only happens inside
    OpenRouter (provider.order + provider.allow_fallbacks), not across.

    Env var mapping uses double-underscore delimiter (see Settings.model_config):
        SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
        SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER='["openai","together"]'
    """
    type: Literal["openai", "huggingface", "ollama", "openai_compatible", "openrouter"]
    model: str

    # API config — meaning depends on `type`:
    #   openai             — OpenAI's hosted API (api_key required, base_url optional)
    #   openrouter         — OpenRouter (api_key required, base_url defaults to https://openrouter.ai/api/v1)
    #   openai_compatible  — any OpenAI-compatible endpoint (base_url required, api_key optional)
    #   huggingface        — local SentenceTransformers (api_key ignored)
    #   ollama             — local Ollama daemon (base_url optional, defaults to http://localhost:11434)
    api_key: SecretStr | None = None
    base_url: str | None = None
    device: str = "cpu"             # huggingface only

    # ---- OpenRouter routing — ignored by all other types ----
    # Canonical reference: https://openrouter.ai/docs/guides/routing/provider-selection
    # Verified against embeddings.md + provider-selection.md (2026-08).
    #
    # IMPORTANT: provider slugs are LOWERCASE identifiers, not display names.
    #   ✅ "openai", "together", "deepinfra", "azure"
    #   ❌ "OpenAI", "Together", "DeepInfra"
    # The service normalizes to lowercase before sending, but users should use
    # lowercase in config to avoid confusion.
    provider_order: list[str] | None = None          # e.g. ["openai", "together"]
    provider_allow_fallbacks: bool | None = None     # None = OpenRouter default (true). Must be None default
                                                      # so _build_openrouter_routing can omit when unset.
    provider_ignore: list[str] | None = None          # provider slugs to skip
    provider_only: list[str] | None = None            # provider slugs to allow (whitelist; takes precedence over ignore)
    provider_require_parameters: bool = False
    provider_data_collection: Literal["allow", "deny"] | None = None  # "deny" = avoid providers that may store data
    provider_max_price: dict | None = None            # e.g. {"prompt": 1} (USD per 1M tokens).
                                                      # For embeddings, only "prompt" is meaningful;
                                                      # "completion" and "image" are chat-only.


class Settings(BaseSettings):
    # env_nested_delimiter="__" lets pydantic-settings map
    #   SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
    # into  settings.embedding_provider.type = "openrouter".
    # Without this, nested BaseModel fields CANNOT be set from env vars
    # (a common pydantic-settings v2 gotcha).
    model_config = SettingsConfigDict(
        env_prefix="SEMSEARCH_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ---- Database ----
    database_url: str = Field(
        default="postgresql+psycopg://semsearch_app:change_me_in_prod@localhost:5432/semsearch"
    )
    # Becomes a SQL table name — strict regex prevents SQL injection via identifier.
    collection_name: str = Field(default="semsearch_chunks")

    @field_validator("collection_name")
    @classmethod
    def _validate_table_name(cls, v: str) -> str:
        if not _TABLE_NAME_RE.match(v):
            raise ValueError(
                f"collection_name {v!r} must match /^[a-z_][a-z0-9_]{{0,62}}$/ "
                f"(lowercase, alphanumeric + underscore, 1-63 chars, must start with a letter or underscore)"
            )
        return v

    # ---- Embedding provider (single, no cascade) ----
    embedding_provider: EmbeddingProviderConfig = Field(
        default=EmbeddingProviderConfig(
            type="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    # ---- Chunking ----
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Search defaults ----
    default_k: int = 5

    # ---- Lifecycle ----
    recreate_collection_on_init: bool = False   # safety: must be opt-in
```

> **pydantic-settings note**: the `embedding_provider` field uses the env prefix `SEMSEARCH_EMBEDDING_PROVIDER__` (double underscore!) thanks to `env_nested_delimiter="__"`. Sub-fields unpack automatically — e.g. `SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter` sets `embedding_provider.type`. List fields like `provider_order` accept JSON-encoded strings (e.g. `'["openai","together"]'`).
>
> **Common gotcha**: a single underscore (`SEMSEARCH_EMBEDDING_PROVIDER_TYPE`) will NOT be recognized — pydantic-settings will look for a flat field named `embedding_provider_type` on `Settings`, not the nested `EmbeddingProviderConfig.type`. Always use double underscore.

### 6.2 `.env.example`

```bash
# Database
SEMSEARCH_DATABASE_URL=postgresql+psycopg://semsearch_app:change_me_in_prod@localhost:5432/semsearch
# Must match /^[a-z_][a-z0-9_]{0,62}$/ — becomes a SQL table name.
SEMSEARCH_COLLECTION_NAME=semsearch_chunks

# ---- Pick ONE embedding provider config block below, comment out the others ----
# NOTE: nested fields use DOUBLE UNDERSCORE after the prefix.
#   SEMSEARCH_EMBEDDING_PROVIDER__TYPE   (not _TYPE)

# Option A: HuggingFace (default, local, free)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=huggingface
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=sentence-transformers/all-MiniLM-L6-v2
SEMSEARCH_EMBEDDING_PROVIDER__DEVICE=cpu

# Option B: OpenAI (hosted, paid)
# SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
# SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
# SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...

# Option C: Ollama (local daemon)
# SEMSEARCH_EMBEDDING_PROVIDER__TYPE=ollama
# SEMSEARCH_EMBEDDING_PROVIDER__MODEL=nomic-embed-text
# SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434

# Option D: OpenRouter (aggregator with internal provider fallback)
# Spec: https://openrouter.ai/docs/guides/routing/provider-selection
#       https://openrouter.ai/docs/api_reference/embeddings.md
# NOTE: provider slugs are LOWERCASE ("openai", not "OpenAI").
# SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
# SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
# SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-or-v1-...
# Try OpenAI first, fall back to Together inside OpenRouter (Level-1 fallback):
# SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER=["openai","together"]
# SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ALLOW_FALLBACKS=true
# SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_IGNORE=["deepinfra"]
# Whitelist specific providers (overrides ignore):
# SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ONLY=["openai","azure"]
# Avoid providers that may store data:
# SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_DATA_COLLECTION=deny
# Cap spend — for embeddings, only "prompt" is meaningful ("completion" is chat-only):
# SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_MAX_PRICE={"prompt":1}

# Option E: OpenAI-compatible endpoint (LM Studio, vLLM, Ollama's OpenAI shim, etc.)
# SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai_compatible
# SEMSEARCH_EMBEDDING_PROVIDER__MODEL=bge-small-en-v1.5
# SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:1234/v1
# SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=not-needed-but-required-by-sdk

# Chunking
SEMSEARCH_CHUNK_SIZE=1000
SEMSEARCH_CHUNK_OVERLAP=200

# Search defaults
SEMSEARCH_DEFAULT_K=5
```

> **No across-provider cascade**: when the active provider is down (OpenRouter unreachable, OpenAI 5xx, etc.), the embedding call fails and the ingest transaction rolls back. To switch providers, edit `.env` and re-run. This is intentional — keeps the design simple, and avoids silent dimension mismatches when fallback providers return different dims.
>
> **Common gotcha**: a single underscore (`SEMSEARCH_EMBEDDING_PROVIDER_TYPE`) will NOT be recognized — pydantic-settings will look for a flat field named `embedding_provider_type` on `Settings`, not the nested `EmbeddingProviderConfig.type`. Always use double underscore (`__`).

---

## 7. API Contract

All public functions live under `semsearch.service`. Type hints are normative — implementers must match them exactly.

### 7.1 Data Models (`models.py`)

```python
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


class SearchResult(BaseModel):
    id: str                               # langchain_id (TEXT — we use deterministic IDs like "source::chunk_index")
    content: str                          # chunk text
    score: float = Field(..., description="Cosine similarity in [-1, 1]; higher is better. "
                                          "Computed as 1.0 - cosine_distance (LangChain returns distance, "
                                          "not similarity — service converts.")
    source: str | None = None             # top-level column (str(path))
    chunk_index: int | None = None        # top-level column (0-based position within source)
    page: int | None = None               # langchain_metadata.page (PDF only)
    row: int | None = None                # langchain_metadata.row (CSV only)
    doc_type: str | None = None            # langchain_metadata.doc_type
    metadata: dict[str, Any] = Field(default_factory=dict)   # full langchain_metadata blob


class DeleteResult(BaseModel):
    deleted_count: int                   # computed via SELECT COUNT(*) in the same transaction before delete
    filter: dict[str, Any]                # the filter dict passed to PGVectorStore.delete()


class IngestResult(BaseModel):
    source: str
    chunks_added: int       # CASE C: newly embedded (no prior row)
    chunks_reused: int      # CASE A: content unchanged; existing embedding reused, no API call
    chunks_updated: int     # CASE B: content changed; re-embedded
    chunks_pruned: int = 0   # CASE D: stale tail chunks deleted (file shortened)
    ingested_at: datetime


class BatchAggregate(BaseModel):
    """Aggregate counts across multiple files in a batch. Separate from IngestResult
    because `source` doesn't apply at the batch level (the previous spec reused
    IngestResult.source with awkward placeholder values like "<batch:data/>").
    """
    chunks_added: int = 0
    chunks_reused: int = 0
    chunks_updated: int = 0
    chunks_pruned: int = 0


class BatchIngestResult(BaseModel):
    """Result of ingest_dir(). Aggregates per-file outcomes + optional prune info."""
    dir: str
    files_discovered: int                 # all files matching glob/exclude (including unsupported extensions)
    files_skipped_unsupported: int        # subset of discovered with unsupported extensions
    files_attempted: int                  # files_discovered - files_skipped_unsupported
    files_succeeded: int
    files_failed: int
    failed_files: list[dict] = Field(
        default_factory=list,
        description='List of {"path": str, "error": str} dicts for each file that failed.',
    )
    aggregate: BatchAggregate             # sum of chunks_added/reused/updated/pruned across all succeeded files
    elapsed_seconds: float
    pruned_sources: list[str] = Field(
        default_factory=list,
        description="Sources whose chunks were deleted (or would be, if prune_dry_run=True). "
                    "Empty unless prune=True was passed.",
    )
    pruned_chunks: int = Field(
        default=0,
        description="Total chunks deleted by prune. 0 if prune_dry_run=True (nothing actually deleted).",
    )
```

> **`SearchFilter` model removed**: the previous custom `SearchFilter` pydantic model is gone. Search filtering now uses `PGVectorStore`'s native filter dict syntax directly (see §8.4 for the operator reference). The service's `search()` accepts `filter: dict | None` and passes it straight through.
>
> **`SearchResult.score` is similarity, not distance**: `PGVectorStore.similarity_search_with_score()` returns `(Document, distance)` tuples where `distance` is cosine distance for the cosine strategy. The service converts: `score = 1.0 - distance` so that higher score = more similar (matches user intuition). Tests must assert on the converted score, not the raw distance.
>
> **`BatchAggregate` is separate from `IngestResult`**: avoids the awkward `<batch:data/>` placeholder for `source`. IngestResult is per-file (has source); BatchAggregate is per-batch (no source).

### 7.2 Embeddings Factory (`embeddings.py`)

```python
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
from semsearch.config import Settings, EmbeddingProviderConfig
from semsearch.errors import ProviderConfigError


def build_embedder(settings: Settings) -> Embeddings:
    """
    Return a LangChain Embeddings instance based on settings.embedding_provider.

    Single-provider design: no runtime cascade across providers. Fallback only
    happens INSIDE OpenRouter (via provider.order + provider.allow_fallbacks).
    If the active provider is unreachable, embed calls raise and the caller
    must catch / propagate.

    Raises:
        ProviderConfigError: required credentials missing for the selected type.
        ImportError: if the corresponding langchain-* package is not installed.
    """
    cfg = settings.embedding_provider

    if cfg.type == "openai":
        _require(cfg.api_key, "openai")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=cfg.api_key.get_secret_value(), model=cfg.model)

    if cfg.type == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=cfg.model,
            model_kwargs={"device": cfg.device},
        )

    if cfg.type == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=cfg.base_url or "http://localhost:11434",
            model=cfg.model,
        )

    if cfg.type in ("openai_compatible", "openrouter"):
        # Both use the OpenAI SDK against a custom base_url.
        from langchain_openai import OpenAIEmbeddings
        if cfg.type == "openrouter":
            base_url = cfg.base_url or "https://openrouter.ai/api/v1"
            # CRITICAL: OpenAIEmbeddings does NOT accept `extra_body` as a direct kwarg.
            # It accepts `model_kwargs: dict[str, Any]` which is unpacked into the
            # underlying `client.create(**model_kwargs)` call. The OpenAI SDK accepts
            # `extra_body` as a top-level kwarg on `.create()`, so we nest it:
            #     model_kwargs = {"extra_body": {"provider": {...}}}
            # Verified against langchain_openai source (Aug 2026).
            extra_body = _build_openrouter_routing(cfg)
            model_kwargs = {"extra_body": extra_body} if extra_body else {}
        else:
            base_url = cfg.base_url or "http://localhost:1234/v1"
            model_kwargs = {}
        return OpenAIEmbeddings(
            api_key=(cfg.api_key or SecretStr("not-needed")).get_secret_value(),
            base_url=base_url,
            model=cfg.model,
            model_kwargs=model_kwargs,    # forwarded into client.create(**model_kwargs)
        )

    raise ProviderConfigError(f"unknown embedding provider type: {cfg.type}")


def _build_openrouter_routing(cfg: EmbeddingProviderConfig) -> dict:
    """Build the OpenRouter `extra_body` dict (to be nested under `model_kwargs`).

    Returns the inner dict — caller wraps it as `model_kwargs={"extra_body": <this>}`.

    Canonical reference: https://openrouter.ai/docs/guides/routing/provider-selection
    Verified against embeddings.md + provider-selection.md (2026-08).

    Field mapping (spec → OpenRouter):
        provider_order              → provider.order                 (string[])
        provider_allow_fallbacks   → provider.allow_fallbacks       (boolean, OR default true)
        provider_ignore            → provider.ignore                (string[])
        provider_only              → provider.only                  (string[])
        provider_require_parameters→ provider.require_parameters    (boolean, default false)
        provider_data_collection   → provider.data_collection       ("allow" | "deny")
        provider_max_price         → provider.max_price             (object, e.g. {"prompt":1})

    Provider slugs: ALL provider identifiers are LOWERCASE (e.g. "openai", "together",
    "deepinfra", "azure") — not display names. The builder normalizes to lowercase
    before sending, but users should use lowercase in config.

    Returns {} when no routing fields are set, so non-OpenRouter types are
    unaffected (their model_kwargs is empty).
    """
    provider_body: dict = {}
    if cfg.provider_order is not None:
        provider_body["order"] = [s.lower() for s in cfg.provider_order]
    # Only emit allow_fallbacks if the user explicitly set it (None = OR default).
    # The previous spec default (True) caused this builder to ALWAYS emit {"allow_fallbacks": True},
    # contradicting the docstring claim of returning {} when nothing is set.
    if cfg.provider_allow_fallbacks is not None:
        provider_body["allow_fallbacks"] = cfg.provider_allow_fallbacks
    if cfg.provider_ignore is not None:
        provider_body["ignore"] = [s.lower() for s in cfg.provider_ignore]
    if cfg.provider_only is not None:
        provider_body["only"] = [s.lower() for s in cfg.provider_only]
    if cfg.provider_require_parameters:
        provider_body["require_parameters"] = True
    if cfg.provider_data_collection is not None:
        provider_body["data_collection"] = cfg.provider_data_collection
    if cfg.provider_max_price is not None:
        # Forward as-is. OpenRouter expects an object like {"prompt": 1}.
        # For embeddings, only "prompt" is meaningful; "completion" and "image"
        # are chat-only and have no effect on embedding requests.
        provider_body["max_price"] = cfg.provider_max_price
    return {"provider": provider_body} if provider_body else {}


def _require(secret: SecretStr | None, name: str) -> None:
    if not secret or not secret.get_secret_value():
        raise ProviderConfigError(f"provider {name!r} requires api_key")
```

**Behavior table**:

| `embedding_provider.type` | Returned class | Required env | Lazy import |
|---|---|---|---|
| `openai` | `langchain_openai.OpenAIEmbeddings(model=...)` | `EMBEDDING_PROVIDER__API_KEY` | `langchain_openai` |
| `huggingface` | `langchain_huggingface.HuggingFaceEmbeddings(model=..., model_kwargs={"device": ...})` | none | `langchain_huggingface` |
| `ollama` | `langchain_ollama.OllamaEmbeddings(base_url=..., model=...)` | none (Ollama must be running) | `langchain_ollama` |
| `openrouter` | `langchain_openai.OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1", model=..., model_kwargs={"extra_body": {"provider": {...}}})` | `EMBEDDING_PROVIDER__API_KEY` | `langchain_openai` |
| `openai_compatible` | `langchain_openai.OpenAIEmbeddings(base_url=<custom>, model=...)` | `EMBEDDING_PROVIDER__BASE_URL` (api_key optional) | `langchain_openai` |

### 7.3 Loaders (`loaders.py`)

```python
from pathlib import Path
from typing import Callable
from langchain_core.documents import Document


# Extension → doc_type mapping. Loaders don't set doc_type natively —
# the service injects it after load() based on this map.
_DOC_TYPE_BY_EXT = {
    ".txt": "text", ".md": "text",
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "json",
}


def pick_loader(path: Path) -> Callable[[], list[Document]]:
    """
    Dispatch on file extension:
      .txt, .md  -> TextLoader (single document)
      .pdf       -> PyMuPDFLoader (one Document per page)
      .csv       -> CSVLoader (one Document per row; first row = headers)
      .json      -> JSONLoader (one Document per element under jq_schema='.[].content')

    Args:
        path: file path. Must exist.

    Returns:
        A callable that returns list[Document]. Each Document.metadata WILL include:
          - source: str  (= str(path))
          - doc_type: str  ('text' | 'pdf' | 'csv' | 'json')  ← injected by service, NOT by loader
        Plus, where applicable (set by the loader itself):
          - page: int (PDF)
          - row: int (CSV)
          - id: Any (JSON)

        IMPORTANT: LangChain loaders do NOT set `doc_type` — they only set
        `source` and type-specific fields like `page`/`row`. The service
        wraps loader output to inject `doc_type` based on file extension.
        See `with_doc_type()` below.

    Raises:
        ValueError: unsupported extension
        FileNotFoundError: path does not exist
    """
    ...


def with_doc_type(docs: list[Document], path: Path) -> list[Document]:
    """Inject `source` (normalized to str(path)) and `doc_type` (from extension)
    into each Document's metadata. Returns the same list (mutates in place).

    Called by the service after loader.load(). This is necessary because
    LangChain's stock loaders don't know about our `doc_type` field.
    """
    doc_type = _DOC_TYPE_BY_EXT.get(path.suffix.lower())
    if doc_type is None:
        raise ValueError(f"unsupported extension: {path.suffix}")
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["doc_type"] = doc_type
    return docs
```

### 7.4 Splitter (`splitter.py`)

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Recursive character splitter with sensible defaults.
    Separators: ["\n\n", "\n", ". ", " ", ""]
    """
    ...


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents, preserving all metadata. Returns >= len(docs) chunks."""
    ...
```

### 7.5 Store (`store.py`)

```python
from langchain_postgres import PGEngine, PGVectorStore, Column  # Column is exported
                                                                 # at the top-level package,
                                                                 # NOT from a `.columns` submodule
from langchain_core.embeddings import Embeddings
from semsearch.config import Settings


# Custom top-level columns added to the chunks table via `metadata_columns=`.
# These are REAL Postgres columns (not JSONB keys) — they can be indexed and
# filtered efficiently by PGVectorStore. Defined here so both init_vectorstore_table
# and PGVectorStore.create_sync reference the same definitions.
CUSTOM_COLUMNS = [
    Column("source", "TEXT", nullable=False),
    Column("chunk_index", "INTEGER", nullable=False),
    Column("document_hash", "CHAR(64)"),
]

# List of column NAMES (strings) — what PGVectorStore.create_sync expects
# at store-construction time (NOT Column objects, which init_vectorstore_table expects).
CUSTOM_COLUMN_NAMES = [c.name for c in CUSTOM_COLUMNS]

# Column names passed explicitly for clarity. `LC_METADATA_COLUMN` and
# `LC_ID_COLUMN` match LangChain's own defaults; `LC_CONTENT_COLUMN` is this
# program's override (PGVectorStore's actual default content column is
# "page_content" — see §4.2).
LC_CONTENT_COLUMN = "content"          # override; NOT "document" (deprecated PGVector's name) or "page_content" (PGVectorStore's actual default)
LC_METADATA_COLUMN = "langchain_metadata"   # this IS the real default; NOT "cmetadata" (deprecated PGVector's name)
LC_ID_COLUMN = "langchain_id"         # this IS the real default name; we override its TYPE from UUID to TEXT


def build_engine(settings: Settings) -> PGEngine:
    """
    Construct a PGEngine bound to settings.database_url.
    The PGEngine holds a connection pool shared across all PGVectorStore
    instances derived from it. Construct ONCE per process; pass to build_store().
    """
    return PGEngine.from_connection_string(url=settings.database_url)


def build_store(settings: Settings, engine: PGEngine, embedder: Embeddings) -> PGVectorStore:
    """
    Construct a PGVectorStore bound to settings.collection_name (used as the
    Postgres table name). Uses PGVectorStore.create_sync (the deprecated
    `PGVector` constructor is NOT used).

    Critical kwargs:
      - id_column="langchain_id" with TEXT type — overridden at init time so
        we can pass deterministic string IDs (f"{source}::{chunk_index}").
        The default UUID PK would fail UUID coercion on string IDs.
      - content_column="content" — this program's chosen override (PGVectorStore's
        actual out-of-the-box default is "page_content"; see §4.2).
      - metadata_json_column="langchain_metadata" — this one IS LangChain's default name.
      - metadata_columns=CUSTOM_COLUMN_NAMES — list of strings (NOT Column objects).
        PGVectorStore.create_sync expects names; PGEngine.init_vectorstore_table
        expects Column objects (see init_schema below).
    """
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.collection_name,
        embedding_service=embedder,
        id_column=LC_ID_COLUMN,
        content_column=LC_CONTENT_COLUMN,
        metadata_json_column=LC_METADATA_COLUMN,
        metadata_columns=CUSTOM_COLUMN_NAMES,
    )


def init_schema(settings: Settings, engine: PGEngine, vector_size: int, *, recreate: bool = False) -> None:
    """
    Idempotently create the chunks table with the right vector dim.

    Implementation notes (these are subtleties of langchain-postgres 0.0.17):
      1. `init_vectorstore_table` uses `CREATE TABLE` (NOT `CREATE TABLE IF NOT EXISTS`),
         so calling it twice raises an error. We check `information_schema.tables`
         first and skip if the table already exists (unless recreate=True).
      2. `init_vectorstore_table` does NOT create the UNIQUE constraint or HNSW/GIN
         indexes. We add them via explicit `CREATE [UNIQUE ] INDEX IF NOT EXISTS`.
      3. We override `id_column=Column("langchain_id", "TEXT", primary_key=True)` so
         we can pass deterministic string IDs. The default UUID PK conflicts with
         our `{source}::{chunk_index}` IDs. NOTE: the `primary_key=` kwarg on `Column`
         is unverified against the installed library — see the caveat in §4.2 before
         relying on this exact call signature.

    Args:
        settings: app settings (reads collection_name).
        engine: PGEngine instance.
        vector_size: dimension of the active embedding provider (see §4.3).
        recreate: if True, DROP TABLE before re-creating. Requires explicit opt-in
                 via `semsearch init --recreate`.

    Raises:
        SchemaMismatchError: if the table exists but its vector_size doesn't match
                             the active provider and recreate=False.
    """
    ...
```

### 7.6 Service (`service.py`) — Primary Surface

```python
from pathlib import Path
from semsearch.config import Settings
from semsearch.models import SearchResult, IngestResult, BatchIngestResult, DeleteResult


class SemanticSearchService:
    """
    High-level facade over loader + splitter + embedder + PGVectorStore.

    Lifecycle:
      with SemanticSearchService.from_settings(settings) as svc:
          svc.ingest(...)
          svc.search(...)
          svc.delete(filter={...})
    """

    @classmethod
    def from_settings(cls, settings: Settings) -> "SemanticSearchService":
        """Build all internal components (engine, embedder, store) from settings.
        Does NOT call init_schema — call svc.init_schema() explicitly if needed."""
        ...

    def __enter__(self) -> "SemanticSearchService": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def close(self) -> None:
        """Release the underlying PGEngine connection pool.

        Note: `PGEngine.close()` is async — it disposes an underlying async
        SQLAlchemy pool. A synchronous `close()` must submit that coroutine to
        the engine's background loop and wait for completion, OR the service
        must be async-native. For v1 (sync API), we submit `pool.dispose()`
        to the engine's loop via `asyncio.run_coroutine_threadsafe(...).result()`
        and propagate any exception.
        """

    # ----- Schema -----

    def init_schema(self, *, recreate: bool = False) -> None:
        """
        Idempotently create the chunks table with the active provider's vector dim.
        Delegates to engine.init_vectorstore_table(table_name=, vector_size=).
        With recreate=True, drops the table first (requires confirmation in CLI).
        """
        ...

    # ----- Ingest -----

    def ingest(self, path: Path, *, reembed_unchanged: bool = False) -> IngestResult:
        """
        Load file at `path`, split into chunks, embed, store. Each chunk's row will have:
            source        = str(path)                          -- top-level column
            chunk_index   = 0-based position within source     -- top-level column
            document_hash = sha256(chunk_text)                  -- top-level column
            langchain_metadata = {page, row, doc_type, ingested_at, chunk_size, chunk_overlap}

        The pair (source, chunk_index) is the canonical identity of a chunk,
        enforced by a UNIQUE constraint at the DB level.

        Idempotency & content-hash cache (see §10):
          For each new chunk, the service looks up the existing row by
          (source, chunk_index) and compares document_hash:
            CASE A — row exists, hashes match:
                SKIP re-embedding. Reuse the existing vector. Bump ingested_at.
            CASE B — row exists, hashes differ:
                Content changed. Re-embed this chunk, UPSERT.
            CASE C — row does not exist:
                Embed and INSERT.
            CASE D — old row exists for the same source but chunk_index >= len(new_chunks):
                File was shortened. DELETE the stale tail rows so they don't
                show up in search results. This case is detected AFTER the
                A/B/C classification — see §10.2.
          This means re-ingesting an unchanged file makes ZERO embedding API
          calls and runs in O(chunks) DB lookups instead of O(chunks) embeddings.

        Write path is service-owned (NOT delegated to PGVectorStore.add_documents):
          1. SELECT existing (chunk_index, document_hash, langchain_id) for the source
          2. Classify each new chunk → A/B/C
          3. Embed CASE B + C only (single batched embed_documents call)
          4. Run all writes inside ONE SQLAlchemy transaction:
             - INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE for B/C
             - UPDATE ingested_at for A
             - DELETE WHERE source = :source AND chunk_index >= :new_len  (CASE D)
          5. COMMIT (or ROLLBACK on any error)
          PGVectorStore.add_documents() is NOT used for writes — it does
          row-by-row commits and would double-embed precomputed vectors.

        Args:
            path: file to ingest.
            reembed_unchanged: if True, force CASE A chunks to be re-embedded
                (e.g. after switching embedding_provider). Default False.
                When True, CASE A is downgraded to CASE B behavior.

        Returns:
            IngestResult with chunks_added (CASE C), chunks_reused (CASE A),
            chunks_updated (CASE B), chunks_pruned (CASE D), and ingested_at.

        Raises:
            FileIngestError: loader/embedding/insert failed; partial state rolled back.
        """
        ...

    # ----- Batch ingest (directories) -----

    # Whitelist of file extensions handled by pick_loader (§7.3).
    # Files with extensions outside this set are silently skipped by ingest_dir
    # with a warning logged.
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".csv", ".json")

    def ingest_dir(
        self,
        dir_path: Path,
        *,
        glob: str = "**/*",
        exclude: list[str] | None = None,
        reembed_unchanged: bool = False,
        continue_on_error: bool = True,
        follow_symlinks: bool = False,
        prune: bool = False,
        prune_dry_run: bool = False,
    ) -> BatchIngestResult:
        """
        Walk `dir_path` and ingest every supported file in a single batch.

        File discovery:
          - Walks `dir_path` using `Path.glob(glob)` (default: all files recursively).
          - Filters by SUPPORTED_EXTENSIONS — unsupported files are skipped with a warning.
          - Skips hidden files (those whose name starts with ".").
          - Skips symlinks by default (set follow_symlinks=True to follow). Hidden-file
            handling is independent of symlink policy — `follow_symlinks` only affects
            whether symlinks are followed, not whether hidden files are skipped.
          - Skips files matching any pattern in `exclude` (fnmatch-style).
          - Iterates in sorted (alphabetical) order for deterministic output.

        Per-file semantics:
          - Calls self.ingest(file_path, reembed_unchanged=reembed_unchanged) per file.
          - Each file's ingest runs in its own transaction. A failure in one file
            does NOT roll back successful ingests of earlier files.
          - If continue_on_error=False, the first failure raises FileIngestError
            and subsequent files are NOT attempted.
          - If continue_on_error=True (default), failures are logged and collected
            in BatchIngestResult.failed_files.

        Idempotency:
          - Re-running ingest_dir on an unchanged dir makes ZERO embedding API
            calls (CASE A reuse kicks in per-file). See §10.
          - However, files DELETED from disk are NOT automatically removed from
            the store. Their chunks become orphans. Use `prune=True` to clean up.
          - Files RENAMED or MOVED are treated as "delete old + add new" by
            `--prune` — content-hash reuse across source renames is NOT supported
            in v1 (the re-embedded new copy pays the full embedding cost).

        Prune:
          - If prune=True, after the per-file loop completes, the service queries
            the DB for every distinct `source` whose value starts with
            `str(dir_path) + "/"`. Any source in that set that is NOT in the set
            of files just ingested has its chunks deleted via store.delete(filter={"source": s}).
          - prune_dry_run=True (requires prune=True) skips the actual delete and
            only populates BatchIngestResult.pruned_sources. Useful for inspecting
            what would be removed before committing.
          - Prune matches sources by literal prefix — see §9 "Path consistency".

        Args:
            dir_path: directory to walk. Must exist and be a directory.
            glob: glob pattern (default "**/*" = all files recursively).
            exclude: list of fnmatch patterns to skip (e.g. ["*/draft/*", "*/.tmp/*"]).
            reembed_unchanged: forwarded to per-file ingest(). Default False.
            continue_on_error: if False, abort the batch on first file failure.
            follow_symlinks: if True, follow symlinks during file discovery. Default False.
            prune: if True, delete chunks for sources under dir_path that no longer
                exist on disk after the ingest loop completes.
            prune_dry_run: if True (and prune=True), do NOT actually delete — only
                list what would be deleted in BatchIngestResult.pruned_sources.

        Returns:
            BatchIngestResult with per-file outcome counts, aggregate IngestResult,
            and (if prune=True) the list of pruned sources + total chunks pruned.

        Raises:
            ValueError: dir_path does not exist or is not a directory.
            FileIngestError: only if continue_on_error=False and a file fails.
        """
        ...

    # ----- Search -----

    def search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """
        Cosine similarity search over the chunks table, optionally scoped.

        Delegates to PGVectorStore.similarity_search_with_score(query, k=k, filter=filter)
        which returns List[Tuple[Document, float]] where the float is a DISTANCE
        (cosine distance for the cosine strategy), NOT similarity. The service
        converts: score = 1.0 - distance so higher = more similar.

        Args:
            query: free-text query.
            k: top-k. Defaults to settings.default_k. Must be 1 <= k <= 50.
            filter: optional PGVectorStore filter dict. If None or empty,
                search the entire table. Passed straight through to
                PGVectorStore.similarity_search_with_score(filter=...).

        Filter dict syntax (native PGVectorStore operators):
            {"source": "docs/handbook.pdf"}            # exact match
            {"source": {"$ilike": "docs/%"}}          # prefix match (grep -r style)
            {"doc_type": "pdf"}                      # langchain_metadata field
            {"year": {"$gte": 2024}}                 # numeric comparison
            {"$and": [{...}, {...}]}                  # logical AND
            {"$or":  [{...}, {...}]}                  # logical OR
            Operators: $eq, $ne, $lt, $lte, $gt, $gte, $in, $nin,
                      $between, $like, $ilike, $and, $or, $exists, $not.
            See §8.4 for examples.

        Returns:
            List[SearchResult] sorted by score DESC (higher = more similar).
            Length <= k. May be empty.

        Raises:
            ValueError: k out of range.
            SearchError: embedding query failed or DB unreachable.
        """
        ...

    # ----- Delete -----

    def delete(self, filter: dict) -> DeleteResult:
        """
        Delete every chunk matching the filter. Delegates to
        PGVectorStore.delete(filter=...) which returns Optional[bool]
        (NOT a rowcount). To return a meaningful `deleted_count`, we compute
        it ourselves via `SELECT COUNT(*) ... WHERE <filter>` in the same
        transaction before calling `delete()`.

        Filter syntax is the same rich dict used by search().

        Examples:
            svc.delete({"source": "docs/handbook.pdf"})
            svc.delete({"doc_type": "invoice", "year": 2024})
            svc.delete({"source": {"$ilike": "docs/old/%"}})   # delete everything under docs/old/
            svc.delete({})                                      # delete EVERYTHING (see note)

        Special case: an empty filter dict deletes the entire table. The CLI
        wraps this with a confirmation flag (`delete --all --yes`).
        NOTE: PGVectorStore.delete(filter=None) or filter={} may return False
        without deleting anything (the underlying implementation requires
        non-empty filter criteria). For `--all`, the service bypasses
        PGVectorStore and issues `DELETE FROM <table>` directly via SQLAlchemy.

        Returns:
            DeleteResult with deleted_count (computed via SELECT COUNT in
            the same transaction) and the filter that was applied.

        Raises:
            DeleteError: PGVectorStore.delete() raised (DB unreachable, etc.).
        """
        ...

    # ----- Stats -----

    def stats(self) -> dict:
        """
        Return:
          {
            "table": str,                       # settings.collection_name
            "embedding_provider": str,
            "embedding_dim": int,
            "chunk_count": int,
            "source_count": int,                # distinct sources
            "sources_by_count": list[tuple[str, int]]   # top 20
          }
        """
        ...
```

### 7.7 Exception Hierarchy

```python
class SemSearchError(Exception):
    """Base class for all semsearch errors."""

class FileIngestError(SemSearchError): ...
class SearchError(SemSearchError): ...
class DeleteError(SemSearchError): ...
class SchemaMismatchError(SemSearchError): ...
class ProviderConfigError(SemSearchError): ...
```

> `DeleteError` is retained even though deletion now delegates to `PGVectorStore.delete()` — any underlying DB error (connection lost, timeout) gets wrapped in `DeleteError` so callers can catch semsearch-specific exceptions without leaking langchain-postgres internals.

---

## 8. CLI (`cli.py`)

Implemented with `typer`. Entry point: `semsearch`.

```
Usage: semsearch [OPTIONS] COMMAND [ARGS]...

Commands:
  init        Create / migrate the chunks table
  ingest      Ingest a single file
  ingest-dir  Recursively ingest all supported files in a directory
  search      Run a similarity search (with optional PGVectorStore filter)
  delete      Delete chunks matching a filter dict (or --all)
  stats       Show table stats
  reingest    Delete + ingest in one step (single file)
```

### 8.1 `init`

```bash
semsearch init [--recreate] [--yes]
```

- Without `--recreate`: idempotent, calls `engine.init_vectorstore_table(table_name=, vector_size=)` which creates the table + indexes if missing.
- With `--recreate`: **drops** the chunks table first, then re-creates with the active provider's vector dim. Asks for confirmation unless `--yes` is passed.

> **Provider swap**: when switching `EMBEDDING_PROVIDER_TYPE` to a provider with a different embedding dim, you MUST pass `--recreate` (the table's `vector(N)` column type can't be altered in-place).

### 8.2 `ingest`

```bash
# Default: use provider from .env
semsearch ingest ./data/handbook.pdf

# Force re-embed of unchanged chunks (e.g. after switching EMBEDDING_PROVIDER_TYPE
# or changing chunk_size)
semsearch ingest --force ./data/handbook.pdf

# Override provider at ingest time without editing .env
semsearch ingest ./data/handbook.pdf \
  --provider openrouter \
  --provider-model openai/text-embedding-3-small \
  --provider-order '["openai","together"]' \
  --provider-allow-fallbacks true \
  --provider-ignore '["deepinfra"]'
```

Flags:
- `--force` — pass `reembed_unchanged=True` to the service. See §10.4.
- `--provider <type>` — override `EMBEDDING_PROVIDER_TYPE` for this run only.
- `--provider-model <name>` — override `EMBEDDING_PROVIDER_MODEL`.
- `--provider-order '<json>'` — OpenRouter routing `provider.order`. Ignored unless `--provider openrouter`.
- `--provider-allow-fallbacks <bool>` — OpenRouter routing `provider.allow_fallbacks`. Default true.
- `--provider-ignore '<json>'` — OpenRouter routing `provider.ignore`.
- `--provider-base-url <url>` — override `EMBEDDING_PROVIDER_BASE_URL` (required for `openai_compatible`).
- `--provider-api-key <key>` — override `EMBEDDING_PROVIDER_API_KEY`.

> **No across-provider cascade**: if `--provider` fails (network, 5xx, auth), the ingest aborts with `FileIngestError`. The CLI does not automatically retry with a backup provider — edit `.env` or pass a different `--provider` and re-run.

Expected output (stdout, JSON):

```json
{
  "source": "data/handbook.pdf",
  "chunks_added": 0,
  "chunks_reused": 47,
  "chunks_updated": 0,
  "ingested_at": "2026-08-18T03:24:11Z"
}
```

> The fields `chunks_added` / `chunks_reused` / `chunks_updated` reflect the three cases defined in §10.1. On first ingest of a fresh file, all chunks will be `chunks_added`. On a no-op re-ingest, all chunks will be `chunks_reused` and **zero embedding API calls** are made.

### 8.3 `ingest-dir`

```bash
# Default: walk the directory recursively, ingest all supported files
semsearch ingest-dir ./data/

# Only PDFs anywhere under ./data/
semsearch ingest-dir ./data/ --glob "**/*.pdf"

# Skip drafts and tmp dirs
semsearch ingest-dir ./data/ \
  --exclude "*/draft/*" \
  --exclude "*/.tmp/*"

# Sync: after ingest, delete chunks for files that no longer exist on disk
semsearch ingest-dir ./data/ --prune

# Dry-run: see what --prune would remove, don't actually delete
semsearch ingest-dir ./data/ --prune --dry-run

# Abort on first file failure (default: continue + collect errors)
semsearch ingest-dir ./data/ --no-continue-on-error

# Re-embed everything (after provider swap)
semsearch ingest-dir ./data/ --force

# Follow symlinks during discovery (default: skip)
semsearch ingest-dir ./data/ --follow-symlinks
```

Flags:
- `--glob <pattern>` — file discovery glob (default `**/*`). Use `"**/*.pdf"` to filter by extension.
- `--exclude <pattern>` — fnmatch pattern to skip. Repeatable: `--exclude "*/draft/*" --exclude "*/.tmp/*"`.
- `--prune` — after ingest, delete chunks for sources under `dir_path/` that no longer exist on disk. See behavior matrix below.
- `--dry-run` — with `--prune`, list what would be deleted without actually deleting. Without `--prune`, this flag is ignored.
- `--no-continue-on-error` — abort the batch on first file failure. Default is to continue and collect errors in `failed_files`.
- `--follow-symlinks` — follow symlinks during file discovery. Default: skip symlinks.
- `--force` — pass `reembed_unchanged=True` to every per-file ingest. See §10.4.

#### Behavior matrix

| Scenario | Plain `ingest-dir` | `--prune` | `--prune --dry-run` |
|---|---|---|---|
| Unchanged dir re-run (disk + DB in sync) | 0 embed calls, 0 writes | 0 embed calls, 0 deletes | 0 embed calls, 0 deletes |
| Unchanged dir re-run (DB has orphans from prior run without `--prune`) | 0 embed calls, 0 writes (orphans remain) | 0 embed calls, deletes the orphans | 0 embed calls, lists orphans in `pruned_sources` (no delete) |
| New file added | CASE C embeds new file | same + 0 deletes | same + 0 deletes |
| File content edited | CASE B re-embeds changed chunks | same + 0 deletes | same + 0 deletes |
| File deleted from disk | Orphaned chunks remain | Orphaned chunks deleted | Orphaned chunks listed in `pruned_sources` |
| File renamed | New chunks added (CASE C); old orphaned | New chunks added (CASE C); old deleted | New chunks added (CASE C); old listed |
| File moved to subdir | Same as rename | Same as rename | Same as rename |

> **Idempotency**: re-running `ingest-dir ./data/` on an unchanged directory makes **zero** embedding API calls — every chunk hits CASE A reuse. See §10.
>
> **Content-hash reuse across renames is NOT supported in v1**: renaming `data/a.pdf` to `data/b.pdf` causes `b.pdf` to be re-embedded from scratch (CASE C) even though the content is identical. The old chunks under `a.pdf` are pruned by `--prune`. This is correct behavior — the `source` metadata reflects the new filename — but it isn't free.
>
> **Prune prefix matching**: `--prune` only deletes chunks whose `source` starts with `str(dir_path) + "/"`. Chunks ingested from other directories are never touched. See §9 "Path consistency".

Expected output (stdout, JSON):

```json
{
  "dir": "data/",
  "files_attempted": 47,
  "files_succeeded": 45,
  "files_failed": 2,
  "failed_files": [
    {"path": "data/corrupt.pdf", "error": "PyMuPDFLoader: EOF marker not found"},
    {"path": "data/empty.csv", "error": "CSVLoader: no rows"}
  ],
  "aggregate": {
    "source": "<batch:data/>",
    "chunks_added": 312,
    "chunks_reused": 0,
    "chunks_updated": 0,
    "ingested_at": "2026-08-18T05:42:11Z"
  },
  "elapsed_seconds": 18.4,
  "pruned_sources": [],
  "pruned_chunks": 0
}
```

With `--prune` against a dir where 3 files were deleted from disk since the last sync:

```json
{
  "dir": "data/",
  "files_attempted": 44,
  "files_succeeded": 44,
  "files_failed": 0,
  "failed_files": [],
  "aggregate": {
    "source": "<batch:data/>",
    "chunks_added": 0,
    "chunks_reused": 412,
    "chunks_updated": 0,
    "ingested_at": "2026-08-18T05:43:02Z"
  },
  "elapsed_seconds": 0.8,
  "pruned_sources": [
    "data/old_report.pdf",
    "data/archive/2023_q1.txt",
    "data/notes/draft.md"
  ],
  "pruned_chunks": 47
}
```

### 8.4 `search`

```bash
# Search the whole table
semsearch search "how do I reset my password" --k 5

# Like `grep -r 'q' docs/` — only chunks whose source starts with `docs/`
semsearch search "how do I reset my password" --filter '{"source": {"$ilike": "docs/%"}}'

# Scope to a single file (exact match)
semsearch search "how do I reset my password" --filter '{"source": "docs/handbook.pdf"}'

# Only PDFs anywhere in the table
semsearch search "how do I reset my password" --filter '{"doc_type": "pdf"}'

# Only PDFs under docs/ (combined with $and)
semsearch search "how do I reset my password" \
  --filter '{"$and": [{"source": {"$ilike": "docs/%"}}, {"doc_type": "pdf"}]}'

# Numeric comparison
semsearch search "invoice amount" --filter '{"year": {"$gte": 2024}}'
```

Flags:
- `--filter '<json>'` — PGVectorStore native filter dict. Passed straight to `similarity_search_with_score(filter=...)`.
- `--k <int>` — top-k, default 5.

#### Filter operator reference (native PGVectorStore)

| Operator | Meaning | Example |
|---|---|---|
| (direct value) | Equality (==) | `{"source": "docs/x.pdf"}` |
| `$eq` | Equality (==) | `{"source": {"$eq": "docs/x.pdf"}}` |
| `$ne` | Inequality (!=) | `{"doc_type": {"$ne": "pdf"}}` |
| `$lt`, `$lte` | Less than / <= | `{"year": {"$lt": 2024}}` |
| `$gt`, `$gte` | Greater than / >= | `{"year": {"$gte": 2024}}` |
| `$in` | In list | `{"source": {"$in": ["a.pdf", "b.pdf"]}}` |
| `$nin` | Not in list | `{"doc_type": {"$nin": ["csv", "json"]}}` |
| `$between` | Between two values | `{"page": {"$between": [1, 10]}}` |
| `$like` | SQL LIKE | `{"source": {"$like": "docs/%"}}` |
| `$ilike` | Case-insensitive LIKE | `{"source": {"$ilike": "DOCS/%"}}` |
| `$and` | Logical AND | `{"$and": [{...}, {...}]}` |
| `$or` | Logical OR | `{"$or": [{...}, {...}]}` |
| `$exists` | Field is present | `{"source": {"$exists": true}}` |
| `$not` | Logical NOT | `{"$not": {"doc_type": "pdf"}}` |

> **⚠️ Verify before implementing**: this operator table is sourced from the deprecated `langchain_postgres.vectorstores.PGVector`'s `SUPPORTED_OPERATORS` constant. This spec mandates the newer `PGVectorStore` (v2) class throughout, and its filter-translation code path has **not** been confirmed to support the identical operator set (`$exists` and `$not` in particular). Before writing `test_service_search.py` or the CLI `--filter` docs, check the filter translator in the installed `langchain-postgres==0.0.17` source (`langchain_postgres/v2/...`) against this table and correct any operator that isn't actually supported by `PGVectorStore`.
>
> **`--in` flag removed**: the old `--in` and `--type` flags are gone. They were a subset of what `--filter` can express. Use `--filter '{"source": {"$ilike": "docs/%"}}'` for grep-style prefix matching and `--filter '{"doc_type": "pdf"}'` for type filtering. The CLI does not normalize paths — use the same path form at search time as at ingest time (see §9 "Path consistency").

Expected output:

```json
{
  "query": "how do I reset my password",
  "k": 5,
  "filter": {"source": {"$ilike": "docs/%"}},
  "results": [
    {
      "id": "docs/handbook.pdf::24",
      "content": "To reset your password, go to Settings > Security and click 'Reset password'...",
      "score": 0.8731,
      "source": "docs/handbook.pdf",
      "chunk_index": 24,
      "page": 12,
      "doc_type": "pdf"
    },
    ...
  ]
}
```

### 8.5 `delete`

```bash
# Delete by exact source (one file)
semsearch delete --filter '{"source": "data/handbook.pdf"}'

# Delete everything under a directory (grep -r style)
semsearch delete --filter '{"source": {"$ilike": "docs/old/%"}}'

# Delete by metadata filter (conjunction)
semsearch delete --filter '{"doc_type": "invoice", "year": 2024}'

# Delete everything in the table (requires --yes)
semsearch delete --all --yes
```

Flags:
- `--filter '<json>'` — PGVectorStore native filter dict. Same syntax as `search --filter`.
- `--all` — shortcut for `--filter '{}'` (deletes every row). Requires `--yes` to confirm.

Expected output:

```json
{
  "deleted_count": 47,
  "filter": {"source": "data/handbook.pdf"}
}
```

### 8.6 `stats`

```bash
semsearch stats
```

Expected output:

```json
{
  "table": "semsearch_chunks",
  "embedding_provider": "huggingface",
  "embedding_dim": 384,
  "chunk_count": 312,
  "source_count": 8,
  "sources_by_count": [
    ["data/handbook.pdf", 47],
    ["data/faq.txt", 31],
    ...
  ]
}
```

### 8.7 `reingest`

```bash
semsearch reingest ./data/handbook.pdf
```

Equivalent to `delete --filter '{"source": "./data/handbook.pdf"}'` followed by `ingest`, executed in a single transaction. Useful when you know the file changed substantially and want to start fresh rather than rely on the chunk-by-chunk content-hash cache.

> **⚠️ Gap — needs a design decision before implementation**: §7.6 does not define a `SemanticSearchService.reingest()` method — only `init_schema`, `ingest`, `ingest_dir`, `search`, `delete`, and `stats` exist. Add one. Also, "executed in a single transaction" is non-trivial as described: `delete()` goes through `PGVectorStore.delete(filter=...)` (§3.4) while `ingest()` uses a separate service-owned `engine.begin()` block (§10.2) — these are two different transaction-management paths that don't automatically share a connection/transaction. To make `reingest` genuinely atomic, either (a) have `reingest()` open one `engine.begin()` block and pass the same connection through both the delete-count/delete SQL and the ingest write SQL, bypassing `PGVectorStore.delete()`'s own transaction handling, or (b) relax the "single transaction" claim to "delete commits, then ingest runs in its own transaction, with the CLI reporting a partial-failure state if ingest fails after delete succeeded." Pick one and document it explicitly — as written, an implementer has no way to know which behavior to build.

---

## 9. Chunking Strategy

- **Strategy**: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`.
- **Defaults**: `chunk_size=1000`, `chunk_overlap=200`.
- **Per source type**:
  | Source | Special handling |
  |---|---|
  | Text / Markdown | None — feed raw text to splitter. |
  | PDF | PyMuPDFLoader returns one Document per page. Splitter further splits each page. `metadata.page` is preserved. |
  | CSV | CSVLoader returns one Document per row with `metadata.row` and column values as content. Splitter is skipped if row content length < chunk_size. |
  | JSON | JSONLoader with `jq_schema='.[].content'` produces one Document per element. Each element is its own chunk if `len(content) <= chunk_size`. |
- **Metadata propagation**: every chunk inherits the source Document's metadata. The service then promotes `source`, `chunk_index`, and `document_hash` to top-level table columns (§4.2) via `metadata_columns=` on `init_vectorstore_table`. The remaining soft metadata (`page`, `row`, `doc_type`, `ingested_at`, `chunk_size`, `chunk_overlap`) is stored inside `langchain_metadata` JSONB, which is what PGVectorStore's filter syntax operates on. (Note: `cmetadata` was the deprecated `PGVector` class's column name — do not use it here.)
- **Path consistency (important)**: `source` is stored verbatim as `str(path)` — the program does NOT normalize relative/absolute paths or resolve symlinks. This is by design, but it has three consequences every user must understand:
  1. **Re-ingest cache hit requires identical path strings.** If you ingest `docs/handbook.pdf` and later re-ingest with `./docs/handbook.pdf` or `/home/alice/docs/handbook.pdf`, the re-ingest SELECT in §10.2 won't find the prior rows. Every chunk becomes CASE C (newly embedded) instead of CASE A (reused). No data corruption — just wasted embedding API calls. The same applies to `ingest_dir` — each file in the directory is ingested by its `str(path)`, so the same dir walked from a different cwd produces different `source` strings.
  2. **`--filter` requires the same path form.** A search filter like `{"source": {"$ilike": "docs/%"}}` performs a literal pattern match on the stored `source` column. If you ingested with absolute paths, you must use `{"$ilike": "/home/alice/docs/%"}` to match.
  3. **`ingest-dir --prune` matches by literal prefix.** Prune queries the DB for `source LIKE 'str(dir_path)/%'` and deletes any whose file no longer exists on disk. If you previously ingested with absolute paths but run `ingest-dir data/ --prune` from a relative cwd, the prefix won't match — nothing gets pruned, even if files are missing. Always prune from the same cwd (or with the same absolute dir) used at ingest time.

  **Recommendation: pick one canonical path form per project and stick to it.** The simplest rule is "always invoke `semsearch ingest` and `semsearch ingest-dir` from the project root directory" so all `source` values are stored as `docs/foo.pdf`, `data/bar.txt`, etc. The CLI will print a warning at ingest time if it detects a path form inconsistent with previously-ingested rows in the same table (best-effort, non-blocking).

---

## 10. Idempotency & Re-ingest

Re-ingesting the same file must:

1. **Not produce duplicates** — old chunks for `(source, chunk_index)` are replaced.
2. **Not re-embed unchanged chunks** — content-hash comparison short-circuits the embedding call.
3. **Be atomic** — all writes for a single `ingest()` call live in ONE SQLAlchemy transaction (NOT `PGVectorStore.add_documents`, which does row-by-row commits).
4. **Clean up stale tail chunks** — if the new file produces fewer chunks than the old, rows with `chunk_index >= len(new_chunks)` are deleted (CASE D).

### 10.1 The four cases

For each chunk produced by the splitter, the service classifies it into exactly one of three per-chunk cases (A/B/C). After processing all new chunks, an additional CASE D — applied once per `ingest()` call, not per chunk — handles file-shortening (deleting stale tail rows). Four cases total; A/B/C are chunk-level, D is file-level.

| Case | Existing row? | `document_hash` matches? | Action | Embedding API call? |
|------|---------------|--------------------------|--------|---------------------|
| **A** | yes | yes | Reuse existing `embedding` value; only UPDATE `ingested_at` | ❌ No |
| **B** | yes | no (content changed) | Re-embed this chunk, UPSERT | ✅ Yes (this chunk only) |
| **C** | no | n/a | Embed and INSERT | ✅ Yes |
| **D** | yes (but `chunk_index >= len(new_chunks)`) | n/a | DELETE the stale row (file was shortened) | ❌ No |

### 10.2 Flow

```
new_chunks = load(path) → with_doc_type(path) → split()    # in-memory only

# One SELECT — fetch all existing rows for this source at once
existing = SELECT langchain_id, chunk_index, document_hash
           FROM semsearch_chunks
           WHERE source = :source
           ORDER BY chunk_index
# Returns at most len(new_chunks) rows (or more if file was previously longer).

cases = []
for i, chunk in enumerate(new_chunks):
    h = sha256(chunk.text)
    row = existing.get(i)              # lookup by chunk_index
    if row is None:
        cases.append(('C', i, chunk, h))
    elif row.document_hash == h and not reembed_unchanged:
        cases.append(('A', i, chunk, h, row.langchain_id))
    else:
        cases.append(('B', i, chunk, h, row.langchain_id))

# Only embed CASE B + CASE C chunks
to_embed = [c.chunk.text for c in cases if c[0] in ('B', 'C')]
vectors  = embedder.embed_documents(to_embed)     # batched, single API call

# Service-owned SQLAlchemy transaction (NOT PGVectorStore.add_documents):
# PGVectorStore.add_documents does row-by-row commits + would re-embed B/C.
with engine.begin() as conn:                      # BEGIN
    # CASE A: cheap UPDATE (ingested_at only)
    if case_a_ids:
        conn.execute(text("""
            UPDATE semsearch_chunks
            SET langchain_metadata = jsonb_set(
                    langchain_metadata, '{ingested_at}', :now_jsonb)
            WHERE langchain_id = ANY(:ids)
        """), {"ids": case_a_ids, "now_jsonb": now_json})

    # CASE B + C: INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE
    # Same SQL handles both — CASE C just doesn't conflict (no prior row).
    for c, vec in zip(case_bc, vectors_for_bc):
        conn.execute(text("""
            INSERT INTO semsearch_chunks
                (langchain_id, embedding, content, langchain_metadata,
                 source, chunk_index, document_hash)
            VALUES (:id, :vec, :text, :meta, :source, :chunk_index, :hash)
            ON CONFLICT (source, chunk_index) DO UPDATE
            SET embedding       = EXCLUDED.embedding,
                content          = EXCLUDED.content,
                document_hash   = EXCLUDED.document_hash,
                langchain_metadata = EXCLUDED.langchain_metadata
        """), {
            "id": f"{source}::{c.chunk_index}",
            "vec": vec,
            "text": c.chunk.text,
            "meta": json.dumps({...c.metadata, "ingested_at": now_iso}),
            "source": source,
            "chunk_index": c.chunk_index,
            "hash": c.hash,
        })

    # CASE D: delete stale tail chunks (file shortened)
    if len(new_chunks) < previous_max_chunk_index + 1:
        conn.execute(text("""
            DELETE FROM semsearch_chunks
            WHERE source = :source AND chunk_index >= :new_len
        """), {"source": source, "new_len": len(new_chunks)})
# COMMIT (automatic on context exit; ROLLBACK on exception)
```

> **Why service-owned SQL instead of `PGVectorStore.add_documents`?**
> 1. **Atomicity**: `PGVectorStore.add_documents()` does row-by-row inserts with per-row commits — a mid-loop failure leaves partial state. Our `engine.begin()` block wraps everything in ONE transaction that commits atomically or rolls back completely.
> 2. **No double embedding**: we precompute embeddings for CASE B/C chunks (to skip CASE A); `add_documents` would re-embed them.
> 3. **`ON CONFLICT` for UPSERT**: `INSERT ... ON CONFLICT (source, chunk_index) DO UPDATE` is the canonical Postgres UPSERT. `add_documents` only handles conflicts via the `langchain_id` PK, which is not the natural key we want.
> 4. **CASE D cleanup**: deleting stale tail chunks requires a separate `DELETE WHERE chunk_index >= :new_len` — `add_documents` has no equivalent.
>
> **Why deterministic string IDs `"{source}::{chunk_index}"`?** They make the row's `langchain_id` human-readable in `psql` and match the natural key. The default UUID PK conflicts with string IDs — `init_vectorstore_table(id_column=Column("langchain_id", "TEXT", primary_key=True))` overrides it (§4.2).
>
> **Atomicity guarantee**: ALL writes for a single `ingest()` call live in ONE SQLAlchemy transaction. If anything fails mid-way, the transaction rolls back completely — no partial state. This is the key property that `PGVectorStore.add_documents` does NOT provide.

### 10.3 Worked example

Re-ingesting a 50-page PDF where **1 page changed**:

| | Chunks | Embedding API calls |
|---|---|---|
| Without content-hash cache (old design) | 50 written | 50 |
| With content-hash cache (this design) | 49 CASE A + 1 CASE B | **1** |

For OpenAI `text-embedding-3-small` at ~$0.02/1M tokens, that's ~$0.001 → ~$0.00002 per re-ingest — a ~50x cost reduction.

### 10.4 When to force re-embedding

Pass `reembed_unchanged=True` (CLI: `semsearch ingest --force file.pdf`) when:

- Tuning chunk_size/chunk_overlap (text is the same but you want new boundaries re-embedded).
- Manually invalidating a corrupted cache (e.g. suspected bad vectors from a prior partial failure).

> **Not needed for provider swaps**: switching `EMBEDDING_PROVIDER` to a provider with a different embedding dimension already requires `semsearch init --recreate` (§4.3, §8.1), which drops and re-creates the table. After a recreate the table is empty, so every chunk is CASE C on the next ingest — `reembed_unchanged` has nothing to do in this scenario. It's only relevant when the schema (and dimension) are staying the same but you want to bypass the content-hash cache for chunks whose hash hasn't changed.

### 10.5 Caveat: chunk_index stability

`chunk_index` is the chunk's 0-based position in the splitter's output for its source file. `RecursiveCharacterTextSplitter` is deterministic given the same `chunk_size`, `chunk_overlap`, and separators — so the same file ingested twice with unchanged settings will produce identical `chunk_index` assignments, and CASE A reuse kicks in.

If the user changes `chunk_size` between ingests, the boundaries shift and most chunks will have different content (and thus a different `document_hash`) than the row at the same `chunk_index`. They become CASE B and are re-embedded. This is correct behavior — it just isn't free. If you change chunking settings on purpose, prefer `semsearch init --recreate` to start from a clean slate.

---

## 11. Test Plan

### 11.1 Test Strategy

- **Unit tests**: pure functions (loader dispatch, splitter config, embedder factory, chunk ID derivation). No DB.
- **Integration tests**: real PostgreSQL via `testcontainers-python`. Each test gets a fresh container to avoid cross-test contamination.
- **CLI tests**: `typer.testing.CliRunner` against the integration DB.
- **Coverage target**: ≥85% line coverage on `src/semsearch`.

### 11.2 Test Cases

#### Unit (`tests/test_loaders.py`, `tests/test_embeddings.py`)

| ID | Case | Input | Expected |
|----|------|-------|----------|
| U-1 | pick_loader dispatches `.txt` | `Path("a.txt")` | returns TextLoader-backed callable |
| U-2 | pick_loader dispatches `.pdf` | `Path("a.pdf")` | returns PyMuPDFLoader-backed callable |
| U-3 | pick_loader rejects `.docx` | `Path("a.docx")` | raises `ValueError` |
| U-4 | pick_loader rejects missing file | `Path("missing.txt")` | raises `FileNotFoundError` |
| U-5 | build_embedder OpenAI without key | `provider=openai`, no `OPENAI_API_KEY` | raises `ProviderConfigError` |
| U-6 | build_embedder HuggingFace default | `provider=huggingface` | returns `HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2` |
| U-7 | build_embedder Ollama unreachable | `provider=ollama`, no server | raises on first embed call (connection refused) |
| U-8 | OpenRouter routing forwarded via `model_kwargs["extra_body"]` (NOT direct `extra_body=` kwarg) | build embedder with `type=openrouter`, `provider_order=["OpenAI","Together"]`; mock `OpenAIEmbeddings`; inspect constructed call | `model_kwargs == {"extra_body": {"provider": {"order": ["OpenAI","Together"]}}}` and NO `extra_body=` kwarg was passed |
| U-9 | OpenRouter routing fields ignored for non-openrouter types | `type=openai` with `provider_order` set | `model_kwargs` is `{}` (routing fields ignored) |

#### Integration (`tests/test_service_*.py`)

| ID | Case | Steps | Expected |
|----|------|-------|----------|
| I-1 | Ingest text file | ingest `sample.txt` (2 pages, ~3 chunks) | `chunks_added == 3`, row count in DB == 3 |
| I-2 | Ingest PDF | ingest `sample.pdf` (5 pages) | `chunks_added >= 5`, each chunk has `metadata.page` |
| I-3 | Ingest CSV | ingest `sample.csv` (10 rows) | `chunks_added == 10`, each chunk has `metadata.row` |
| I-4 | Search returns ranked results | ingest docs, search "password reset" with k=3 | returns ≤3 results, sorted by score DESC (higher = more similar; service converts `1 - distance`) |
| I-5 | Search empty table | fresh DB, search "anything" | returns `[]` |
| I-6 | **Delete by source** (exact) | ingest `a.pdf` and `b.pdf`, `delete({"source": "a.pdf"})` | `deleted_count == chunks_of_a` (computed via SELECT COUNT before delete), only `a.pdf` chunks gone, `b.pdf` intact |
| I-7 | **Delete by metadata filter** | ingest 2 invoices (`doc_type=invoice`) + 3 contracts (`doc_type=contract`), `delete({"doc_type": "invoice"})` | `deleted_count == 2`, only invoices gone |
| I-8 | Delete whole table (empty filter) | ingest 10 chunks, `delete({})` | `deleted_count == 10`, table empty (service bypasses PGVectorStore for empty filter and issues `DELETE FROM` directly via SQLAlchemy) |
| I-9  | Re-ingest idempotency (unchanged) | ingest `a.txt` (5 chunks), ingest `a.txt` again with `reembed_unchanged=False` | total row count unchanged (5); `chunks_added == 0`, `chunks_reused == 5`, `chunks_updated == 0` |
| I-9b | Re-ingest makes zero embedding calls | same as I-9, with `embedder.embed_documents` mocked | mock called **0 times** on 2nd run |
| I-10 | Re-ingest with single-chunk change | ingest `a.txt` (v1, 5 chunks), edit chunk #2, re-ingest | `chunks_reused == 4`, `chunks_updated == 1`, `chunks_added == 0`; only chunk #2 was embedded |
| I-10b | Verify embed call count on partial change | same as I-10, with `embedder.embed_documents` mocked | mock called exactly **once**, with a list of length 1 |
| I-10c | Verify DB state after partial change | after I-10 | row count == 5; chunk #2 has new `content` + `document_hash`; chunks #0,#1,#3,#4 keep their original `embedding` byte-for-byte |
| I-10d | **CASE D: stale tail chunks removed on file shortening** | ingest `a.txt` (5 chunks), edit file to produce only 3 chunks, re-ingest | `chunks_reused + chunks_updated == 3`, `chunks_pruned == 2`, row count == 3 (the 2 stale tail chunks are deleted in the same transaction) |
| I-10e | CASE D does NOT fire when file grows | ingest `a.txt` (3 chunks), edit to produce 5 chunks, re-ingest | `chunks_pruned == 0`; row count == 5 (no stale tail to delete) |
| I-11 | `--force` re-embeds everything | ingest `a.txt`, then `semsearch ingest --force a.txt` | `chunks_reused == 0`, `chunks_updated == 5`, `chunks_added == 0`, `chunks_pruned == 0`; mock called once with list of 5 |
| I-12 | `delete --all` without `--yes` is blocked | CLI `semsearch delete --all` | exits non-zero, prints error |
| I-13 | `delete --all --yes` wipes table | ingest 10 chunks, `semsearch delete --all --yes` | `deleted_count == 10`, table empty |
| I-14 | Provider swap | init with HuggingFace, ingest, search; then `init --recreate` with OpenAI, ingest, search | both providers return valid search results (no schema mismatch error) |
| I-15 | Search with `$ilike` prefix filter (grep-style) | ingest `docs/a.txt`, `docs/sub/b.txt`, `other/c.txt`; search with `filter={"source": {"$ilike": "docs/%"}}` | returns chunks only from `a.txt` and `b.txt`; `c.txt` excluded |
| I-16 | Search with exact source filter | ingest 2 files; search with `filter={"source": "docs/a.txt"}` | returns chunks only from `docs/a.txt` |
| I-17 | Search with `doc_type` filter | ingest mix of pdf/txt/csv; search with `filter={"doc_type": "pdf"}` | returns only PDF chunks |
| I-18 | Combined `$and` filter | ingest `docs/a.pdf`, `docs/b.txt`, `other/c.pdf`; search with `filter={"$and": [{"source": {"$ilike": "docs/%"}}, {"doc_type": "pdf"}]}` | returns only `docs/a.pdf` chunks |
| I-19 | Filter matches nothing | ingest `docs/a.txt`; search with `filter={"source": {"$ilike": "nonexistent/%"}}` | returns `[]`, no error |
| I-20 | Filter with numeric comparison | ingest invoice chunks with `year=2024` in `langchain_metadata`; search `filter={"year": {"$gte": 2024}}` | returns only 2024 invoices |
| I-21 | OpenRouter default base_url when unset | `type=openrouter`, `base_url=None` | requests go to `https://openrouter.ai/api/v1` |
| I-22 | OpenRouter `allow_fallbacks=false` propagated | `provider_allow_fallbacks=False` (explicitly set; default is None which OMITS the field) | `model_kwargs["extra_body"]["provider"]["allow_fallbacks"] == False` |
| I-22b | OpenRouter `allow_fallbacks=None` (default) is OMITTED, not emitted as True | `provider_allow_fallbacks=None` (default) | `"allow_fallbacks" not in model_kwargs["extra_body"]["provider"]` (does not contradict docstring) |
| I-23 | OpenRouter `max_price` propagated | `provider_max_price={"prompt":1}` (embeddings use "prompt" only, not "completion") | `model_kwargs["extra_body"]["provider"]["max_price"] == {"prompt":1}` |
| I-24 | `openai_compatible` hits custom base_url | `type=openai_compatible`, `base_url=http://localhost:1234/v1`; capture HTTP requests | requests go to localhost:1234, not api.openai.com |
| I-25 | OpenRouter down → error propagates | mock `OpenAIEmbeddings.embed_documents` to raise `APIConnectionError`; ingest | `FileIngestError` raised; transaction rolled back; no DB writes |
| I-26 | CLI `--provider` flag overrides env | settings has `type=huggingface`; run `semsearch ingest --provider openrouter --provider-model ...` | embedder built with openrouter, not huggingface |
| I-27 | OpenRouter `data_collection` propagated | `provider_data_collection="deny"` | `model_kwargs["extra_body"]["provider"]["data_collection"] == "deny"` |
| I-28 | OpenRouter `only` whitelist propagated | `provider_only=["openai","azure"]` (lowercase slugs) | `model_kwargs["extra_body"]["provider"]["only"] == ["openai","azure"]` |
| I-28b | OpenRouter provider slugs normalized to lowercase | `provider_order=["OpenAI","Together"]` (uppercase input) | `model_kwargs["extra_body"]["provider"]["order"] == ["openai","together"]` (builder lowercases) |
| I-28c | OpenRouter `provider_ignore` slugs normalized to lowercase | `provider_ignore=["DeepInfra"]` | `model_kwargs["extra_body"]["provider"]["ignore"] == ["deepinfra"]` |
| I-29 | `ingest_dir` with mixed files | ingest dir containing 2 txt + 1 pdf + 1 unsupported .docx | `files_discovered == 4`, `files_skipped_unsupported == 1`, `files_attempted == 3`, `files_succeeded == 3`, `aggregate.chunks_added == sum of 3 files' chunks` |
| I-30 | `ingest_dir` with `--glob` filter | ingest dir with `--glob "**/*.pdf"` | only PDFs ingested; `files_attempted == pdf_count` |
| I-31 | `ingest_dir` with `--exclude` | ingest dir with `--exclude "*/draft/*"` | draft subdir skipped; `files_attempted` excludes draft files |
| I-32 | `ingest_dir` with failing file + `continue_on_error=True` | ingest dir with one corrupt PDF | `files_failed == 1`, other files still ingested, `failed_files` contains the corrupt file |
| I-33 | `ingest_dir` with failing file + `--no-continue-on-error` | same as I-32 but `--no-continue-on-error` | raises `FileIngestError` on the corrupt file, no subsequent files attempted |
| I-34 | `ingest_dir` skips hidden files | ingest dir with `.secret.txt` and `notes.md` | only `notes.md` ingested |
| I-35 | `ingest_dir` skips symlinks by default | ingest dir with a symlink to a real file | symlink skipped; with `--follow-symlinks`, symlink target ingested |
| I-36 | `ingest_dir` idempotent on re-run | ingest dir twice unchanged | 2nd run: `aggregate.chunks_reused == total`, `chunks_added == 0`, zero embedding API calls |
| I-37 | `ingest_dir` with new file added | ingest dir v1, add file, ingest dir v2 | `aggregate.chunks_added == new_file_chunks` on 2nd run; existing files hit CASE A |
| I-38 | `ingest_dir --prune` deletes orphans | ingest dir v1, delete file from disk, ingest dir v2 --prune | `pruned_sources == ["<deleted_file>"]`, `pruned_chunks == deleted_file_chunks`, total chunk count back to v1 minus deleted |
| I-39 | `ingest_dir` without `--prune` leaves orphans | same as I-38 but no `--prune` flag | `pruned_sources == []`, `pruned_chunks == 0`, deleted file's chunks remain in DB |
| I-40 | `ingest_dir --prune --dry-run` lists but doesn't delete | same as I-38 with `--dry-run` | `pruned_sources == ["<deleted_file>"]`, `pruned_chunks == 0`, chunks remain in DB |
| I-41 | `ingest_dir --prune` handles file rename | ingest dir v1, rename file, ingest dir v2 --prune | old `source` in `pruned_sources`, new `source` ingested as CASE C (re-embeds) — content reuse across rename is NOT supported in v1 |
| I-42 | `ingest_dir --prune` handles file move to subdir | same as I-41 with move | same as I-41 |
| I-43 | `ingest_dir --prune` only deletes sources under `dir_path/` | ingest two dirs `data/` and `other/`; run `ingest_dir data/ --prune` after deleting a file from `other/` | `other/`'s chunks NOT pruned (prefix match protects them); `pruned_sources` contains only `data/`-prefixed entries |

#### CLI (`tests/test_cli.py`)

| ID | Case | Command | Expected |
|----|------|---------|----------|
| C-1 | `init` idempotent | `semsearch init` x2 | both succeed, no error on 2nd |
| C-2 | `ingest` prints JSON | `semsearch ingest ./data/sample.txt` | stdout is valid JSON with `chunks_added` |
| C-3 | `search` returns results | `semsearch search "test" --k 3` | stdout is valid JSON with `results` array |
| C-4 | `delete --filter` works | `semsearch delete --filter '{"source": "./data/sample.txt"}'` | stdout is JSON with `deleted_count` > 0 |
| C-5 | `delete --all` without `--yes` is blocked | `semsearch delete --all` | exits non-zero, prints error |
| C-6 | `stats` shows counts | `semsearch stats` | stdout JSON matches §8.5 schema |

### 11.3 Test Fixtures (`conftest.py`)

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg

@pytest.fixture(scope="session")
def settings(pg_container):
    from semsearch.config import Settings, EmbeddingProviderConfig
    return Settings(
        database_url=pg_container.get_connection_url(driver="psycopg"),
        embedding_provider=EmbeddingProviderConfig(
            type="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

@pytest.fixture
def service(settings):
    from semsearch.service import SemanticSearchService
    with SemanticSearchService.from_settings(settings) as svc:
        svc.init_schema(recreate=True)  # fresh table per test
        yield svc
```

> **Why `init_schema(recreate=True)` per test**: each test gets a clean table, avoiding cross-test contamination. The `recreate=True` flag drops the table (CASCADE) and re-creates it with the active provider's vector dim. With testcontainers spinning up a fresh Postgres container per session, this is fast enough for unit-test-style integration tests.

---

## 12. Error Handling & Logging

- **Logging**: stdlib `logging`, configured via `logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))`. Module-level logger `semsearch`.
- **Error responses**: CLI catches `SemSearchError` subclasses, prints `{"error": "...", "type": "..."}` to stderr, exits with code 1. Unexpected exceptions propagate (exit code 2) so CI fails loudly.
- **Transaction safety**: `ingest` writes are **service-owned** — a single `engine.begin() ... ` SQLAlchemy transaction wraps all CASE A/B/C/D statements for one `ingest()` call (see §3.2, §10.2). `PGVectorStore.add_documents()` is NOT used for writes. If anything fails mid-way, the transaction is rolled back in full and a `FileIngestError` is raised — no partial state. `delete()` similarly wraps its `SELECT COUNT(*)` + `PGVectorStore.delete(filter=...)` pair in one transaction (§3.4), raising `DeleteError` on failure. Only `search()` (a read-only `similarity_search_with_score` call) actually delegates to `PGVectorStore` directly.
- **Connection retries**: the `PGEngine` connection pool handles transient failures; no application-level retry logic in v1.

---

## 13. Security Considerations

1. **Secrets**: API keys live in environment variables only — never in code, never logged. `Settings.embedding_provider.api_key` is a `SecretStr`, so it does not leak in `repr()`.
2. **SQL injection**: two separate surfaces, both mitigated:
   - **Search & delete filters**: go through `PGVectorStore.similarity_search(filter=...)` and `PGVectorStore.delete(filter=...)`. Filter values are JSON-serialized and parameterized by langchain-postgres / psycopg3 — not a concern.
   - **Service-owned write path (§3.2, §10.2)**: `ingest()` executes raw parameterized SQL directly via SQLAlchemy (`INSERT ... ON CONFLICT`, `UPDATE`, `DELETE WHERE chunk_index >= :new_len`). All *values* (source, chunk_index, hash, vector, metadata) are bound parameters — safe by construction. The one place a string is interpolated rather than bound is the **table name** itself (`collection_name`, used in every write-path statement and in `init_vectorstore_table(table_name=...)`), because Postgres does not allow identifiers to be bound parameters. This is why `Settings.collection_name` is validated at load time against `_TABLE_NAME_RE` (§6.1) — reject anything that isn't `^[a-z_][a-z0-9_]{0,62}$` before it ever reaches a SQL string. Any code path that builds write-path SQL must re-use that validated value, never accept a table name from an unvalidated source (e.g. a CLI flag added later without going through `Settings`).
3. **Role separation**: the application role `semsearch_app` has `CREATE` on `public` schema (needed for `init_vectorstore_table`) but `NOSUPERUSER`. For production, tighten this: pre-create the table with a migration role, then grant only `INSERT/SELECT/DELETE` to the app role.
4. **Delete safety**: `delete({})` (empty filter) deletes the entire table. The CLI wraps this with `--all --yes`; programmatic callers are expected to be careful.

---

## 14. Local Development Setup

### 14.1 Prerequisites

- Python 3.11+
- Docker (for local Postgres) or a remote PostgreSQL 14+ with pgvector
- (Optional) Ollama daemon if testing with the Ollama provider
- (Optional) OpenAI API key if testing with OpenAI

### 14.2 Steps

```bash
# 1. Clone & enter
git clone <repo> pg-semantic-search && cd pg-semantic-search

# 2. Create venv
python -m venv .venv && source .venv/bin/activate

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Start Postgres with pgvector (one-line via Docker)
docker run -d --name pgvector-dev \
  -e POSTGRES_USER=semsearch_app \
  -e POSTGRES_PASSWORD=change_me_in_prod \
  -e POSTGRES_DB=semsearch \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 5. Run the one-time SQL setup (§4.1)
psql "postgresql://semsearch_app:change_me_in_prod@localhost:5432/semsearch" \
  -f scripts/init_db.sql

# 6. Configure
cp .env.example .env
# edit .env if you want a different provider

# 7. Initialize schema
semsearch init

# 8. Ingest
semsearch ingest ./data/sample.txt

# 9. Search
semsearch search "your query here" --k 5

# 10. Delete
semsearch delete --filter '{"source": "./data/sample.txt"}'

# 11. Run tests
pytest -v
```

---

## 15. Open Questions & Future Work

These are explicitly **out of v1** but noted here so they aren't forgotten:

1. **HNSW tuning**: with HNSW, `ef_search` should be configurable per-query. Currently hardcoded to default.
2. **Embedding dim auto-detection**: schema mismatch on provider swap is currently resolved by `--recreate`. A migration tool that re-embeds existing chunks under a new provider would be valuable.
3. **Parallel batch ingest**: `ingest_dir` is currently sequential. For very large directories (>1k files), parallel ingestion with a bounded worker pool would reduce wall time. Must preserve per-file transaction isolation and deterministic `failed_files` ordering.
4. **Content-hash reuse across renames**: when a file is renamed, `--prune` correctly deletes the old chunks and re-embeds the new ones — but the content is identical. A future enhancement could detect content-hash matches across different `source` values and reuse the embedding (would require a secondary index on `document_hash` and a more complex UPSERT path).
5. **Hybrid search**: `PGVectorStore` supports `HybridSearchConfig` (vector + BM25 full-text). Future `search --hybrid` flag could expose this.
6. **RAG**: once a stable retrieval layer is in place, add a `qa` command that wires up `RetrievalQA` with an LLM of choice.
7. **Observability**: structured logging + OpenTelemetry spans for ingest/search/delete.
8. **Async**: `AsyncSemanticSearchService` mirroring the sync API on top of `PGVectorStore`'s async methods (`PGVectorStore.create_async`, `asimilarity_search_with_score`, `adelete`).

---

## 16. Acceptance Checklist

Before the implementation is considered "done", the following must all be true:

- [ ] All public API methods in §7 implemented with exact signatures.
- [ ] All unit tests in §11.2 pass (including U-8/U-9 verifying `model_kwargs["extra_body"]` nesting for OpenRouter).
- [ ] All integration tests in §11.2 pass against a real Postgres + pgvector container.
- [ ] All CLI commands in §8 work as described, with JSON output matching the schemas.
- [ ] `.env.example` is sufficient to bootstrap a fresh dev environment.
- [ ] `README.md` covers quickstart in ≤50 lines.
- [ ] `pytest --cov=src/semsearch` reports ≥85% line coverage.
- [ ] Switching `SEMSEARCH_EMBEDDING_PROVIDER__TYPE` between `openai`, `huggingface`, `ollama`, `openrouter`, and `openai_compatible` requires only env-var changes (after `init --recreate` if dims differ).
- [ ] `delete({"source": "<file>"})` removes exactly that file's chunks and nothing else (verified by I-6).
- [ ] `ingest_dir(path, prune=True)` correctly deletes orphans (verified by I-38) and leaves foreign-dir chunks untouched (verified by I-43).
- [ ] `ingest_dir` re-run on unchanged dir makes zero embedding API calls (verified by I-36).
- [ ] Implementation uses `PGVectorStore` (the new class) — NOT the deprecated `PGVector`.
- [ ] OpenRouter routing fields are forwarded via `model_kwargs={"extra_body": {...}}` — NOT via a direct `extra_body=` kwarg.
- [ ] Search results expose `score = 1.0 - cosine_distance` (similarity, not raw distance from `similarity_search_with_score`).
- [ ] Write path is service-owned SQLAlchemy transaction (NOT `PGVectorStore.add_documents`) — verified by I-10d (CASE D atomicity).
- [ ] `langchain_id` column is TEXT (not UUID) — required for deterministic string IDs `"{source}::{chunk_index}"`.
- [ ] `SEMSEARCH_EMBEDDING_PROVIDER__*` env vars use double underscore (single underscore will NOT be recognized by pydantic-settings).
- [ ] OpenRouter provider slugs are lowercase in config and normalized to lowercase before sending (verified by I-28b/I-28c).
