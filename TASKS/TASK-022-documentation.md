# TASK-022: Documentation & Polish

> **Phase**: 11 | **Priority**: Medium | **Status**: Not Started
> **Depends on**: TASK-001 through TASK-021
> **Blocks**: TASK-023

## Objective

Write README.md and perform final code review against spec checklist.

## Files to Create/Modify

### 1. `README.md`

≤50 lines covering:

```markdown
# Semantic Search Service

Semantic search over local documents using LangChain + PostgreSQL + pgvector.

## Supported File Types
- Text: .txt, .md
- PDF: .pdf
- Structured: .csv, .json

## Quickstart

```bash
# Install
pip install -e .

# Start PostgreSQL with pgvector
docker run -d --name pgvector -e POSTGRES_USER=semsearch_app -e POSTGRES_PASSWORD=change_me -e POSTGRES_DB=semsearch -p 5432:5432 pgvector/pgvector:pg16

# Initialize
semsearch init

# Ingest
semsearch ingest ./data/handbook.pdf

# Search
semsearch search "how do I reset my password" --k 5

# Delete
semsearch delete --filter '{"source": "./data/handbook.pdf"}'
```

## Configuration

See `.env.example` for all options. Key settings:
- `SEMSEARCH_EMBEDDING_PROVIDER__TYPE` — Provider: openai, huggingface, ollama, openrouter, openai_compatible
- `SEMSEARCH_DATABASE_URL` — PostgreSQL connection string

## CLI Commands

- `semsearch init` — Create/migrate table
- `semsearch ingest <file>` — Ingest a single file
- `semsearch ingest-dir <dir>` — Recursively ingest directory
- `semsearch search <query>` — Similarity search
- `semsearch delete --filter <json>` — Delete matching chunks
- `semsearch stats` — Show table statistics
- `semsearch reingest <file>` — Delete + re-ingest
```

### 2. Code Review Checklist

Verify against SPEC §16:

#### Architecture
- [ ] Write path is service-owned SQL (NOT `PGVectorStore.add_documents`)
- [ ] `PGVectorStore` used ONLY for `similarity_search_with_score` and `delete(filter=...)`
- [ ] `langchain_id` is TEXT with deterministic IDs `"{source}::{chunk_index}"`
- [ ] Content-hash caching (CASE A/B/C/D) avoids re-embedding
- [ ] Search scores converted: `score = 1.0 - cosine_distance`
- [ ] OpenRouter routing via `model_kwargs={"extra_body": {...}}` (NOT direct `extra_body=`)

#### Configuration
- [ ] `env_nested_delimiter="__"` in Settings
- [ ] Double underscore in all `SEMSEARCH_EMBEDDING_PROVIDER__*` env vars
- [ ] `collection_name` validated against `_TABLE_NAME_RE`
- [ ] `SecretStr` for API keys
- [ ] `provider_allow_fallbacks` defaults to None (not True)

#### Database
- [ ] `init_schema` idempotent (checks `information_schema.tables`)
- [ ] HNSW, GIN, and composite indexes created separately
- [ ] `UNIQUE (source, chunk_index)` constraint enforced
- [ ] `ON CONFLICT (source, chunk_index) DO UPDATE` in ingest SQL
- [ ] `DELETE WHERE chunk_index >= :new_len` for CASE D

#### API
- [ ] All public methods match SPEC §7 signatures
- [ ] `SearchResult` models correct with score as similarity
- [ ] `DeleteResult` includes filter dict
- [ ] `IngestResult` has all four count fields
- [ ] `BatchIngestResult` has all required fields

#### CLI
- [ ] All commands implement expected flags
- [ ] JSON output to stdout, errors to stderr
- [ ] `--all` requires `--yes` confirmation
- [ ] Provider overrides work

#### Error Handling
- [ ] Exception hierarchy: `SemSearchError` base class
- [ ] `FileIngestError` for ingest failures
- [ ] `SearchError` for search failures
- [ ] `DeleteError` for delete failures
- [ ] Transactions roll back on error

#### Testing
- [ ] ≥85% line coverage
- [ ] All unit tests pass (U-1 to U-9)
- [ ] All integration tests pass (I-1 to I-43)
- [ ] All CLI tests pass (C-1 to C-6)
- [ ] Testcontainers for real Postgres testing

## Verification

- [ ] README ≤50 lines
- [ ] README quickstart works from clean clone
- [ ] All checklist items verified
- [ ] No PGVectorStore.add_documents() calls in codebase
- [ ] grep for "extra_body" confirms no direct kwarg usage
