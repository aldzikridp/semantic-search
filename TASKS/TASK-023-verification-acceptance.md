# TASK-023: Verification & Acceptance

> **Phase**: 12 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-001 through TASK-022
> **Blocks**: None (final task)

## Objective

Run the full acceptance checklist from SPEC §16 to verify the implementation is complete and correct.

## Verification Steps

### 1. Run All Tests

```bash
# Run with verbose output
pytest -v --tb=short

# Expected: All tests pass
```

### 2. Check Coverage

```bash
# Generate coverage report
pytest --cov=src/semsearch --cov-report=term-missing

# Expected: ≥85% line coverage
```

### 3. Manual Smoke Test

```bash
# Initialize schema
semsearch init

# Ingest a file
semsearch ingest ./tests/fixtures/sample.txt

# Search
semsearch search "test query" --k 5

# Delete
semsearch delete --filter '{"source": "./tests/fixtures/sample.txt"}'

# Stats
semsearch stats
```

### 4. Provider Swap Test

```bash
# With OpenAI (default)
semsearch init --recreate --yes
semsearch ingest ./tests/fixtures/sample.txt
semsearch search "test" --k 3

# Switch to Ollama
# Edit .env: SEMSEARCH_EMBEDDING_PROVIDER__TYPE=ollama
semsearch init --recreate --yes
semsearch ingest ./tests/fixtures/sample.txt
semsearch search "test" --k 3
```

### 5. Directory Ingest + Prune Test

```bash
# Initial ingest
semsearch ingest-dir ./tests/fixtures/

# Delete a file from fixtures
rm ./tests/fixtures/sample.txt

# Dry run prune
semsearch ingest-dir ./tests/fixtures/ --prune --dry-run

# Actual prune
semsearch ingest-dir ./tests/fixtures/ --prune
```

### 6. Reingest Test

```bash
# Ingest
semsearch ingest ./tests/fixtures/sample.csv

# Reingest (should reuse chunks)
semsearch reingest ./tests/fixtures/sample.csv

# Verify stats
semsearch stats
```

### 7. Filter Test

```bash
# Ingest multiple files
semsearch ingest-dir ./tests/fixtures/

# Search with filter
semsearch search "test" --filter '{"source": {"$ilike": "tests/fixtures/%"}}'

# Search with doc_type filter
semsearch search "test" --filter '{"doc_type": "text"}'
```

## Acceptance Checklist (SPEC §16)

- [ ] All public API methods in §7 implemented with exact signatures
- [ ] All unit tests in §11.2 pass (including U-8/U-9)
- [ ] All integration tests in §11.2 pass against real Postgres
- [ ] All CLI commands in §8 work as described, with JSON output
- [ ] `.env.example` sufficient to bootstrap fresh dev environment
- [ ] `README.md` covers quickstart in ≤50 lines
- [ ] `pytest --cov=src/semsearch` reports ≥85% line coverage
- [ ] Provider switching requires only env-var changes (after `init --recreate` if dims differ)
- [ ] `delete({"source": "<file>"})` removes exactly that file's chunks (I-6)
- [ ] `ingest_dir(path, prune=True)` correctly deletes orphans (I-38) and leaves foreign-dir chunks untouched (I-43)
- [ ] `ingest_dir` re-run on unchanged dir makes zero embedding API calls (I-36)
- [ ] Implementation uses `PGVectorStore` — NOT deprecated `PGVector`
- [ ] OpenRouter routing via `model_kwargs={"extra_body": {...}}` — NOT direct `extra_body=`
- [ ] Search results expose `score = 1.0 - cosine_distance` (similarity, not distance)
- [ ] Write path is service-owned SQLAlchemy transaction — NOT `PGVectorStore.add_documents`
- [ ] `langchain_id` column is TEXT (not UUID)
- [ ] `SEMSEARCH_EMBEDDING_PROVIDER__*` uses double underscore
- [ ] OpenRouter provider slugs are lowercase (I-28b/I-28c)

## Critical Gotchas to Verify

1. **`PGVectorStore.add_documents()` is NEVER called** — grep codebase
2. **`model_kwargs` for OpenRouter** — grep for direct `extra_body=` usage
3. **Score conversion** — verify in test assertions
4. **Idempotent schema** — verify `init_schema` can be called twice
5. **CASE D atomicity** — verify stale chunks deleted in same transaction

## Final Verification Command

```bash
# Run everything
pytest -v --cov=src/semsearch --cov-report=term-missing --tb=short

# Manual end-to-end
semsearch init --recreate --yes
semsearch ingest-dir ./tests/fixtures/
semsearch search "test query" --k 5 --filter '{"doc_type": "text"}'
semsearch delete --filter '{"source": {"$ilike": "tests/fixtures/sample%"}}'
semsearch stats
semsearch delete --all --yes
semsearch stats
```

## Sign-off

When all checkboxes above are checked:

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Coverage ≥85%
- [ ] Documentation complete
- [ ] Ready for use
