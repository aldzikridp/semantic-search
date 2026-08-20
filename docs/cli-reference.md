# CLI Reference

All commands output JSON to stdout. Errors go to stderr.

## `semsearch init`

Create or migrate the database table.

```bash
semsearch init [--recreate] [--yes]
```

| Flag | Description |
|------|-------------|
| `--recreate` | Drop and recreate table (WARNING: deletes all data) |
| `--yes` | Skip confirmation for `--recreate` |

**Examples:**

```bash
# Create table (idempotent)
semsearch init

# Recreate from scratch
semsearch init --recreate --yes
```

**Output:**

```json
{
  "status": "created",
  "table": "semsearch_chunks"
}
```

---

## `semsearch ingest`

Ingest a single file.

```bash
semsearch ingest <path> [--force] [--provider TYPE] [--provider-model MODEL] [--provider-api-key KEY]
```

| Argument/Flag | Description |
|---------------|-------------|
| `path` | File to ingest (required) |
| `--force` | Force re-embed all chunks (bypass cache) |
| `--provider` | Override provider type |
| `--provider-model` | Override model name |
| `--provider-api-key` | Override API key |
| `--provider-base-url` | Override base URL |
| `--provider-order` | OpenRouter provider order (JSON) |
| `--provider-allow-fallbacks` | OpenRouter allow fallbacks |
| `--provider-ignore` | OpenRouter ignore list (JSON) |

**Examples:**

```bash
# Basic ingest
semsearch ingest docs/readme.md

# Force re-embed
semsearch ingest docs/readme.md --force

# Use different provider for this ingest
semsearch ingest docs/readme.md \
  --provider openrouter \
  --provider-model openai/text-embedding-3-small \
  --provider-api-key sk-or-v1-...
```

**Output:**

```json
{
  "source": "docs/readme.md",
  "chunks_added": 5,
  "chunks_reused": 0,
  "chunks_updated": 0,
  "chunks_pruned": 0,
  "ingested_at": "2026-08-18T06:48:57.984588+00:00"
}
```

**Cases:**

| Case | Meaning |
|------|---------|
| `chunks_added` | New chunks embedded |
| `chunks_reused` | Unchanged chunks (no API call) |
| `chunks_updated` | Changed chunks re-embedded |
| `chunks_pruned` | Stale tail chunks deleted |

---

## `semsearch ingest-dir`

Recursively ingest all supported files in a directory.

```bash
semsearch ingest-dir <dir_path> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `dir_path` | Directory to walk (required) |
| `--glob PATTERN` | Glob pattern (default: `**/*`) |
| `--exclude PATTERN` | fnmatch patterns to skip (repeatable) |
| `--prune` | Delete orphaned chunks after ingest |
| `--dry-run` | Preview what prune would delete |
| `--continue-on-error/--no-continue-on-error` | Continue on file failure (default: continue) |
| `--follow-symlinks` | Follow symlinks during discovery |
| `--force` | Force re-embed all chunks |

**Examples:**

```bash
# Ingest all files
semsearch ingest-dir docs/

# Only PDFs
semsearch ingest-dir data/ --glob "**/*.pdf"

# Skip drafts
semsearch ingest-dir data/ --exclude "*/draft/*"

# Ingest and clean up deleted files
semsearch ingest-dir docs/ --prune

# Preview what prune would delete
semsearch ingest-dir docs/ --prune --dry-run

# Abort on first failure
semsearch ingest-dir docs/ --no-continue-on-error
```

**Output:**

```json
{
  "dir": "TASKS",
  "files_discovered": 24,
  "files_skipped_unsupported": 0,
  "files_attempted": 24,
  "files_succeeded": 24,
  "files_failed": 0,
  "failed_files": [],
  "aggregate": {
    "chunks_added": 149,
    "chunks_reused": 0,
    "chunks_updated": 0,
    "chunks_pruned": 0
  },
  "elapsed_seconds": 157.4,
  "pruned_sources": [],
  "pruned_chunks": 0
}
```

---

## `semsearch search`

Cosine similarity search.

```bash
semsearch search <query> [--k N] [--filter JSON] [--rerank]
```

| Argument/Option | Description |
|-----------------|-------------|
| `query` | Search query (required) |
| `--k N` | Top-k results (default: 5) |
| `--filter JSON` | PGVectorStore filter dict |

**Examples:**

```bash
# Basic search
semsearch search "how to deploy"

# Top 10 results
semsearch search "api docs" --k 10

# Filter by doc_type
semsearch search "pdf content" --filter '{"doc_type": "pdf"}'

# With reranking
semsearch search "database setup" --rerank --k 5

# Filter by source prefix
semsearch search "old docs" --filter '{"source": {"$ilike": "docs/old/%"}}'

# Combined filter
semsearch search "test" --filter '{"$and": [{"doc_type": "text"}, {"source": {"$ilike": "TASKS/%"}}]}'
```

**Filter Operators:**

| Operator | Example |
|----------|---------|
| Exact match | `{"source": "docs/file.pdf"}` |
| `$ilike` | `{"source": {"$ilike": "docs/%"}}` |
| `$eq` | `{"doc_type": {"$eq": "pdf"}}` |
| `$ne` | `{"doc_type": {"$ne": "csv"}}` |
| `$gt`, `$gte` | `{"year": {"$gte": 2024}}` |
| `$lt`, `$lte` | `{"year": {"$lt": 2024}}` |
| `$in` | `{"doc_type": {"$in": ["pdf", "csv"]}}` |
| `$nin` | `{"doc_type": {"$nin": ["json"]}}` |
| `$between` | `{"page": {"$between": [1, 10]}}` |
| `$like` | `{"source": {"$like": "docs/%"}}` |
| `$and` | `{"$and": [{...}, {...}]}` |
| `$or` | `{"$or": [{...}, {...}]}` |
| `$exists` | `{"page": {"$exists": true}}` |
| `$not` | `{"$not": {"doc_type": "pdf"}}` |

**Output:**

```json
{
  "query": "database setup",
  "k": 3,
  "filter": null,
  "results": [
    {
      "id": "TASKS/TASK-007-database-store.md::0",
      "content": "# TASK-007: Database Store...",
      "score": 0.537,
      "source": "TASKS/TASK-007-database-store.md",
      "chunk_index": 0,
      "page": null,
      "row": null,
      "doc_type": "text",
      "metadata": {...}
    }
  ]
}
```

---

## `semsearch delete`

Delete chunks by filter.

```bash
semsearch delete [--filter JSON] [--all] [--yes]
```

| Option | Description |
|--------|-------------|
| `--filter JSON` | Filter dict (same syntax as search) |
| `--all` | Delete everything |
| `--yes` | Confirm `--all` |

**Examples:**

```bash
# Delete by source
semsearch delete --filter '{"source": "docs/old.md"}'

# Delete by doc_type
semsearch delete --filter '{"doc_type": "pdf"}'

# Delete by prefix
semsearch delete --filter '{"source": {"$ilike": "docs/old/%"}}'

# Delete everything (requires --yes)
semsearch delete --all --yes
```

**Output:**

```json
{
  "deleted_count": 5,
  "filter": {
    "source": "docs/old.md"
  }
}
```

---

## `semsearch stats`

Show table statistics.

```bash
semsearch stats
```

**Output:**

```json
{
  "table": "semsearch_chunks",
  "embedding_provider": "openrouter",
  "embedding_dim": 4096,
  "chunk_count": 149,
  "source_count": 24,
  "sources_by_count": [
    ["TASKS/TASK-014-cli.md", 13],
    ["TASKS/TASK-016-integration-tests-ingest.md", 10]
  ]
}
```

---

## `semsearch reingest`

Delete + ingest in one step. Forces re-embedding.

```bash
semsearch reingest <path>
```

**Example:**

```bash
semsearch reingest docs/readme.md
```

**Output:**

```json
{
  "source": "docs/readme.md",
  "chunks_added": 5,
  "chunks_reused": 0,
  "chunks_updated": 0,
  "chunks_pruned": 0,
  "ingested_at": "2026-08-18T06:48:57.984588+00:00"
}
```

---

## `semsearch serve`

Start the HTTP server. Keeps the service warm between requests, eliminating cold-start overhead for AI agent tool calling.

```bash
semsearch serve [--host HOST] [--port PORT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8383` | Bind port |

**Examples:**

```bash
# Start with defaults
semsearch serve

# Custom host and port
semsearch serve --host 127.0.0.1 --port 9000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/search` | Similarity search (supports `rerank`) |
| `GET` | `/stats` | Table statistics |
| `GET` | `/docs` | OpenAPI documentation |

**Note:** The server is read-only. Use CLI commands for data modification:
```bash
semsearch ingest <file>       # Ingest single file
semsearch ingest-dir <dir>    # Ingest directory
semsearch delete --filter ... # Delete by filter
```

**AI Agent Integration:**

```python
import httpx2

client = httpx2.Client(base_url="http://localhost:8383")

# Search
results = client.post("/search", json={"query": "how to deploy", "k": 5}).json()

# Stats
stats = client.get("/stats").json()
```

---

## `semsearch version`

Show version.

```bash
semsearch version
```

**Output:**

```
semsearch 0.1.0
```

---

## Common Patterns

### Idempotent ingest

```bash
# First ingest (calls API)
semsearch ingest docs/file.md
# chunks_added: 5

# Second ingest (no API call)
semsearch ingest docs/file.md
# chunks_reused: 5
```

### Update a file

```bash
# Edit the file
vim docs/file.md

# Re-ingest (only changed chunks re-embedded)
semsearch ingest docs/file.md
# chunks_reused: 3, chunks_updated: 2
```

### Directory with prune

```bash
# Initial ingest
semsearch ingest-dir docs/

# Delete a file
rm docs/old.md

# Clean up orphaned chunks
semsearch ingest-dir docs/ --prune
# pruned_sources: ["docs/old.md"]
```

### Switch provider

```bash
# 1. Update .env
vim .env

# 2. Recreate table (different dimensions)
semsearch init --recreate --yes

# 3. Re-ingest everything
semsearch ingest-dir docs/
```

## Global Options

### `--config` / `-c`

Specify a custom config file path instead of the default `.env`.

```bash
semsearch --config prod.env search "query"
semsearch -c staging.env ingest file.md
semsearch -c /path/to/config.env stats
```

**Notes:**
- File must exist (error if not found)
- Environment variables override config file values
- If not specified, uses `.env` in current directory

**Example config files:**

```bash
# prod.env
SEMSEARCH_DATABASE_URL=postgresql://prod:pass@prod-db:5432/semsearch
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-large

# staging.env
SEMSEARCH_DATABASE_URL=postgresql://staging:pass@staging-db:5432/semsearch
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
```

**Environment variable override:**

```bash
# Config file has: SEMSEARCH_DATABASE_URL=postgres://...
# Env var overrides:
SEMSEARCH_DATABASE_URL=postgres://other semsearch --config prod.env search "query"
# Uses postgres://other (env var wins)
```
