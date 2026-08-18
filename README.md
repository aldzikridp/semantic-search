# semsearch

Semantic search over local documents using LangChain + PostgreSQL + pgvector.

Ingest files (PDF, CSV, JSON, TXT, MD), split into chunks, generate embeddings, and search by meaning — not just keywords.

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit .env with your database URL and API key

# 2. Initialize the database
semsearch init

# 3. Ingest files
semsearch ingest docs/readme.md
semsearch ingest-dir docs/

# 4. Search
semsearch search "how to reset password" --k 5

# 5. View stats
semsearch stats
```

## Installation

### NixOS (recommended)

```bash
git clone <repo-url> && cd semantic-search
nix develop  # enters dev shell, installs dependencies automatically
```

### Other systems (Linux/macOS)

```bash
git clone <repo-url> && cd semantic-search

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with your preferred provider
pip install -e ".[openai]"        # OpenAI only
pip install -e ".[ollama]"       # Ollama only
pip install -e ".[all]"           # All providers

# For development (includes test dependencies)
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your database URL and API key

# Initialize database
semsearch init
```

**Available extras:**

| Extra | Packages | Use case |
|-------|----------|----------|
| `openai` | langchain-openai | OpenAI, OpenRouter, OpenAI-compatible |
| `ollama` | langchain-ollama | Local Ollama daemon |
| `all` | Both above | All providers |
| `test` | pytest, testcontainers | Running tests |
| `dev` | all + test + ruff + mypy | Development |

## Configuration

All configuration is via environment variables (or `.env` file). See `.env.example` for the full list.

### Database

```bash
SEMSEARCH_DATABASE_URL=postgresql://user:pass@host:5432/dbname
SEMSEARCH_COLLECTION_NAME=semsearch_chunks
```

### Embedding Provider

Pick ONE provider. Nested fields use **double underscore** (`__`):

```bash
# OpenAI
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...

# OpenRouter
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-or-v1-...

# Ollama (local)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=ollama
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=nomic-embed-text
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434

# OpenAI-compatible (LM Studio, vLLM, etc.)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai_compatible
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=bge-small-en-v1.5
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:1234/v1
```

### Chunking

```bash
SEMSEARCH_CHUNK_SIZE=1000
SEMSEARCH_CHUNK_OVERLAP=200
SEMSEARCH_DEFAULT_K=5
```

## CLI Commands

### `semsearch init`

Create or migrate the database table.

```bash
semsearch init                  # Create table (idempotent)
semsearch init --recreate --yes # Drop and recreate (WARNING: deletes all data)
```

### `semsearch ingest <file>`

Ingest a single file. Re-ingesting an unchanged file reuses existing embeddings (zero API calls).

```bash
semsearch ingest docs/readme.md
semsearch ingest docs/readme.md --force  # Force re-embed everything
```

**Provider override** (per-command, does not modify `.env`):

```bash
semsearch ingest file.pdf \
  --provider openrouter \
  --provider-model openai/text-embedding-3-small \
  --provider-api-key sk-or-v1-...
```

### `semsearch ingest-dir <directory>`

Recursively ingest all supported files in a directory.

```bash
semsearch ingest-dir docs/
semsearch ingest-dir data/ --glob "**/*.pdf"       # Only PDFs
semsearch ingest-dir data/ --exclude "*/draft/*"   # Skip drafts
semsearch ingest-dir data/ --prune                  # Delete orphaned chunks
semsearch ingest-dir data/ --prune --dry-run        # Preview prune
semsearch ingest-dir data/ --no-continue-on-error   # Abort on first failure
semsearch ingest-dir data/ --follow-symlinks        # Follow symlinks
semsearch ingest-dir data/ --force                  # Force re-embed all
```

### `semsearch search <query>`

Cosine similarity search.

```bash
semsearch search "how to deploy"
semsearch search "api docs" --k 10
semsearch search "pdf content" --filter '{"doc_type": "pdf"}'
semsearch search "old docs" --filter '{"source": {"$ilike": "docs/old/%"}}'
semsearch search "database setup" --rerank --k 5  # With reranking
```

**Filter operators:**

| Operator | Example |
|----------|---------|
| Exact match | `{"source": "docs/file.pdf"}` |
| Prefix | `{"source": {"$ilike": "docs/%"}}` |
| Type filter | `{"doc_type": "csv"}` |
| Numeric | `{"year": {"$gte": 2024}}` |
| AND | `{"$and": [{...}, {...}]}` |
| OR | `{"$or": [{...}, {...}]}` |

### `semsearch delete`

Delete chunks by filter.

```bash
semsearch delete --filter '{"source": "docs/old.md"}'
semsearch delete --filter '{"doc_type": "pdf"}'
semsearch delete --all --yes  # Delete everything
```

### `semsearch stats`

Show table statistics.

```bash
semsearch stats
```

Output:
```json
{
  "table": "semsearch_chunks",
  "embedding_provider": "openrouter",
  "embedding_dim": 4096,
  "chunk_count": 149,
  "source_count": 24,
  "sources_by_count": [["TASKS/meet.md", 5], ...]
}
```

### `semsearch reingest <file>`

Delete + re-ingest in one step. Forces re-embedding.

```bash
semsearch reingest docs/readme.md
```

### `semsearch version`

Show version.

```bash
semsearch version
```

## Supported File Types

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.txt` | TextLoader | Single document |
| `.md` | TextLoader | Single document |
| `.pdf` | PyMuPDFLoader | One chunk per page |
| `.csv` | CSVLoader | One chunk per row |
| `.json` | JSONLoader | `jq_schema=".[].content"` |

## Idempotent Ingest

Re-ingesting an unchanged file makes **zero embedding API calls**. The service computes a SHA-256 hash of each chunk and compares it with the stored hash:

| Case | Existing row? | Hash matches? | Action | API call? |
|------|---------------|---------------|--------|-----------|
| A | yes | yes | Reuse embedding | No |
| B | yes | no | Re-embed | Yes |
| C | no | n/a | Embed + insert | Yes |
| D | stale tail | n/a | Delete | No |

## Path Consistency

`source` is stored verbatim as `str(path)`. For consistent results:

- Always run `semsearch ingest` from the same working directory
- Use relative paths from project root: `semsearch ingest docs/file.md`
- Prune from the same directory used at ingest time

## Provider Details

| Provider | Class | API Key Required | Default Model |
|----------|-------|------------------|---------------|
| `openai` | OpenAIEmbeddings | Yes | `text-embedding-3-small` |
| `openrouter` | OpenAIEmbeddings | Yes | — |
| `openai_compatible` | OpenAIEmbeddings | No | — |
| `ollama` | OllamaEmbeddings | No | — |

## Reranker (Optional)

Reranking improves search precision by re-scoring results with a cross-encoder.

### Configuration

```bash
# OpenRouter (reuse existing API key)
SEMSEARCH_RERANKER__BASE_URL=https://openrouter.ai/api/v1/rerank
SEMSEARCH_RERANKER__MODEL=cohere/rerank-v3.5

# Jina (separate API key)
SEMSEARCH_RERANKER__BASE_URL=https://api.jina.ai/v1/rerank
SEMSEARCH_RERANKER__MODEL=jina-reranker-v2-base-multilingual
SEMSEARCH_RERANKER__API_KEY=jina_...
```

### Usage

```bash
# Search with reranking
semsearch search "database setup" --rerank --k 5

# Search without reranking (default)
semsearch search "database setup" --k 5
```

### How it works

1. Vector search retrieves `k * 4` candidates
2. Reranker re-scores candidates against the query
3. Top-k results returned with `rerank_score` in metadata

**OpenRouter routing** (optional):

```bash
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER=["deepinfra","together"]
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ALLOW_FALLBACKS=true
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_IGNORE=["bad-provider"]
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ONLY=["deepinfra"]
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_DATA_COLLECTION=deny
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_MAX_PRICE={"prompt":1}
```

## Output Format

All commands output JSON to stdout. Errors go to stderr.

```bash
# Capture output
semsearch search "query" --k 5 | jq '.results[0].content'

# Pipe errors
semsearch ingest bad.pdf 2>errors.log
```

## Database Setup

### PostgreSQL + pgvector

```sql
-- As superuser
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE semsearch_app LOGIN PASSWORD 'change_me';
CREATE DATABASE semsearch OWNER semsearch_app;
\c semsearch
CREATE EXTENSION IF NOT EXISTS vector;
GRANT USAGE ON SCHEMA public TO semsearch_app;
GRANT CREATE ON SCHEMA public TO semsearch_app;
```

### Table Setup

After PostgreSQL is running and the database is created, initialize the table:

```bash
# Create the table (idempotent — safe to run multiple times)
semsearch init

# Or recreate from scratch (deletes all data)
semsearch init --recreate --yes
```

This creates the `semsearch_chunks` table with:
- `langchain_id TEXT PRIMARY KEY` — deterministic IDs (`source::chunk_index`)
- `embedding vector(N)` — N = embedding dimension (e.g., 1536 for OpenAI)
- `content TEXT` — chunk text
- `langchain_metadata JSONB` — soft metadata (doc_type, page, row, ingested_at)
- `source TEXT NOT NULL` — file path
- `chunk_index INTEGER NOT NULL` — 0-based position within source
- `document_hash CHAR(64)` — SHA-256 of content (for idempotent re-ingest)
- `UNIQUE (source, chunk_index)` — prevents duplicate chunks

**Indexes created automatically:**
- HNSW cosine similarity index (for vectors ≤2000 dim)
- GIN index on metadata JSONB (for filter queries)
- Composite index on `(source, chunk_index)` (for re-ingest lookups)

**Verify the table:**

```bash
# Check stats
semsearch stats

# Or check directly in psql
psql -d semsearch -c "\d semsearch_chunks"
```

**Switching providers:**

If you change the embedding provider to one with a different dimension, recreate the table:

```bash
# Example: switch from OpenAI (1536) to Ollama nomic-embed-text (768)
# 1. Edit .env with new provider
# 2. Recreate table
semsearch init --recreate --yes
# 3. Re-ingest all files
semsearch ingest-dir docs/
```

### NixOS (local PostgreSQL)

```bash
# In nix develop shell
initdb -D .pgdata --auth=trust
pg_ctl -D .pgdata -l .pglog start
createdb semsearch
psql -d semsearch -c "CREATE EXTENSION vector;"
```

### Non-NixOS Linux (Ubuntu/Debian)

```bash
# Install PostgreSQL and pgvector
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Install pgvector (requires build from source)
sudo apt install -y postgresql-server-dev-16 build-essential git
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
cd ..

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create role and database
sudo -u postgres psql <<EOF
CREATE ROLE semsearch LOGIN PASSWORD 'change_me';
CREATE DATABASE semsearch OWNER semsearch;
\c semsearch
CREATE EXTENSION vector;
GRANT USAGE ON SCHEMA public TO semsearch;
GRANT CREATE ON SCHEMA public TO semsearch;
EOF

# Add to .env
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch
```

**Fedora/RHEL/CentOS:**

```bash
# Install PostgreSQL
sudo dnf install -y postgresql-server postgresql-devel
sudo postgresql-setup --initdb
sudo systemctl start postgresql

# Install pgvector
sudo dnf install -y git make gcc
# Or install from EPEL/RPMFusion if available
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
cd ..

# Create role and database (same as above)
sudo -u postgres psql <<EOF
CREATE ROLE semsearch LOGIN PASSWORD 'change_me';
CREATE DATABASE semsearch OWNER semsearch;
\c semsearch
CREATE EXTENSION vector;
GRANT USAGE ON SCHEMA public TO semsearch;
GRANT CREATE ON SCHEMA public TO semsearch;
EOF
```

### macOS

```bash
# Install PostgreSQL and pgvector via Homebrew
brew install postgresql@16 pgvector

# Start PostgreSQL
brew services start postgresql@16

# Add to PATH (if needed)
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Create role and database
createdb semsearch
psql -d semsearch <<EOF
CREATE EXTENSION vector;
EOF

# Add to .env
SEMSEARCH_DATABASE_URL=postgresql://localhost:5432/semsearch
```

**macOS with Postgres.app:**

```bash
# Install from https://postgresapp.com/
# pgvector can be installed via:
brew install pgvector
# Then create database:
psql -d semsearch -c "CREATE EXTENSION vector;"
```

### Docker (any platform)

```bash
# Run PostgreSQL with pgvector
docker run -d \
  --name semsearch-pg \
  -e POSTGRES_USER=semsearch \
  -e POSTGRES_PASSWORD=change_me \
  -e POSTGRES_DB=semsearch \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Create extension
docker exec -it semsearch-pg psql -U semsearch -d semsearch \
  -c "CREATE EXTENSION vector;"

# Add to .env
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch
```

## Development

```bash
# Run tests
nix develop --command bash -c "TEST_DATABASE_URL='...' pytest -v"

# Run specific test
nix develop --command bash -c "pytest tests/test_service_ingest.py -v"

# Check coverage
nix develop --command bash -c "pytest --cov=semsearch --cov-report=term-missing"
```

## Architecture

```
src/semsearch/
├── cli.py          # Typer CLI
├── config.py       # Pydantic Settings
├── embeddings.py   # Provider dispatch
├── errors.py       # Exception hierarchy
├── loaders.py      # File type dispatch
├── models.py       # Pydantic models
├── service.py      # Main orchestration
├── splitter.py     # Text chunking
└── store.py        # PostgreSQL + pgvector
```

## Custom Config Files

Use `--config` (`-c`) to specify a custom config file:

```bash
semsearch --config prod.env search "query"
semsearch -c staging.env ingest file.md
semsearch -c /path/to/config.env stats
```

If not specified, uses `.env` in current directory. Environment variables override config file values.
