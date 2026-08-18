# Getting Started

This guide walks you through installing semsearch, setting up the database, and running your first search.

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- API key for an embedding provider (OpenAI, OpenRouter, etc.)

## Installation

### NixOS (recommended)

```bash
git clone <repo-url> && cd semantic-search
nix develop  # enters dev shell, installs everything
```

### Linux/macOS

```bash
git clone <repo-url> && cd semantic-search

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with your provider
pip install -e ".[openai]"    # For OpenAI/OpenRouter
# or
pip install -e ".[ollama]"   # For Ollama (local)
# or
pip install -e ".[all]"       # All providers
```

## Database Setup

### 1. Install PostgreSQL + pgvector

**Ubuntu/Debian:**
```bash
sudo apt install -y postgresql postgresql-contrib
# Install pgvector from source or package manager
```

**macOS:**
```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
```

**Docker (any platform):**
```bash
docker run -d --name semsearch-pg \
  -e POSTGRES_USER=semsearch \
  -e POSTGRES_PASSWORD=change_me \
  -e POSTGRES_DB=semsearch \
  -p 5432:5432 \
  pgvector/pgvector:pg16

docker exec -it semsearch-pg psql -U semsearch -d semsearch \
  -c "CREATE EXTENSION vector;"
```

### 2. Create Database

```sql
-- As superuser
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE semsearch LOGIN PASSWORD 'change_me';
CREATE DATABASE semsearch OWNER semsearch;
\c semsearch
CREATE EXTENSION vector;
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Database
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch

# Provider (pick one)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

### 4. Initialize Table

```bash
semsearch init
```

This creates the `semsearch_chunks` table with all required columns and indexes.

## First Search

### 1. Ingest Documents

```bash
# Single file
semsearch ingest docs/readme.md

# Entire directory
semsearch ingest-dir docs/
```

### 2. Search

```bash
semsearch search "how to configure" --k 5
```

### 3. View Stats

```bash
semsearch stats
```

## Example Session

```bash
$ semsearch init
{
  "status": "created",
  "table": "semsearch_chunks"
}

$ semsearch ingest-dir TASKS/
{
  "dir": "TASKS",
  "files_discovered": 24,
  "files_attempted": 24,
  "files_succeeded": 24,
  "aggregate": {
    "chunks_added": 149,
    "chunks_reused": 0
  },
  "elapsed_seconds": 157.4
}

$ semsearch search "database setup" --k 3
{
  "query": "database setup",
  "k": 3,
  "results": [
    {
      "content": "# TASK-007: Database Store...",
      "score": 0.537,
      "source": "TASKS/TASK-007-database-store.md",
      "doc_type": "text"
    }
  ]
}

$ semsearch stats
{
  "table": "semsearch_chunks",
  "embedding_provider": "openrouter",
  "embedding_dim": 4096,
  "chunk_count": 149,
  "source_count": 24
}
```

## Next Steps

- [Configuration](configuration.md) — All configuration options
- [CLI Reference](cli-reference.md) — All commands
- [Providers](providers.md) — Embedding provider details
