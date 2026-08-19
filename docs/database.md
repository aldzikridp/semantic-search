# Database

This document describes the database schema, setup, and maintenance.

## Schema

### Table: `semsearch_chunks`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `langchain_id` | `TEXT` | NOT NULL | Primary key. Format: `{source}::{chunk_index}` |
| `embedding` | `vector(N)` | NOT NULL | Embedding vector. N = provider dimension |
| `content` | `TEXT` | NOT NULL | Chunk text |
| `langchain_metadata` | `JSONB` | — | Soft metadata (doc_type, page, row, etc.) |
| `source` | `TEXT` | NOT NULL | File path |
| `chunk_index` | `INTEGER` | NOT NULL | 0-based position within source |
| `document_hash` | `CHAR(64)` | — | SHA-256 of content |

### Indexes

| Name | Type | Columns | Purpose |
|------|------|---------|---------|
| `semsearch_chunks_pkey` | PRIMARY KEY | `langchain_id` | Unique ID |
| `semsearch_chunks_diskann_idx` | DiskANN | `embedding` | Vector similarity (all dims, SBQ compressed) |
| `semsearch_chunks_hnsw_idx` | HNSW | `embedding` | Vector similarity fallback (≤2000 dim, no vectorscale) |
| `semsearch_chunks_metadata_gin_idx` | GIN | `langchain_metadata` | JSONB filter queries |
| `semsearch_chunks_source_chunk_idx` | B-tree | `source, chunk_index` | Re-ingest lookups |
| `semsearch_chunks_source_chunk_unique` | UNIQUE | `source, chunk_index` | Prevent duplicates |

**Note:** Only one vector index is created — DiskANN when pgvectorscale is installed, HNSW otherwise. The HNSW index is automatically upgraded to DiskANN when `init_schema()` detects vectorscale.

### Vector Index Strategy

The vector index type is auto-detected based on what's installed:

| pgvectorscale installed? | Vector dims | Index created |
|-------------------------|-------------|---------------|
| Yes | Any | DiskANN (SBQ compressed, disk-based) |
| No | ≤2000 | HNSW (in-memory) |
| No | >2000 | None (sequential scan) |

**DiskANN advantages over HNSW:**
- No dimension limit (HNSW is capped at 2000)
- Disk-based (works when index exceeds RAM)
- SBQ compression: 16-32x smaller index
- Tunable accuracy vs latency at query time

**DiskANN defaults:**
- `storage_layout = memory_optimized` (SBQ)
- `num_bits_per_dimension = 2` (auto-downgrades to 1 for >900 dims)
- `num_neighbors = 50`
- `search_list_size = 100`

Override via env vars:
```bash
SEMSEARCH_DISKANN__STORAGE_LAYOUT=memory_optimized
SEMSEARCH_DISKANN__NUM_BITS_PER_DIMENSION=2
SEMSEARCH_DISKANN__NUM_NEIGHBORS=50
SEMSEARCH_DISKANN__SEARCH_LIST_SIZE=100
```

### Metadata JSONB Structure

```json
{
  "doc_type": "text",
  "page": 5,
  "row": 3,
  "ingested_at": "2026-08-18T06:48:57.984588+00:00",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

| Field | Type | Description |
|-------|------|-------------|
| `doc_type` | `string` | File type: "text", "pdf", "csv", "json" |
| `page` | `int` | PDF page number (PDF only) |
| `row` | `int` | CSV row number (CSV only) |
| `ingested_at` | `string` | ISO timestamp of last ingest |
| `chunk_size` | `int` | Chunk size used |
| `chunk_overlap` | `int` | Overlap used |

## Setup

### Using `semsearch init`

```bash
# Create table (idempotent)
semsearch init

# Recreate from scratch
semsearch init --recreate --yes
```

### Manual Setup

```sql
-- Create table
CREATE TABLE IF NOT EXISTS semsearch_chunks (
    langchain_id TEXT PRIMARY KEY,
    embedding vector(4096),
    content TEXT NOT NULL,
    langchain_metadata JSONB,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    document_hash CHAR(64)
);

-- DiskANN index (preferred — requires pgvectorscale extension)
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE INDEX IF NOT EXISTS semsearch_chunks_diskann_idx
    ON semsearch_chunks USING diskann (embedding vector_cosine_ops)
    WITH (
        storage_layout = 'memory_optimized',
        num_bits_per_dimension = 2
    );

-- OR: HNSW index (fallback — only for vectors ≤2000 dim)
CREATE INDEX IF NOT EXISTS semsearch_chunks_hnsw_idx
    ON semsearch_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for JSONB filters
CREATE INDEX IF NOT EXISTS semsearch_chunks_metadata_gin_idx
    ON semsearch_chunks USING gin ((langchain_metadata::jsonb) jsonb_path_ops);

-- Composite index for re-ingest lookups
CREATE INDEX IF NOT EXISTS semsearch_chunks_source_chunk_idx
    ON semsearch_chunks (source, chunk_index);

-- Unique constraint
ALTER TABLE semsearch_chunks
    ADD CONSTRAINT semsearch_chunks_source_chunk_unique
    UNIQUE (source, chunk_index);
```

## Vector Dimensions

| Provider | Model | Dimensions |
|----------|-------|------------|
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |
| OpenRouter | qwen/qwen3-embedding-8b | 4096 |
| Ollama | nomic-embed-text | 768 |

**Note:** HNSW index only supports vectors ≤2000 dimensions. For larger vectors, install pgvectorscale for DiskANN index, or sequential scan is used (still accurate, just slower on large datasets).

## Switching Providers

When switching to a provider with different dimensions:

```bash
# 1. Update .env
vim .env

# 2. Recreate table
semsearch init --recreate --yes

# 3. Re-ingest everything
semsearch ingest-dir docs/
```

## Upgrading from HNSW to DiskANN

If you have an existing table with HNSW index and install pgvectorscale:

```bash
# Just run init — it auto-upgrades the index
semsearch init
```

No reingest needed. The data stays the same; only the index changes.

## Maintenance

### Check Table Stats

```bash
semsearch stats
```

Or directly:

```sql
SELECT COUNT(*) FROM semsearch_chunks;
SELECT COUNT(DISTINCT source) FROM semsearch_chunks;
SELECT source, COUNT(*) FROM semsearch_chunks GROUP BY source ORDER BY COUNT(*) DESC LIMIT 20;
```

### Check Index Type

```sql
SELECT c.relname AS index_name, am.amname AS index_type
FROM pg_class c
JOIN pg_index i ON c.oid = i.indexrelid
JOIN pg_am am ON c.relam = am.oid
WHERE i.indrelid = 'semsearch_chunks'::regclass
AND am.amname IN ('diskann', 'hnsw');
```

### Vacuum and Analyze

```sql
VACUUM ANALYZE semsearch_chunks;
```

### Reindex

```sql
REINDEX TABLE semsearch_chunks;
```

### Check Index Usage

```sql
SELECT indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE relname = 'semsearch_chunks';
```

## Troubleshooting

### "relation does not exist"

The table hasn't been created:

```bash
semsearch init
```

### "column cannot have more than 2000 dimensions for hnsw index"

The embedding dimension exceeds HNSW limit. Install pgvectorscale for DiskANN index, which has no dimension limit.

### "vectorscale does not exist"

The pgvectorscale extension is not installed. Install it:

```sql
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
```

Or fall back to HNSW with a model that produces ≤2000 dimensions.

### "Id column does not exist"

Table schema mismatch. Recreate:

```bash
semsearch init --recreate --yes
```

### "vector dimension mismatch"

Provider produces different dimension than existing table:

```bash
# Recreate table with new dimension
semsearch init --recreate --yes
semsearch ingest-dir docs/
```

## Connection Strings

### Standard PostgreSQL

```
postgresql://user:password@host:5432/dbname
```

### With SSL

```
postgresql://user:password@host:5432/dbname?sslmode=require
```

### Unix Socket

```
postgresql://user:password@/dbname?host=/var/run/postgresql
```

### PgBouncer

```
postgresql://user:password@host:6432/dbname?prepared_statements=false
```
