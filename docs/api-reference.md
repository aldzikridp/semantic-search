# HTTP API Reference

> API documentation for `semsearch serve` — the read-only HTTP server mode.

## Overview

The `semsearch serve` command starts a FastAPI HTTP server that keeps the
`SemanticSearchService` warm between requests, eliminating cold-start overhead
for AI agent tool calling.

**Startup warmup:** at startup the server pre-builds lazily-initialized local
resources (PGVectorStore, one DB round-trip, reranker client construction) so
the first request skips local lazy-init. Warmup **never contacts the embedding
or reranker APIs** — startup stays available even when providers are down
(fail-open: failures are logged and deferred to the first real request).

```bash
semsearch serve --host 0.0.0.0 --port 8383 --log-level info
```

**Interactive API docs:** `http://localhost:8383/docs` (Swagger UI)

## Base URL

```
http://localhost:8383
```

### Unix Domain Socket Mode

Start the server with `--uds` to listen on a Unix socket instead of TCP:

```bash
semsearch serve --uds ./semsearch.sock
```

All endpoints work identically. Call them with curl's `--unix-socket` flag
(path part is ignored, use any hostname):

```bash
curl --unix-socket ./semsearch.sock http://localhost/health
curl --unix-socket ./semsearch.sock -X POST http://localhost/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to deploy", "k": 3}'
```

## Endpoints

### Health Check

Check if the server is running.

```
GET /health
```

**Response:**

```json
{
  "status": "ok"
}
```

**Example:**

```bash
curl http://localhost:8383/health
```

---

### Search

Perform semantic similarity search over ingested documents.

```
POST /search
```

**Request Body:**

```json
{
  "query": "how to deploy",
  "k": 5,
  "filter": null,
  "rerank": false
}
```

| Field | Type | Default | Description |
| ------- | ------ | --------- | ------------- |
| `query` | string | *required* | Search query text |
| `k` | integer | `5` | Number of results to return (1–50) |
| `filter` | object | `null` | Filter criteria (see Filter Syntax) |
| `rerank` | boolean | `false` | Enable reranking for better precision |

**Response:**

```json
{
  "query": "how to deploy",
  "k": 5,
  "filter": null,
  "reranked": false,
  "results": [
    {
      "id": "docs/deploy.md::0",
      "content": "# Deployment Guide\n\nTo deploy...",
      "score": 0.85,
      "source": "docs/deploy.md",
      "chunk_index": 0,
      "page": null,
      "row": null,
      "doc_type": "text",
      "metadata": {
        "doc_type": "text",
        "ingested_at": "2026-08-20T05:32:56.257931+00:00",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "source": "docs/deploy.md",
        "chunk_index": 0
      }
    }
  ]
}
```

**Result Fields:**

| Field | Type | Description |
| ------- | ------ | ------------- |
| `id` | string | Chunk ID (`{source}::{chunk_index}`) |
| `content` | string | Chunk text content |
| `score` | float | Cosine similarity score (0–1, higher = better) |
| `source` | string | File path of the source document |
| `chunk_index` | integer | 0-based chunk position within source |
| `page` | integer | PDF page number (null for non-PDF) |
| `row` | integer | CSV row number (null for non-CSV) |
| `doc_type` | string | Document type: text, pdf, csv, json |
| `metadata` | object | Full metadata including `rerank_score` if reranked |

**Examples:**

```bash
# Basic search
curl -X POST http://localhost:8383/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to deploy", "k": 3}'

# With filter
curl -X POST http://localhost:8383/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database", "k": 5, "filter": {"doc_type": "pdf"}}'

# With reranking
curl -X POST http://localhost:8383/search \
  -H "Content-Type: application/json" \
  -d '{"query": "security best practices", "k": 5, "rerank": true}'
```

**Error Responses:**

| Status | Cause | Example |
| -------- | ------- | --------- |
| 422 | Invalid `k` value | `{"detail": "k must be between 1 and 50, got 0"}` |
| 422 | Missing `query` | `{"detail": "... validation error ..."}` |
| 500 | Reranker not configured | `{"detail": "Reranker not configured. Set SEMSEARCH_RERANKER__BASE_URL..."}` |
| 500 | Search failed | `{"detail": "Search failed: ..."}` |

---

### Stats

Get statistics about the ingested data.

```
GET /stats
```

**Response:**

```json
{
  "table": "semsearch_chunks",
  "embedding_provider": "openrouter",
  "embedding_dim": 4096,
  "chunk_count": 149,
  "source_count": 24,
  "sources_by_count": [
    ["docs/deploy.md", 15],
    ["docs/api.md", 12],
    ["docs/config.md", 8]
  ]
}
```

**Response Fields:**

| Field | Type | Description |
| ------- | ------ | ------------- |
| `table` | string | Database table name |
| `embedding_provider` | string | Active embedding provider |
| `embedding_dim` | integer | Vector dimensions |
| `chunk_count` | integer | Total number of chunks |
| `source_count` | integer | Number of unique source files |
| `sources_by_count` | array | Top 20 sources by chunk count |

**Example:**

```bash
curl http://localhost:8383/stats
```

---

## Filter Syntax

Filters use the same syntax as `PGVectorStore` filters. They are applied
to the `langchain_metadata` JSONB column.

### Simple Filters

```json
{"doc_type": "pdf"}
```

```json
{"source": "docs/deploy.md"}
```

### Comparison Operators

| Operator | Example | Description |
| ---------- | --------- | ------------- |
| `$eq` | `{"doc_type": {"$eq": "pdf"}}` | Equal (default) |
| `$ne` | `{"doc_type": {"$ne": "json"}}` | Not equal |
| `$gt` | `{"page": {"$gt": 5}}` | Greater than |
| `$gte` | `{"page": {"$gte": 1}}` | Greater than or equal |
| `$lt` | `{"page": {"$lt": 10}}` | Less than |
| `$lte` | `{"page": {"$lte": 5}}` | Less than or equal |

### Pattern Matching

| Operator | Example | Description |
|----------|---------|-------------|
| `$like` | `{"source": {"$like": "docs/%"}}` | SQL LIKE pattern |
| `$ilike` | `{"source": {"$ilike": "%DEPLOY%"}}` | Case-insensitive LIKE |

### List Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$in` | `{"doc_type": {"$in": ["pdf", "csv"]}}` | Match any value |
| `$nin` | `{"doc_type": {"$nin": ["json"]}}` | Match none of values |

### Logical Operators

```json
{"$and": [
  {"doc_type": "pdf"},
  {"page": {"$gte": 5}}
]}
```

```json
{"$or": [
  {"doc_type": "pdf"},
  {"doc_type": "csv"}
]}
```

### Existence

```json
{"page": {"$exists": true}}
```

---

## Reranking

When `rerank: true`, the search process:

1. Fetches `k * 4` candidates from vector search
2. Sends candidates to the reranker API
3. Returns top-k results with `rerank_score` in metadata

**Configuration:**

```bash
SEMSEARCH_RERANKER__BASE_URL=https://openrouter.ai/api/v1/rerank
SEMSEARCH_RERANKER__MODEL=cohere/rerank-v3.5
SEMSEARCH_RERANKER__API_KEY=sk-or-v1-...
```

**Response with reranking:**

```json
{
  "reranked": true,
  "results": [
    {
      "score": 0.85,
      "metadata": {
        "rerank_score": 0.92
      }
    }
  ]
}
```

---

## Error Handling

All errors return JSON with a `detail` field:

```json
{
  "detail": "Error message here"
}
```

| Status Code | Description |
| ------------- | ------------- |
| 200 | Success |
| 422 | Validation error (bad request body) |
| 500 | Server error (search failed, reranker not configured, etc.) |

---

## Python Client Example

```python
import httpx

client = httpx.Client(base_url="http://localhost:8383")

# Health check
health = client.get("/health").json()
print(health["status"])  # "ok"

# Search
results = client.post("/search", json={
    "query": "how to deploy",
    "k": 5,
    "rerank": True,
}).json()

for r in results["results"]:
    print(f"{r['score']:.3f} | {r['source']} | {r['content'][:80]}...")

# Stats
stats = client.get("/stats").json()
print(f"Total chunks: {stats['chunk_count']}")
print(f"Sources: {stats['source_count']}")

# Filtered search
results = client.post("/search", json={
    "query": "security",
    "k": 3,
    "filter": {"doc_type": "pdf"},
}).json()
```

---

## JavaScript/TypeScript Client Example

```typescript
interface SearchRequest {
  query: string;
  k?: number;
  filter?: Record<string, any>;
  rerank?: boolean;
}

interface SearchResult {
  id: string;
  content: string;
  score: number;
  source: string;
  chunk_index: number;
  metadata: Record<string, any>;
}

interface SearchResponse {
  query: string;
  k: number;
  filter: Record<string, any> | null;
  reranked: boolean;
  results: SearchResult[];
}

const BASE_URL = "http://localhost:8383";

async function search(req: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return response.json();
}

// Usage
const results = await search({
  query: "how to deploy",
  k: 5,
  rerank: true,
});

results.results.forEach((r) => {
  console.log(`${r.score.toFixed(3)} | ${r.source}`);
});
```

---

## Shell Script Examples

```bash
#!/bin/bash
BASE_URL="http://localhost:8383"

# Search and format results
search() {
  curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$1\", \"k\": ${2:-5}}" | \
    jq -r '.results[] | "\(.score)\t\(.source)\t\(.content[0:80])"'
}

# Usage
search "database setup" 3
search "deployment guide" 5
```

---

## Performance Notes

| Metric | Typical Value |
| -------- | --------------- |
| Cold start (first request) | ~2–5s (model loading) |
| Warm request (no rerank) | 50–200ms |
| Warm request (with rerank) | 100–500ms |
| Stats endpoint | 5–50ms |

**Timeouts:**

- Embedding API: 10s (configurable via `SEMSEARCH_TIMEOUT__EMBEDDING`)
- Database: 10s (configurable via `SEMSEARCH_TIMEOUT__DB_CONNECT`)
- Reranker: 10s (configurable via `SEMSEARCH_RERANKER__TIMEOUT`)

---

## See Also

- [CLI Reference](cli-reference.md) — Command-line interface
- [Configuration](configuration.md) — Environment variables
- [Getting Started](getting-started.md) — Installation and first search
