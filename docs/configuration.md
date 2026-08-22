# Configuration

semsearch is configured via environment variables or a `.env` file.

## Environment Variables

All variables use the `SEMSEARCH_` prefix. Nested fields use **double underscore** (`__`).

### Database

```bash
# PostgreSQL connection string
SEMSEARCH_DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Table name (must match /^[a-z_][a-z0-9_]{0,62}$/)
SEMSEARCH_COLLECTION_NAME=semsearch_chunks
```

### Connection Pool (opt-in)

By default every operation opens a fresh psycopg connection. For long-running
`serve` mode you can enable an opt-in connection pool so requests reuse
c connections instead of paying connection setup per request:

```bash
# 0 disables pooling (default — pure per-call connections)
SEMSEARCH_POOL__MIN_SIZE=1

# Maximum pooled connections (default: 4)
SEMSEARCH_POOL__MAX_SIZE=4

# Seconds to wait for a free connection before failing (default: 5.0)
SEMSEARCH_POOL__TIMEOUT=5.0
```

Only connections the service opens itself are pooled; any caller-provided
`conn=` parameter keeps the caller-owned lifecycle (never closed, never
pooled). Requires `psycopg-pool` (declared as a core dependency).

### Embedding Provider

```bash
# Provider type
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai

# Model name
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small

# API key (required for openai, openrouter)
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...

# Base URL (required for openai_compatible, optional for ollama)
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434
```

### Chunking

```bash
# Chunk size in characters (default: 1000)
SEMSEARCH_CHUNK_SIZE=1000

# Overlap between chunks (default: 200)
SEMSEARCH_CHUNK_OVERLAP=200

# Default k for search (default: 5)
SEMSEARCH_DEFAULT_K=5
```

## Provider Configuration

### OpenAI

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

### OpenRouter

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-or-v1-...

# Optional routing
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER=["deepinfra","together"]
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ALLOW_FALLBACKS=true
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_IGNORE=["bad-provider"]
```

### Ollama

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=ollama
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=nomic-embed-text
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434
```

### OpenAI-compatible

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai_compatible
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=bge-small-en-v1.5
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:1234/v1
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=not-needed-but-required
```

## OpenRouter Routing

OpenRouter supports advanced routing options:

```bash
# Provider order (lowercase slugs)
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER=["deepinfra","together"]

# Allow fallbacks (default: null = OpenRouter default)
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ALLOW_FALLBACKS=true

# Ignore specific providers
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_IGNORE=["deepseek"]

# Only use specific providers
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ONLY=["deepinfra","azure"]

# Data collection policy
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_DATA_COLLECTION=deny

# Max price (USD per 1M tokens)
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_MAX_PRICE={"prompt":1}
```

## .env.example

```bash
# Database
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch
SEMSEARCH_COLLECTION_NAME=semsearch_chunks

# Provider (pick one)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...

# Chunking
SEMSEARCH_CHUNK_SIZE=1000
SEMSEARCH_CHUNK_OVERLAP=200
SEMSEARCH_DEFAULT_K=5
```

## Environment-specific Config

### Development

```bash
# Use local PostgreSQL
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch

# Use cheaper/faster model
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
```

### Production

```bash
# Use managed PostgreSQL
SEMSEARCH_DATABASE_URL=postgresql://user:pass@db.example.com:5432/semsearch

# Use higher quality model
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-large
```

### CI/Testing

```bash
# Use testcontainers (no external DB needed)
SEMSEARCH_DATABASE_URL=postgresql://test:test@localhost:5432/test

# Use mock embeddings (no API calls)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=test-key
```

## Validation

The configuration is validated at startup:

- `collection_name` must match `/^[a-z_][a-z0-9_]{0,62}$/`
- `provider.type` must be one of: `openai`, `ollama`, `openai_compatible`, `openrouter`
- API keys are validated for providers that require them

## Troubleshooting

### "provider 'openai' requires api_key"

Set the API key:

```bash
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

### "collection_name must match..."

Use only lowercase letters, numbers, and underscores:

```bash
SEMSEARCH_COLLECTION_NAME=semsearch_chunks  # ✅
SEMSEARCH_COLLECTION_NAME=My-Table          # ❌
```

### "Id column does not exist"

The table was created with a different schema. Recreate:

```bash
semsearch init --recreate --yes
```

## HNSW Index Tuning

HNSW index parameters control the trade-off between recall and latency.

### Configuration

```bash
# Index construction parameters (set at `semsearch init` time)
SEMSEARCH_HNSW__M=16
SEMSEARCH_HNSW__EF_CONSTRUCTION=200

# Query-time parameter (set as table default)
SEMSEARCH_HNSW__EF_SEARCH=80
```

### HNSW Fields

| Field | Default | Min | Max | Description |
| ------- | --------- | ----- | ----- | ------------- |
| `m` | 16 | 2 | 100 | Max connections per layer |
| `ef_construction` | 200 | 4 | 1000 | Build-time search width |
| `ef_search` | 80 | 10 | 1000 | Query-time search width |

### Tuning Guide

| Use Case | M | ef_construction | ef_search |
| ---------- | --- | ----------------- | ---------- |
| Fast, lower recall | 8 | 64 | 40 |
| Balanced (default) | 16 | 200 | 80 |
| High recall, slower | 32 | 300 | 150 |

### HNSW Troubleshooting

#### "ef_search not taking effect"

`ef_search` is set at the table level. Verify with:

```sql
SELECT reloptions FROM pg_class WHERE relname = 'semsearch_chunks';
```

#### "Index creation slow"

Higher `ef_construction` means slower builds but better recall. Reduce to 64 for faster builds.

## Timeout Configuration

Network timeouts prevent the server from hanging on slow or failed API calls.

```bash
# Embedding API timeout (default: 10s)
SEMSEARCH_TIMEOUT__EMBEDDING=10.0

# Database connection timeout (default: 10s)
SEMSEARCH_TIMEOUT__DB_CONNECT=10

# Database connection pool recycle time (default: 300s)
SEMSEARCH_TIMEOUT__DB_POOL_RECYCLE=300

# TCP keep-alive settings (detect dead connections)
SEMSEARCH_TIMEOUT__DB_KEEPALIVE_IDLE=60
SEMSEARCH_TIMEOUT__DB_KEEPALIVE_INTERVAL=10
SEMSEARCH_TIMEOUT__DB_KEEPALIVE_COUNT=5

# Reranker API timeout (default: 10s)
SEMSEARCH_RERANKER__TIMEOUT=10.0
```

### Timeout Fields

| Field | Default | Min | Max | Description |
| ------- | --------- | ----- | ----- | ------------- |
| `timeout.embedding` | 10.0 | 1.0 | 120.0 | Embedding API request timeout |
| `timeout.db_connect` | 10 | 1 | 60 | Database connection timeout |
| `timeout.db_pool_recycle` | 300 | 30 | 3600 | Connection pool recycle time |
| `timeout.db_keepalive_idle` | 60 | 10 | 600 | Seconds before sending keep-alive probes |
| `timeout.db_keepalive_interval` | 10 | 5 | 60 | Seconds between keep-alive probes |
| `timeout.db_keepalive_count` | 5 | 1 | 20 | Failed probes before connection considered dead |
| `reranker.timeout` | 10.0 | 1.0 | 120.0 | Reranker API request timeout |

### How Keep-Alive Prevents Stale Connections

```
Client                         Database Server
  │                                 │
  │── TCP connection ──────────────>│
  │                                 │
  │   ... idle for 60 seconds ...   │
  │                                 │
  │── Keep-alive probe #1 ────────>│  (tcp_keepalives_idle=60)
  │<─ ACK ─────────────────────────│
  │                                 │
  │   ... 10 seconds later ...      │
  │                                 │
  │── Keep-alive probe #2 ────────>│  (tcp_keepalives_interval=10)
  │<─ ACK ─────────────────────────│
  │                                 │
  │   Connection stays alive!       │
```

If the server doesn't respond to 5 probes, the connection is considered dead
and a new one is created automatically.

### When to Adjust

| Scenario | Recommendation |
| ---------- | ---------------- |
| Slow network / high latency | Increase `timeout.embedding` to 30-60s |
| Large documents / many chunks | Keep defaults (timeouts are per-request) |
| Remote database with high latency | Increase `timeout.db_connect` to 20-30s |
| Connection pool exhaustion | Decrease `timeout.db_pool_recycle` to 120s |
| Frequent stale connections | Decrease `timeout.db_keepalive_idle` to 30s |

## Reranker

Reranking improves search precision by re-scoring results with a cross-encoder model.

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

### Reranker Fields

| Field | Default | Description |
| ------- | --------- | ------------- |
| `base_url` | `https://openrouter.ai/api/v1/rerank` | API endpoint |
| `model` | `cohere/rerank-v3.5` | Model name |
| `api_key` | None (falls back to embedding provider key) | API key |
| `top_n` | `5` | Default top_n for reranking |
| `timeout` | `10.0` | Request timeout in seconds |

### Reranker Troubleshooting

### "Reranker not configured"

Set the reranker configuration:

```bash
SEMSEARCH_RERANKER__BASE_URL=https://openrouter.ai/api/v1/rerank
SEMSEARCH_RERANKER__MODEL=cohere/rerank-v3.5
```
