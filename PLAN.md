# Performance Optimization Plan — Phase 1

> Based on `semantic-search-performance-audit.md`. Implements the "Quick Wins"
> phase plus an HTTP server for AI agent integration. Includes before/after
> benchmarks to validate every change.

---

## Summary of Changes

| # | Change | Source Finding | Files Modified | Estimated Effort |
|---|--------|---------------|----------------|------------------|
| 1 | Cache `_get_vector_size()` + DB read in `stats()` | Audit Finding 1 | `service.py` | 15 min |
| 2 | Expose HNSW `ef_search` / `ef_construction` in config, tune defaults | Audit Finding 3 | `config.py`, `store.py`, `service.py` | 30 min |
| 3 | Persistent `httpx.Client` in `Reranker` with retry | Audit Finding 7 | `reranker.py` | 30 min |
| 4 | Add `semsearch serve` — FastAPI HTTP server | Audit Finding 2 (replaces REPL) | `server.py` (new), `cli.py`, `config.py`, `pyproject.toml` | 2–3 hrs |
| 5 | Benchmark harness | New | `scripts/bench_search.py` | 30 min |

**Total estimated effort: ~4 hours**

---

## Change 1: Cache `_get_vector_size()` + DB Read in `stats()`

### Problem

`stats()` calls `_get_vector_size()` on every invocation, which embeds the
literal string `"dimension probe"` via a real API call (~150–800ms + API cost).
`init_schema()` also calls it. For a metadata command, this is wasteful.

### Solution

1. Cache the vector dimension in `self._cached_vector_size` after first probe.
2. Add `_get_vector_size_from_db()` that reads the dimension directly from
   `pg_attribute` — no API call, no network round-trip.
3. In `stats()`, prefer the DB read; fall back to cached probe only when the
   table doesn't exist yet.

### Files to Modify

**`src/semsearch/service.py`:**

- Add `self._cached_vector_size: int | None = None` in `__init__`.
- Refactor `_get_vector_size()` to check cache first:

```python
def _get_vector_size(self, *, force_probe: bool = False) -> int:
    if not force_probe and self._cached_vector_size is not None:
        return self._cached_vector_size
    size = len(self.embedder.embed_query("dimension probe"))
    self._cached_vector_size = size
    return size
```

- Add `_get_vector_size_from_db()`:

```python
def _get_vector_size_from_db(self) -> int | None:
    table = self.settings.collection_name
    conn = self._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (regexp_match(format_type(atttypid, atttypmod), "
                "'\\((\\d+)\\)'))[1]::int "
                "FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()
```

- Update `stats()` to use DB read first:

```python
"embedding_dim": self._get_vector_size_from_db() or self._get_vector_size(),
```

### Validation

- `semsearch stats` should return instantly (no network call) when the table
  already exists.
- `semsearch init` should only probe once (first time), then use cache.

---

## Change 2: Expose HNSW Tuning in Config

### Problem

pgvector's HNSW defaults to `ef_search = 40` at query time and
`ef_construction = 64` at index build time. These are never tuned and there's
no way to configure them. The pgvector docs recommend:
- `ef_construction = 200` for high-recall indexes
- `ef_search = 100+` for good recall at query time

### Solution

1. Add `HnswConfig` model in `config.py` with `m`, `ef_construction`, `ef_search`.
2. Use it in `store.py` when creating the HNSW index.
3. Set `hnsw.ef_search` per-query in `service.py` search path.
4. Set default `ef_construction = 200` (up from 64) and `ef_search = 80`
   (up from 40).

### Files to Modify

**`src/semsearch/config.py`:**

Add new config model:

```python
class HnswConfig(BaseModel):
    """HNSW index tuning parameters.

    Env var mapping:
        SEMSEARCH_HNSW__M=16
        SEMSEARCH_HNSW__EF_CONSTRUCTION=200
        SEMSEARCH_HNSW__EF_SEARCH=80
    """
    m: int = Field(default=16, ge=2, le=100)
    ef_construction: int = Field(default=200, ge=4, le=1000)
    ef_search: int = Field(default=80, ge=10, le=1000)
```

Add to `Settings`:

```python
hnsw: HnswConfig = Field(default_factory=HnswConfig)
```

**`src/semsearch/store.py`:**

Update the HNSW index creation in `init_schema()` to use config values:

```python
hnsw = settings.hnsw
cur.execute(
    f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
    f"WITH (m = {hnsw.m}, ef_construction = {hnsw.ef_construction})"
)
```

**`src/semsearch/service.py`:**

Update `search()` to set `ef_search` before querying. Since we use
`PGVectorStore.similarity_search_with_score()` (which manages its own
connection), the cleanest approach is to use the table-level default:

```python
# In init_schema(), after creating the index:
cur.execute(f"ALTER TABLE {table} SET (hnsw.ef_search = {hnsw.ef_search})")
```

This sets the default for all queries on the table without needing to inject
`SET` into each search call. Users can override per-session via `SET
hnsw.ef_search = 200;` in psql.

### Validation

- `semsearch init` should create index with new `ef_construction`.
- `semsearch search "test"` should use the configured `ef_search`.
- Verify with `SHOW hnsw.ef_search;` in psql after a search.

---

## Change 3: Persistent `httpx.Client` in Reranker with Retry

### Problem

`Reranker.rerank()` creates a new `httpx.post()` request each call — no
connection reuse, no retry. Each call pays TCP+TLS handshake overhead (~150ms).

### Solution

1. Create a persistent `httpx.Client` in `Reranker.__init__()` with connection
   pooling and timeouts.
2. Add exponential backoff retry (3 attempts) for 429 rate limits.
3. Use `self._client.post()` instead of `httpx.post()`.

### Files to Modify

**`src/semsearch/reranker.py`:**

```python
class Reranker:
    def __init__(self, config, api_key):
        self.base_url = config.base_url
        self.model = config.model
        self.api_key = api_key
        self.default_top_n = config.top_n

        # Persistent client — connection pool survives across calls
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=2.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def rerank(self, query, documents, top_n=None):
        if not documents:
            return []

        n = top_n or self.default_top_n
        texts = [doc.page_content for doc in documents]
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": n,
        }

        # Retry with exponential backoff for 429s
        for attempt in range(3):
            try:
                response = self._client.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    import time
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise SearchError(f"Rerank failed: {e}") from e
            except httpx.HTTPError as e:
                raise SearchError(f"Rerank failed: {e}") from e

        # ... rest unchanged (map indices back to documents) ...
```

### Validation

- First `semsearch search "test" --rerank` pays TCP+TLS handshake.
- Second call reuses the connection (faster).
- Rate-limited requests retry automatically.

---

## Change 4: FastAPI HTTP Server (`semsearch serve`)

### Problem

Every CLI command spins up a fresh `PGEngine` + embedder + connections
(~500ms–1.5s cold start). For AI agents making repeated tool calls, this is
wasteful. The audit proposed a `repl` command, but that only helps humans
typing interactively — not AI agents doing tool calling.

### Solution

Add `semsearch serve --host 0.0.0.0 --port 8383` that starts a FastAPI HTTP
server. The `SemanticSearchService` instance lives for the lifetime of the
server — zero cold-start on every request.

### Endpoints

| Method | Path | Body | Response | Description |
|--------|------|------|----------|-------------|
| `POST` | `/search` | `{"query": "...", "k": 5, "filter": {}, "rerank": false}` | `{"results": [...]}` | Similarity search |
| `POST` | `/ingest` | `{"path": "/path/to/file", "force": false}` | `{"source": "...", ...}` | Ingest single file |
| `POST` | `/ingest-dir` | `{"dir_path": "/path", "glob": "**/*", "prune": false, ...}` | `{"dir": "...", ...}` | Batch ingest directory |
| `DELETE` | `/delete` | `{"filter": {"source": "..."}, "all": false}` | `{"deleted_count": N, ...}` | Delete by filter |
| `GET` | `/stats` | — | `{"table": "...", "chunk_count": N, ...}` | Table statistics |
| `GET` | `/health` | — | `{"status": "ok"}` | Health check |

### Files to Create / Modify

**New file: `src/semsearch/server.py`**

```python
"""FastAPI HTTP server for semsearch.

Usage:
    semsearch serve --host 0.0.0.0 --port 8383
    semsearch serve  # defaults to 0.0.0.0:8383
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semsearch.config import Settings, get_settings
from semsearch.errors import SemSearchError
from semsearch.service import SemanticSearchService


# --- Request/Response models ---

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter: dict[str, Any] | None = None
    rerank: bool = False


class IngestRequest(BaseModel):
    path: str
    force: bool = False


class IngestDirRequest(BaseModel):
    dir_path: str
    glob: str = "**/*"
    exclude: list[str] | None = None
    prune: bool = False
    prune_dry_run: bool = False
    continue_on_error: bool = True
    follow_symlinks: bool = False
    force: bool = False


class DeleteRequest(BaseModel):
    filter: dict[str, Any] | None = None
    all: bool = False


# --- App factory ---

def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app with a shared SemanticSearchService."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: create service, keep it warm
        svc = SemanticSearchService.from_settings(settings)
        app.state.service = svc
        yield
        # Shutdown: close engine
        svc.close()

    app = FastAPI(
        title="semsearch",
        description="Semantic search HTTP API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _get_svc(request) -> SemanticSearchService:
        return request.app.state.service

    # --- Routes ---

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/search")
    async def search(req: SearchRequest, request):
        svc = _get_svc(request)
        try:
            results = svc.search(
                query=req.query,
                k=req.k,
                filter=req.filter,
                rerank=req.rerank,
            )
            return {
                "query": req.query,
                "k": req.k,
                "filter": req.filter,
                "reranked": req.rerank,
                "results": [r.model_dump() for r in results],
            }
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest")
    async def ingest(req: IngestRequest, request):
        svc = _get_svc(request)
        try:
            result = svc.ingest(Path(req.path), reembed_unchanged=req.force)
            return result.model_dump()
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest-dir")
    async def ingest_dir(req: IngestDirRequest, request):
        svc = _get_svc(request)
        try:
            result = svc.ingest_dir(
                Path(req.dir_path),
                glob=req.glob,
                exclude=req.exclude,
                reembed_unchanged=req.force,
                continue_on_error=req.continue_on_error,
                follow_symlinks=req.follow_symlinks,
                prune=req.prune,
                prune_dry_run=req.prune_dry_run,
            )
            return result.model_dump()
        except (ValueError, SemSearchError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.delete("/delete")
    async def delete(req: DeleteRequest, request):
        svc = _get_svc(request)
        if not req.all and req.filter is None:
            raise HTTPException(
                status_code=422,
                detail="Provide filter or set all=true",
            )
        filter_dict = {} if req.all else req.filter
        try:
            result = svc.delete(filter_dict)
            return result.model_dump()
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def stats(request):
        svc = _get_svc(request)
        try:
            return svc.stats()
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
```

**Modified: `src/semsearch/cli.py`**

Add `serve` command:

```python
@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8383, help="Bind port"),
) -> None:
    """Start the HTTP server (keeps service warm between requests)."""
    import uvicorn
    from semsearch.server import create_app

    settings = get_settings(_config_path)
    application = create_app(settings)
    uvicorn.run(application, host=host, port=port)
```

**Modified: `pyproject.toml`**

Add `fastapi` and `uvicorn` to dependencies:

```toml
dependencies = [
    # ... existing deps ...
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
]
```

### Server Lifecycle

```
semsearch serve --port 8383
  └── uvicorn starts
       └── lifespan: SemanticSearchService.from_settings(settings)
            ├── PGEngine (connection pool)
            ├── OpenAIEmbeddings (HTTP client)
            └── Reranker (if configured)
       └── Requests:
            ├── POST /search  → svc.search()  (no cold start)
            ├── POST /ingest  → svc.ingest()
            ├── GET /stats    → svc.stats()
            └── ...
       └── Shutdown: svc.close()
```

### AI Agent Integration Example

```python
import httpx

# Agent makes repeated calls — server stays warm
client = httpx.Client(base_url="http://localhost:8383")

# Search
results = client.post("/search", json={
    "query": "how to deploy",
    "k": 5,
}).json()

# Ingest
client.post("/ingest", json={"path": "/docs/new.md"})

# Stats
stats = client.get("/stats").json()
```

### Validation

- `semsearch serve` starts and binds to port 8383.
- `curl http://localhost:8383/health` returns `{"status": "ok"}`.
- `curl -X POST http://localhost:8383/search -d '{"query":"test","k":3}'` returns results.
- OpenAPI docs at `http://localhost:8383/docs`.
- Second search request is noticeably faster than first (warm embedder).

---

## Change 5: Benchmark Harness

### Problem

No way to measure before/after performance. Need a script that runs a set of
queries N times and reports latency percentiles.

### Solution

Create `scripts/bench_search.py` that:
1. Reads queries from a file (one per line) or generates them.
2. Runs each query N times through `SemanticSearchService.search()`.
3. Reports p50, p95, p99, mean latency in JSON.
4. Can be run before and after each change to measure impact.

### File to Create

**`scripts/bench_search.py`:**

```python
"""Benchmark the search path.

Usage:
    # Run against live DB with real embeddings
    python scripts/bench_search.py --queries queries.txt --n 50

    # Run with mock embeddings (no API calls, measures pure overhead)
    python scripts/bench_search.py --mock --n 100

    # Output JSON for diffing
    python scripts/bench_search.py --queries queries.txt --n 50 --json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from semsearch.config import get_settings
from semsearch.service import SemanticSearchService


def main():
    p = argparse.ArgumentParser(description="Benchmark semsearch search path")
    p.add_argument("--queries", type=Path, help="File with queries (one per line)")
    p.add_argument("--n", type=int, default=50, help="Repetitions per query")
    p.add_argument("--k", type=int, default=5, help="Top-k results")
    p.add_argument("--rerank", action="store_true", help="Enable reranking")
    p.add_argument("--mock", action="store_true", help="Use mock embeddings")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    args = p.parse_args()

    # Load queries
    if args.queries:
        queries = [l.strip() for l in args.queries.read_text().splitlines() if l.strip()]
    else:
        queries = ["test query", "how to deploy", "database setup", "search optimization"]

    if not queries:
        print("No queries found", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()

    if args.mock:
        from tests.conftest import MockEmbeddings
        from semsearch.store import build_engine, init_schema
        embedder = MockEmbeddings(dim=128)
        engine = build_engine(settings)
        init_schema(settings, engine, embedder.dim)
        svc = SemanticSearchService(settings, engine, embedder)
    else:
        svc = SemanticSearchService.from_settings(settings)

    with svc:
        # Warmup
        for _ in range(args.warmup):
            svc.search(queries[0], k=args.k)

        # Benchmark
        all_latencies: list[float] = []
        per_query: dict[str, list[float]] = {q: [] for q in queries}

        for _ in range(args.n):
            for q in queries:
                t0 = time.perf_counter()
                svc.search(q, k=args.k, rerank=args.rerank)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                all_latencies.append(elapsed_ms)
                per_query[q].append(elapsed_ms)

        # Report
        sorted_lat = sorted(all_latencies)
        report = {
            "n_queries": len(queries),
            "n_repetitions": args.n,
            "k": args.k,
            "rerank": args.rerank,
            "mock": args.mock,
            "total_requests": len(all_latencies),
            "p50_ms": round(sorted_lat[int(len(sorted_lat) * 0.50)], 2),
            "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 2),
            "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 2),
            "mean_ms": round(statistics.mean(all_latencies), 2),
            "min_ms": round(min(all_latencies), 2),
            "max_ms": round(max(all_latencies), 2),
        }

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Benchmark: {report['total_requests']} requests")
            print(f"  p50:  {report['p50_ms']}ms")
            print(f"  p95:  {report['p95_ms']}ms")
            print(f"  p99:  {report['p99_ms']}ms")
            print(f"  mean: {report['mean_ms']}ms")
            print(f"  min:  {report['min_ms']}ms")
            print(f"  max:  {report['max_ms']}ms")


if __name__ == "__main__":
    main()
```

### Benchmark Workflow

```bash
# 1. Record baseline (BEFORE any changes)
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json" > bench_before.json

# 2. Make changes (apply plan)

# 3. Record after
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json" > bench_after.json

# 4. Diff
python -c "
import json
before = json.load(open('bench_before.json'))
after = json.load(open('bench_after.json'))
for k in ['p50_ms', 'p95_ms', 'p99_ms', 'mean_ms']:
    b, a = before[k], after[k]
    pct = ((b - a) / b) * 100
    print(f'{k}: {b}ms → {a}ms ({pct:+.1f}%)')
"
```

---

## Implementation Order

Execute changes in this order to minimize risk and maximize testability:

```
Step 1: Create benchmark harness (Change 5)
        ↓
Step 2: Run baseline benchmarks, record results
        ↓
Step 3: Apply Change 1 (vector size caching)
        → Run tests: pytest tests/test_service_search.py -v
        → Run benchmark, compare to baseline
        ↓
Step 4: Apply Change 3 (Reranker httpx.Client)
        → Run tests: pytest tests/test_service_search.py -v
        → Run benchmark, compare
        ↓
Step 5: Apply Change 2 (HNSW tuning)
        → Run tests: pytest tests/test_service_search.py -v
        → Run benchmark, compare
        ↓
Step 6: Apply Change 4 (FastAPI server)
        → Run tests: pytest -v
        → Manual: start server, curl endpoints
        ↓
Step 7: Final benchmark comparison
        → Generate before/after report
```

---

## Testing Strategy

### Existing Tests (must still pass)

All 71 existing tests must pass after each change. Run:

```bash
nix develop --command bash -c "pytest -v"
```

### New Tests to Add

**`tests/test_server.py`** — HTTP endpoint tests using `httpx.AsyncClient`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from semsearch.server import create_app

@pytest.fixture
def app(settings_with_mock):
    return create_app(settings_with_mock)

@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

async def test_health(client):
    r = await client.get("/health")
    assert r.json() == {"status": "ok"}

async def test_search(client):
    r = await client.post("/search", json={"query": "test", "k": 3})
    assert r.status_code == 200
    assert "results" in r.json()

async def test_stats(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    assert "chunk_count" in r.json()
```

**`tests/test_vector_size_cache.py`** — Unit tests for caching:

```python
def test_vector_size_cached(service):
    """_get_vector_size returns cached value on second call."""
    size1 = service._get_vector_size()
    size2 = service._get_vector_size()
    assert size1 == size2
    assert service._cached_vector_size == size1

def test_vector_size_from_db(service):
    """_get_vector_size_from_db reads dimension from pg_attribute."""
    service.init_schema()
    size = service._get_vector_size_from_db()
    assert size == 128  # MockEmbeddings dim
```

---

## Dependencies to Add

**`pyproject.toml`** — new runtime deps:

```toml
"fastapi>=0.115.0",
"uvicorn[standard]>=0.30.0",
```

**No new test deps** — use `httpx.AsyncClient` with `ASGITransport` for
testing (built into httpx, no extra package).

---

## Files Summary

| File | Action | Changes |
|------|--------|---------|
| `src/semsearch/service.py` | Modify | Cache `_get_vector_size()`, add `_get_vector_size_from_db()`, update `stats()` |
| `src/semsearch/config.py` | Modify | Add `HnswConfig`, add `hnsw` to `Settings` |
| `src/semsearch/store.py` | Modify | Use `HnswConfig` in `init_schema()`, set table-level `ef_search` |
| `src/semsearch/reranker.py` | Modify | Persistent `httpx.Client`, retry logic |
| `src/semsearch/server.py` | **Create** | FastAPI app with all endpoints |
| `src/semsearch/cli.py` | Modify | Add `serve` command |
| `pyproject.toml` | Modify | Add `fastapi`, `uvicorn` deps |
| `scripts/bench_search.py` | **Create** | Benchmark harness |
| `tests/test_server.py` | **Create** | HTTP endpoint tests |
| `tests/test_vector_size_cache.py` | **Create** | Caching unit tests |

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Vector size caching | Low | Cache is per-instance, cleared on new service. DB read is a simple SELECT. |
| HNSW tuning | Low | Table-level SET is session-scoped by default. Old indexes still work. Higher `ef_construction` only affects new index builds. |
| Reranker httpx.Client | Low | Drop-in replacement. Same API, just persistent connections. Retry only for 429s. |
| FastAPI server | Medium | New module, new deps. Mitigated by: keeping CLI path unchanged, using lifespan for clean startup/shutdown, comprehensive endpoint tests. |

---

## What's NOT in Phase 1

These are deferred to later phases (per audit roadmap):

- **Connection pooling** (`psycopg_pool.ConnectionPool`) — Phase 2
- **Raw SQL for search** (replacing `PGVectorStore.similarity_search_with_score`) — Phase 2
- **Embedding cache** (LRU on query vectors) — Phase 2
- **Result cache** (LRU on search results) — Phase 2
- **Async conversion** (`async def search_async`) — Phase 3
- **Batch embedding** (`search_by_vector()`) — Phase 3
