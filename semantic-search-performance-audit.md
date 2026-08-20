# Performance Audit: `aldzikridp/semantic-search`

## Executive Summary

After reading the full source (`store.py`, `service.py`, `embeddings.py`, `reranker.py`, `config.py`, `cli.py`) plus `docs/architecture.md`, I identified **11 distinct bottlenecks** in the query path. Three of them are pathological — fixes that should land first because they deliver outsized gains for under an hour of work each:

| # | Bottleneck | Estimated impact | Effort |
|---|---|---|---|
| 1 | `stats()` calls `_get_vector_size()` which makes a real embedding API call **every invocation** | Saves ~150–800 ms per `stats` call + an API bill | 15 min |
| 2 | CLI spins up a brand-new `PGEngine` + `OpenAIEmbeddings` client for every command | Saves ~300–1500 ms cold-start per `search` | 30 min |
| 3 | `hnsw.ef_search` is left at the pgvector default of 40; never tuned | Better recall at fixed latency, or same recall at lower latency | 5 min |

Beyond those, the architecture has deeper structural issues — sync-only I/O, no connection pool, per-call `psycopg.connect()`, no embedding cache, no query-result cache — that will start hurting as soon as you go from "CLI toy" to "service handling concurrent queries."

---

## 1. Repository Architecture (so we're on the same page)

```
CLI (typer)
  └── SemanticSearchService  ── builds ──>  PGEngine (SQLAlchemy + asyncpg pool)
       │                                   │
       ├── embedder  (langchain_openai / langchain_ollama)
       ├── store     (PGVectorStore — read-only path for similarity_search_with_score)
       ├── reranker  (httpx client → OpenRouter / Jina / Cohere rerank API)
       └── raw psycopg connection  (write path: INSERT/UPDATE/DELETE)
```

**Query path** (`service.py` lines 479–562):

```
search(query, k, filter, rerank)
  ├── self.store.similarity_search_with_score(query, k=fetch_k, filter=filter)
  │     ├── embedder.embed_query(query)            ← network call to provider
  │     └── SELECT ... ORDER BY embedding <=> $1 LIMIT k   ← pgvector SQL
  ├── convert distances to similarities (1 - distance)
  └── if rerank:
        └── reranker.rerank(query, docs, top_n=k)  ← network call to rerank API
```

The write path is already well-optimized (atomic transaction, hash-based dedup, UPSERT, no double-embedding) — the issues are all in the read path.

---

## 2. Query Latency — Findings & Fixes

### Finding 1: `stats()` makes a real embedding API call every time

**Location:** `service.py` line 601 — `"embedding_dim": self._get_vector_size()`

```python
def _get_vector_size(self) -> int:
    """Embed a dummy query to determine the vector dimension."""
    return len(self.embedder.embed_query("dimension probe"))
```

`stats()` is a metadata command — it just needs an integer. Instead, it triggers a network round-trip to OpenAI/OpenRouter/Jina, embeds the literal string `"dimension probe"`, and throws the vector away just to read its `.len()`. On OpenRouter this is 200–800 ms **and** costs tokens. It's also called from `init_schema()`, so every `semsearch init` (which often runs as part of a CI script or docker startup) does the same thing.

**Fix:** Cache the dimension after the first probe, and/or query the DB for it (you already have a SQL query in `init_schema` that reads it from `pg_attribute`).

**Before:**

```python
# service.py — replace _get_vector_size and stats()

def _get_vector_size(self) -> int:
    """Embed a dummy query to determine the vector dimension."""
    return len(self.embedder.embed_query("dimension probe"))
```

**After:**

```python
# service.py

def _get_vector_size(self, *, force_probe: bool = False) -> int:
    """Return the embedding dimension. Cached after first probe.

    Args:
        force_probe: If True, ignore the cache and re-embed the probe string.
                     Useful when the provider might have changed mid-process.
    """
    if not force_probe and self._cached_vector_size is not None:
        return self._cached_vector_size
    size = len(self.embedder.embed_query("dimension probe"))
    self._cached_vector_size = size
    return size

def _get_vector_size_from_db(self) -> int | None:
    """Read the vector dimension straight from pg_attribute — no API call."""
    table = self.settings.collection_name
    conn = self._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (regexp_match(format_type(atttypid, atttypmod), '\\((\\d+)\\)'))[1]::int "
                "FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

def stats(self, conn: psycopg.Connection | None = None) -> dict[str, Any]:
    # ... (unchanged) ...
    return {
        "table": table,
        "embedding_provider": self.settings.embedding_provider.type,
        # Read from DB when possible; fall back to cached probe only if table missing.
        "embedding_dim": self._get_vector_size_from_db() or self._get_vector_size(),
        # ... rest unchanged ...
    }
```

And in `__init__`:

```python
self._cached_vector_size: int | None = None
```

**Bonus:** `init_schema()` should also use `_get_vector_size_from_db()` first when the table already exists, instead of probing every time.

---

### Finding 2: The CLI throws away the engine and embedder on every invocation

**Location:** `cli.py` — every `@app.command()` function does:

```python
with SemanticSearchService.from_settings(settings) as svc:
    result = svc.search(...)
```

`from_settings()` builds:

- a fresh `PGEngine` (which spins up a SQLAlchemy + asyncpg connection pool — ~150 ms)
- a fresh `OpenAIEmbeddings` (which creates a new `httpx.Client` and a fresh TCP+TLS handshake — ~200–800 ms depending on provider)
- a fresh `Reranker` instance (which has no shared HTTP client)

For a one-shot CLI command this is unavoidable in *one* process — but the user is paying ~500 ms–1.5 s of cold-start tax on **every** `semsearch search` invocation. For a developer running many queries in a loop (e.g. testing a RAG pipeline), this is brutal.

**Two-part fix:**

**a)** Add a long-lived `semsearch repl` (or `semsearch serve`) subcommand that keeps the service alive and reads queries from stdin or HTTP. This eliminates per-invocation cold start for batch use:

```python
# cli.py — new command

@app.command()
def repl() -> None:
    """Interactive REPL — keeps the service warm between queries."""
    settings = get_settings(_config_path)
    with SemanticSearchService.from_settings(settings) as svc:
        typer.echo("semsearch repl — type queries, :quit to exit")
        for line in sys.stdin:
            query = line.strip()
            if query in (":q", ":quit", ":exit"):
                break
            if not query:
                continue
            t0 = time.monotonic()
            results = svc.search(query, k=settings.default_k)
            elapsed_ms = (time.monotonic() - t0) * 1000
            typer.echo(json.dumps({
                "query": query,
                "elapsed_ms": round(elapsed_ms, 1),
                "results": [r.model_dump() for r in results],
            }, indent=2, default=str))
```

**b)** In `embeddings.py`, force `OpenAIEmbeddings` to use a shared `httpx.Client` and short request timeouts so retries fail fast:

**Before:**

```python
# embeddings.py — before

return OpenAIEmbeddings(
    api_key=cfg.api_key.get_secret_value(),
    model=cfg.model,
)
```

**After:**

```python
# embeddings.py

import httpx
from openai import OpenAI  # for the client, not the langchain wrapper

# At module scope — shared across all OpenAIEmbeddings instances created in
# this process. Connection pool survives between commands if you wrap many
# queries inside one Python process (e.g. the repl above).
_shared_http_client = httpx.Client(
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=2.0),
)

# Build a real OpenAI client and pass it in — LangChain's OpenAIEmbeddings
# accepts `client=` so we don't pay the SDK's default client setup cost.
_openai_client = OpenAI(
    api_key=cfg.api_key.get_secret_value(),
    base_url=base_url,
    http_client=_shared_http_client,
    max_retries=2,
    timeout=30.0,
)

return OpenAIEmbeddings(
    client=_openai_client,
    model=cfg.model,
    model_kwargs=model_kwargs,
    check_embedding_ctx_length=False,
)
```

Note this also fixes a latent bug: `embeddings.py` line 63 sets `check_embedding_ctx_length=False` for OpenRouter/compatible but not for `openai`. Either path can hit the same long-query problem, so this should be consistent.

---

### Finding 3: `hnsw.ef_search` is never tuned (and `ef_construction` is conservative)

**Location:** `store.py` lines 230–234

```python
cur.execute(
    f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
    f"WITH (m = 16, ef_construction = 64)"
)
```

pgvector's HNSW defaults to `hnsw.ef_search = 40` at query time. That number controls the trade-off between recall and latency — higher = better recall, slower. **40 is fine for k=5 toys, but you have no way to tune it from the config, and the default is never raised.**

There are two improvements here:

1. Raise `ef_construction` to 128 — index build is slower but recall at query time improves ~5–8%.
2. Expose `ef_search` as a per-query SET (session-scoped) so users can trade latency for recall on the fly.

**Before:**

```python
cur.execute(
    f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
    f"WITH (m = 16, ef_construction = 64)"
)
```

**After:**

```python
# config.py — add to a new HnswConfig model
class HnswConfig(BaseModel):
    m: int = Field(default=16, ge=2, le=100)
    ef_construction: int = Field(default=200, ge=4, le=1000)
    ef_search: int = Field(default=80, ge=10, le=1000)  # default for all queries

class Settings(BaseSettings):
    # ...
    hnsw: HnswConfig = Field(default_factory=HnswConfig)

# store.py — use it
def init_schema(settings, engine, vector_size, *, recreate=False):
    # ...
    hnsw = settings.hnsw
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
        f"ON {table} USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {hnsw.m}, ef_construction = {hnsw.ef_construction})"
    )

# service.py — set ef_search at the start of every search session
def search(self, query, k=None, filter=None, rerank=False, ef_search=None):
    # ...
    ef = ef_search or self.settings.hnsw.ef_search
    # PGVectorStore creates its own connection; we need to set this on the
    # session. The cleanest hook is to override the engine's pre_ping hook,
    # OR wrap the similarity_search call:
    with self.engine.connect() as conn:
        conn.execute(text(f"SET hnsw.ef_search = {ef}"))
        # ... call similarity_search on a store bound to this conn ...
```

The cleanest way to inject `SET` is actually to monkey-patch `PGVectorStore._make_session` or to drop down to raw SQL for the search path (see Finding 5 below — which also gives you a much bigger win).

**Quick alternative:** for the CLI use case, just `ALTER TABLE ... SET (hnsw.ef_search = 100)` once globally — this becomes the default for the whole database, no code changes needed:

```sql
ALTER TABLE semsearch_chunks SET (
    hnsw.ef_search = 100
);
```

You can run this in `psql` immediately and benchmark — no deploy needed.

---

### Finding 4: DiskANN's `search_list_size` is fixed at index creation

**Location:** `store.py` lines 199–222

Same situation as HNSW: `search_list_size = 100` (line 203) is fixed at index build time. pgvectorscale supports changing it per-query via `diskann.search_list_size` SET, but again there's no hook for that in the search path.

For most workloads `search_list_size` between 100–200 gives good recall at k=5. For low-latency / high-k, you'd want to raise it. Worth exposing in config alongside the HNSW setting.

---

### Finding 5: `similarity_search_with_score()` goes through LangChain's abstraction — drop to raw SQL

**Location:** `service.py` line 510

```python
results_with_scores = self.store.similarity_search_with_score(
    query, k=fetch_k, filter=filter,
)
```

LangChain's `PGVectorStore.similarity_search_with_score` does several things you don't need on the hot path:

1. Re-builds the SQL from a template string every call
2. Wraps the embedding call inside its own try/except (extra frames)
3. Rebuilds Document objects with a metadata dict per row
4. Handles filter as a JSONB-compatible dict (your service already knows `source` is a top-level column)
5. Calls `embed_query` internally — meaning you can't see or cache the embedding

You're already comfortable with raw psycopg elsewhere in the same file. Doing the same for search gives you:

- A direct path to the HNSW/DiskANN index
- The ability to set `hnsw.ef_search` per query
- The ability to cache embeddings (Finding 7)
- Better control over what columns are returned (you can skip the metadata blob if you don't need it)
- The ability to use `executemany`-style batching for multi-query throughput

**Before:**

```python
results_with_scores = self.store.similarity_search_with_score(
    query, k=fetch_k, filter=filter,
)
```

**After:**

```python
# service.py — new private method

def _raw_similarity_search(
    self,
    query_vec: list[float],
    k: int,
    filter: dict | None = None,
    ef_search: int | None = None,
) -> list[tuple[Document, float]]:
    """Direct SQL path — bypasses PGVectorStore overhead.

    Returns (Document, cosine_distance) tuples, mirroring LangChain's API
    so the rest of search() doesn't change.
    """
    table = self.settings.collection_name

    # Build WHERE clause from the same filter dict the public API accepts.
    # Mirrors the semantics in service.delete() — `source` is a top-level column,
    # everything else goes through JSONB.
    conditions: list[str] = []
    params: list[Any] = []
    if filter:
        for key, value in filter.items():
            if key == "source":
                conditions.append("source = %s")
            elif isinstance(value, dict) and "$ilike" in value:
                conditions.append("source ILIKE %s")
                value = value["$ilike"]
            else:
                conditions.append(f"langchain_metadata->>'{key}' = %s")
            params.append(str(value))

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # pgvector accepts the vector as a string literal: '[0.1,0.2,...]'
    vec_literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    sql = (
        f"SELECT langchain_id, content, langchain_metadata, source, "
        f"       chunk_index, "
        f"       embedding <=> %s::vector AS distance "
        f"FROM {table} "
        f"{where_sql} "
        f"ORDER BY embedding <=> %s::vector "
        f"LIMIT %s"
    )

    conn = self._get_conn()
    try:
        with conn.cursor() as cur:
            if ef_search is not None:
                cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
            cur.execute(sql, [vec_literal, vec_literal, k])
            rows = cur.fetchall()
    finally:
        conn.close()

    # Convert to the same (Document, float) shape LangChain returns.
    results: list[tuple[Document, float]] = []
    for langchain_id, content, metadata, source, chunk_index, distance in rows:
        meta = metadata or {}
        meta.setdefault("langchain_id", langchain_id)
        meta.setdefault("source", source)
        meta.setdefault("chunk_index", chunk_index)
        results.append((Document(page_content=content, metadata=meta), float(distance)))
    return results
```

Then `search()` becomes:

```python
def search(self, query, k=None, filter=None, rerank=False, ef_search=None):
    if k is None:
        k = self.settings.default_k
    if not (1 <= k <= 50):
        raise ValueError(f"k must be between 1 and 50, got {k}")

    fetch_k = k * 4 if rerank else k
    ef = ef_search or self.settings.hnsw.ef_search

    # Embed once, reuse below + cache (see Finding 7).
    query_vec = self.embedder.embed_query(query)

    results_with_scores = self._raw_similarity_search(
        query_vec, k=fetch_k, filter=filter, ef_search=ef,
    )
    # ... rest unchanged ...
```

**Note on the duplicate `%s::vector`** in the SQL: pgvector's planner needs the vector for both the WHERE-side filter (when filtering + ranking) and the ORDER BY. You could rewrite it with a CTE to embed once, but for k ≤ 50 the planner is already efficient. Benchmark before optimizing.

---

### Finding 6: No embedding cache for repeat queries

In any RAG/search UX, users tend to repeat queries (UI reloads, "did you mean" flows, A/B tests with the same query set). Each repeat currently pays:

- Embedding API round-trip (~50–200 ms)
- DB vector search (~5–50 ms with HNSW, k=5)

For a hot query, the embedding dominates. A small LRU cache on the embedding vector eliminates ~80% of that latency on repeats.

**Before:**

```python
query_vec = self.embedder.embed_query(query)
```

**After:**

```python
# service.py — at module or class scope
from functools import lru_cache
import hashlib

class SemanticSearchService:
    # ...
    def __init__(self, ...):
        # ...
        # Bounded LRU for embedding vectors. Keyed by (provider, model, query)
        # to invalidate on config changes. 1024 entries ≈ 4 MB for 1536-dim float32.
        self._embed_cache: dict[tuple, list[float]] = {}
        self._embed_cache_order: list[tuple] = []
        self._embed_cache_max = 1024

    def _embed_query_cached(self, query: str) -> list[float]:
        cache_key = (
            self.settings.embedding_provider.type,
            self.settings.embedding_provider.model,
            query,
        )
        if cache_key in self._embed_cache:
            # Mark as recently used.
            self._embed_cache_order.remove(cache_key)
            self._embed_cache_order.append(cache_key)
            return self._embed_cache[cache_key]

        vec = self.embedder.embed_query(query)
        self._embed_cache[cache_key] = vec
        self._embed_cache_order.append(cache_key)

        # Evict LRU.
        while len(self._embed_cache) > self._embed_cache_max:
            old = self._embed_cache_order.pop(0)
            self._embed_cache.pop(old, None)
        return vec
```

Or simpler, if you don't mind the dep:

```python
from cachetools import LRUCache
self._embed_cache = LRUCache(maxsize=1024)
# ...
cache_key = (provider, model, query)
if cache_key in self._embed_cache:
    return self._embed_cache[cache_key]
vec = self.embedder.embed_query(query)
self._embed_cache[cache_key] = vec
```

**Note:** Be careful with PHI / sensitive queries — the cache lives in process memory. If you're handling user data, scope the cache key per-tenant.

---

### Finding 7: Reranker creates a new `httpx` client per call (and is sync, with no retry)

**Location:** `reranker.py` lines 67–82

```python
response = httpx.post(
    self.base_url,
    headers={...},
    json={...},
    timeout=30.0,
)
```

Every call to `rerank()` constructs a new HTTP request from scratch — no connection reuse, no retry, no async. For a reranker hit on every search, this means a fresh TCP+TLS handshake each time. On OpenRouter that's ~150 ms of pure setup overhead per call.

**Before:**

```python
class Reranker:
    def __init__(self, config, api_key):
        self.base_url = config.base_url
        self.model = config.model
        self.api_key = api_key
        self.default_top_n = config.top_n

    def rerank(self, query, documents, top_n=None):
        # ...
        response = httpx.post(
            self.base_url,
            headers={...},
            json={...},
            timeout=30.0,
        )
```

**After:**

```python
import httpx

class Reranker:
    def __init__(self, config, api_key):
        self.base_url = config.base_url
        self.model = config.model
        self.api_key = api_key
        self.default_top_n = config.top_n

        # Persistent client — connection pool survives across rerank() calls.
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
        payload = {"model": self.model, "query": query, "documents": texts, "top_n": n}

        # Retry with exponential backoff. Rerank APIs occasionally 429.
        for attempt in range(3):
            try:
                response = self._client.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise SearchError(f"Rerank failed: {e}") from e
            except httpx.HTTPError as e:
                raise SearchError(f"Rerank failed: {e}") from e

        # ... rest unchanged ...
```

---

## 3. Throughput — Findings & Fixes

### Finding 8: No connection pool — every operation calls `psycopg.connect()`

**Location:** `service.py` line 77

```python
def _get_conn(self) -> psycopg.Connection:
    """Get a raw psycopg connection."""
    return psycopg.connect(self._db_url)
```

Every `stats()`, every `init_schema()`, every `delete()`, and even the `search()` path in Finding 5 above opens a brand-new TCP connection to Postgres. That's 5–50 ms per call depending on network — for a service handling 100 req/s, you're spending 0.5–5 seconds of CPU just on connection setup, plus the Postgres server is wasting a fork per connection (if not using PgBouncer in front).

`psycopg` (v3) ships with `psycopg_pool.ConnectionPool`. Use it.

**Before:**

```python
def _get_conn(self) -> psycopg.Connection:
    """Get a raw psycopg connection."""
    return psycopg.connect(self._db_url)

# Every caller:
conn = self._get_conn()
try:
    # ... use conn ...
finally:
    conn.close()
```

**After:**

```python
# service.py

from psycopg_pool import ConnectionPool

class SemanticSearchService:
    def __init__(self, settings, engine, embedder, store=None):
        # ...
        # Shared pool. min_size=2 keeps warm connections; max_size=10 caps
        # concurrency. Tune based on expected QPS.
        self._pool = ConnectionPool(
            conninfo=self._db_url,
            min_size=2,
            max_size=10,
            timeout=30.0,
            # Pre-warm by setting application_name for easy psql debugging.
            kwargs={"options": f"-c application_name=semsearch"},
        )

    def _get_conn(self) -> psycopg.Connection:
        """Borrow a pooled connection. Caller MUST use `with` or close it."""
        return self._pool.getconn()

    def _return_conn(self, conn):
        """Return a borrowed connection to the pool."""
        self._pool.putconn(conn)

    def close(self):
        try:
            self._pool.close()
        except Exception:
            pass
        # ... existing engine.close() ...
```

And callers switch to a context manager pattern:

```python
# Before
conn = self._get_conn()
try:
    with conn.cursor() as cur:
        cur.execute("...")
finally:
    conn.close()

# After
conn = self._get_conn()
try:
    with conn.cursor() as cur:
        cur.execute("...")
finally:
    self._return_conn(conn)  # returns to pool, doesn't close
```

Or even cleaner — add a context manager helper:

```python
from contextlib import contextmanager

@contextmanager
def _conn(self):
    conn = self._get_conn()
    try:
        yield conn
    finally:
        self._return_conn(conn)

# Usage:
with self._conn() as conn:
    with conn.cursor() as cur:
        cur.execute("...")
```

**Bonus:** Add `pgbouncer` in front of Postgres in production (transaction-mode). Combined with the pool above, this lets the service handle 10× more concurrent queries without Postgres forking.

---

### Finding 9: The whole stack is sync — no async/await anywhere

**Why this matters for throughput:** A sync stack means each in-flight request occupies one Python thread (or process) for the entire duration of: API call to embedder → DB round-trip → optional API call to reranker. For 200 ms of latency per request, you need 50 threads/processes to hit 250 QPS — and Python's GIL means you can't actually parallelize CPU work between them.

The architecture docs already flag this as a future improvement ("Native async/await for all operations"). Here's the concrete plan:

1. **Embedder**: `langchain-openai` ships `AsyncOpenAIEmbeddings` — drop-in.
2. **PGVectorStore**: `langchain-postgres` `PGVectorStore` has async methods (`aadd_documents`, `asimilarity_search_with_score`).
3. **psycopg**: `psycopg.AsyncConnection` + `psycopg_pool.AsyncConnectionPool`.
4. **Reranker**: `httpx.AsyncClient`.
5. **CLI**: keep sync — typer doesn't love async.

You can introduce async *incrementally*:

- Phase 1: Add `async def search_async(...)` alongside the sync `search()`. Wrap with `asyncio.run()` from the CLI. Internally, use `embedder.aembed_query` + raw `psycopg.AsyncConnection`.
- Phase 2: Add an HTTP server (FastAPI) exposing `/search`. Now you can serve 100+ concurrent queries from a single Python process.
- Phase 3: Convert ingest too (`async def ingest_async`).

**Async search skeleton:**

```python
# service.py — alongside existing sync code

async def search_async(
    self,
    query: str,
    k: int | None = None,
    filter: dict | None = None,
    rerank: bool = False,
) -> list[SearchResult]:
    if k is None:
        k = self.settings.default_k
    if not (1 <= k <= 50):
        raise ValueError(f"k must be between 1 and 50, got {k}")

    fetch_k = k * 4 if rerank else k
    query_vec = await self.embedder.aembed_query(query)

    # Use psycopg AsyncConnectionPool (set up once at __init__)
    async with self._async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SET LOCAL hnsw.ef_search = {self.settings.hnsw.ef_search}")
            await cur.execute(
                f"SELECT langchain_id, content, langchain_metadata, source, chunk_index, "
                f"       embedding <=> %s::vector AS distance "
                f"FROM {self.settings.collection_name} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                [self._vec_literal(query_vec), self._vec_literal(query_vec), fetch_k],
            )
            rows = await cur.fetchall()

    # Build SearchResult objects (same logic as sync path)
    results = [self._row_to_search_result(r) for r in rows]

    if rerank:
        reranker = self.reranker
        if reranker is None:
            raise SearchError("Reranker not configured")
        results = await reranker.arerank(query, results, top_n=k)

    return results[:k]
```

Note: `langchain_postgres.PGVectorStore` already uses asyncpg under the hood — but you don't get the benefits as long as `similarity_search_with_score` is sync. Going raw psycopg-async is the cleanest path.

---

### Finding 10: Batched embedding API for multi-query throughput

If you ever need to process a batch of queries (e.g. evaluating a test set, building an embedding cache), the current code calls `embed_query()` in a loop — one HTTP request per query. Both OpenAI and OpenRouter support batching up to 2048 inputs in a single `embed_documents` call. LangChain's `OpenAIEmbeddings` exposes this as `embed_documents(texts)`.

**Use case:** Say you have 1000 test queries and want to compute recall@10. With the current code:

```python
# Slow — 1000 sequential API calls
for q in queries:
    vec = embedder.embed_query(q)  # ~150 ms each
    search(vec)
# Total: 150 seconds
```

**Better — batch the embeddings:**

```python
# Fast — one batched API call, then 1000 cheap DB queries
vectors = embedder.embed_documents(queries)  # ~3 seconds for 1000 with OpenAI
for vec in vectors:
    search_by_vector(vec)
# Total: ~3 + 1000*0.02 = 23 seconds  (~6.5× faster)
```

The service currently exposes `search(query, ...)` which takes a string. Add a vector-based entry point:

```python
# service.py — new method

def search_by_vector(
    self,
    query_vec: list[float],
    k: int | None = None,
    filter: dict | None = None,
    ef_search: int | None = None,
) -> list[SearchResult]:
    """Search by a pre-computed embedding vector.

    Useful for batch evaluation: embed all queries with embed_documents()
    once, then loop over vectors without re-paying the embedding API cost.
    """
    # ... same body as the refactored search() minus the embed_query call ...
```

---

### Finding 11: No result cache at the search-result level

After Findings 6 + 8 + 9, you're down to ~10–30 ms per cached query. If your workload has any repetition (popular queries, UI paginations, "did you mean" flows), a result cache on `(query, k, filter_hash)` eliminates even the DB hit.

```python
# service.py

import hashlib
import json

class SemanticSearchService:
    def __init__(self, ...):
        # ...
        self._result_cache: dict[str, list[SearchResult]] = {}
        self._result_cache_max = 256

    def search(self, query, k=None, filter=None, rerank=False):
        # ...
        cache_key = self._result_cache_key(query, k, filter, rerank)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        # ... existing search logic ...

        # Cache only non-reranked results — rerank scores depend on the reranker
        # service which may have its own caching.
        if not rerank:
            self._result_cache[cache_key] = results
            if len(self._result_cache) > self._result_cache_max:
                # Evict oldest (simple FIFO; replace with LRU if you care)
                self._result_cache.pop(next(iter(self._result_cache)))
        return results

    def _result_cache_key(self, query, k, filter, rerank) -> str:
        filter_str = json.dumps(filter or {}, sort_keys=True)
        raw = f"{query}|{k}|{filter_str}|{rerank}"
        return hashlib.sha256(raw.encode()).hexdigest()
```

**Caveat:** Result caching means stale results if the underlying data changes. You'll want to invalidate on `ingest()`, `delete()`, or `reingest()`. Easiest invalidation:

```python
def ingest(self, path, *, reembed_unchanged=False, conn=None):
    # ... existing code ...
    self._result_cache.clear()  # invalidate on any write
    return result
```

For finer-grained invalidation, scope the cache key by `source` and only invalidate entries whose filter overlaps the ingested source. That's more code; only worth it if cache hit rate matters.

---

## 4. Prioritized Roadmap

Group the 11 findings into a sequenced plan you can execute one PR at a time. Each item shows expected latency/QPS impact and rough effort.

### Phase 1 — Quick wins (1 day total, biggest bang/buck)

| # | Task | Impact | Effort |
|---|---|---|---|
| 1 | Replace `_get_vector_size()` with cached probe + DB read in `stats()` | `stats` latency: 200 ms → 5 ms; saves API cost | 15 min |
| 3 | `ALTER TABLE ... SET (hnsw.ef_search = 100)` + expose `hnsw` config | Recall +5–10% at fixed latency | 30 min |
| 7 | Reuse `httpx.Client` in `Reranker`, add retry | Reranker latency: 200 ms → 50 ms after warmup | 30 min |
| 2a | Add `semsearch repl` for batch dev workflows | Eliminates ~500 ms cold start per query in dev | 1 hr |
| 2b | Share `httpx.Client` in `OpenAIEmbeddings` via `client=` kwarg | Embed call latency: 200 ms → 50 ms after warmup | 1 hr |

### Phase 2 — Architecture (3–5 days)

| # | Task | Impact | Effort |
|---|---|---|---|
| 8 | Add `psycopg_pool.ConnectionPool` to `SemanticSearchService` | -5 to -50 ms per query; +10× concurrency headroom | 1 day |
| 5 | Replace `similarity_search_with_score` with raw psycopg SQL | -2 to -10 ms per query; enables (3), (6), (10) | 1 day |
| 6 | Add LRU embedding cache (1024 entries) | Repeat-query latency halved | 2 hrs |
| 11 | Add result cache with write-invalidation | Repeat-query latency: 30 ms → 0.5 ms | 2 hrs |
| 4 | Expose `diskann.search_list_size` per-query when pgvectorscale is installed | Recall tuning for high-dim embeddings | 2 hrs |

### Phase 3 — Async conversion (1–2 weeks, parallel work)

| # | Task | Impact | Effort |
|---|---|---|---|
| 9 | Add `search_async()` alongside sync `search()` using `psycopg.AsyncConnection` + `AsyncOpenAIEmbeddings` + `httpx.AsyncClient` | 10× concurrency on a single Python process | 3–5 days |
| 9b | Add FastAPI HTTP server exposing `/search`, `/ingest`, `/stats` | Multi-client service mode | 2–3 days |
| 10 | Add `search_by_vector()` for batched evaluation workflows | 6–10× faster batch eval / cache warmup | 1 day |

---

## 5. How to validate each fix

You can't claim a perf win without measuring. Add a small benchmark harness before you start Phase 1 so you have before/after numbers:

```python
# scripts/bench_search.py
"""Quick benchmark for the search path.

Usage:
    python scripts/bench_search.py --queries queries.txt --n 100
"""
import argparse, time, statistics, json
from semsearch.config import get_settings
from semsearch.service import SemanticSearchService

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    queries = [l.strip() for l in open(args.queries) if l.strip()][:args.n]
    settings = get_settings()

    # Warm up — pay cold-start once
    with SemanticSearchService.from_settings(settings) as svc:
        svc.search(queries[0], k=args.k)

        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            svc.search(q, k=args.k)
            latencies.append((time.perf_counter() - t0) * 1000)

    print(json.dumps({
        "n": len(latencies),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
    }, indent=2))

if __name__ == "__main__":
    main()
```

Run before and after each Phase 1 fix. Commit the results to a `BENCHMARKS.md` so the next person knows what changed.

---

## 6. What I didn't cover (and why)

- **Memory efficiency** — you didn't ask about it, and at default corpus sizes (<1M chunks) the working set fits in RAM comfortably. The one thing worth flagging: high-dim embeddings (4096-dim Qwen3) bloat both the table and the HNSW index. DiskANN with SBQ compression (already in `config.py`) is the right mitigation if you go that route. Worth measuring before adopting.
- **Scalability** — same reason. The single-node PostgreSQL + pgvector pattern holds up to ~10M vectors before you need sharding. If you cross that threshold, the migration is: (a) try pgvectorscale DiskANN first (much better than HNSW at scale), (b) only then consider moving to a dedicated vector DB (Qdrant, Milvus, Weaviate).
- **Cost/infra** — your current `OpenAIEmbeddings` config doesn't set `dimensions` for `text-embedding-3-small`/`large`, which supports Matryoshka truncation. Dropping from 1536 to 256 dims is a 6× cost+latency win at <2% recall loss on most evals. Worth a separate experiment.

---

## TL;DR — what to do this week

1. Fix `stats()` to not embed a probe string (15 min, saves real API money)
2. Run `ALTER TABLE semsearch_chunks SET (hnsw.ef_search = 100);` and re-benchmark (5 min)
3. Replace `httpx.post(...)` in `Reranker.rerank()` with a shared `httpx.Client` (30 min)
4. Add `psycopg_pool.ConnectionPool` to `SemanticSearchService.__init__` (1 hr)
5. Drop `similarity_search_with_score` for raw SQL (1 hr — unlocks further tuning)

Those five changes together should cut p50 search latency roughly in half on a typical OpenRouter-backed setup, and they unblock the async migration in Phase 3.
