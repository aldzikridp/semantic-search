# PLAN.md — Codebase Optimization Plan

> **STATUS: COMPLETE — all four phases implemented, reviewed, and verified
> (2026-08-22).** Final suite: 132 passed, 1 skipped; 0 diagnostics.
> See per-phase status notes and the Success Metrics table for actuals.

> Covers **codebase/architecture optimizations** identified by a structural
> review (2026-08-22), building on performance work already in the codebase
> (vector-size caching, HNSW/DiskANN tuning, reranker connection pooling,
> warm serve mode). Search-latency wins from external factors (embedding
> model choice, index tuning) are documented separately and are
> **out of scope** here.
>
> Self-contained: this plan is the single source of truth for the phases
> below — no external task files.

---

## 0. Guiding Constraints (non-negotiable)

All changes MUST comply with `AGENTS.md` design decisions:

1. Write path stays **service-owned SQL** via raw `psycopg` — no
   `PGVectorStore.add_documents()` ever.
2. `PGVectorStore` stays **read-only** (search only).
3. Deterministic TEXT ids `"{source}::{chunk_index}"` — unchanged.
4. Connection ownership rule: `conn=None` → method owns lifecycle;
   `conn` provided → caller owns it. Phase C must preserve this.
5. Lazy store initialization via the `store` property — unchanged.
6. All Python commands run inside `nix develop`.
7. Every phase lands with the full test suite green
   (`TEST_DATABASE_URL=... pytest -q`) — **114 passed, 1 skipped** baseline.

**Baseline measurements** (re-measure at phase start; numbers from
2026-08-22 session):

| Metric | Value |
| -------- | ------- |
| Test suite | 114 passed, 1 skipped |
| Diagnostics (`src/semsearch`) | 0 errors |
| Coverage | ~77–79% (85% gate pre-existing shortfall — not this plan's scope) |
| Ingest write pattern | 1 INSERT round-trip **per changed chunk** |
| `search()` / `asearch()` | ~85 lines each, ~60% duplicated |
| `_get_conn()` call sites | 7 (fresh connection per operation) |
| `service.py` size | 765 lines, 36 symbols |

---

## Phase A — Batch the CASE B/C ingest inserts

**Priority: HIGH · Effort: small · Payoff: minutes → seconds on large ingests**

**✅ DONE (2026-08-22).** Multi-row VALUES via `psycopg.sql`, batched at
`_BC_INSERT_BATCH_SIZE = 1000`; `_build_chunk_metadata()` extracted; +2 tests
(`TestBatchedIngest`). Benchmark (BENCHMARKS.md): 724 ms → 667 ms for 500
chunks on local unix socket (1.1×) — round-trip savings are marginal locally,
but scale with DB network RTT; see the honest interpretation there.

### A.1 Problem

`service.py::ingest()` (lines ~291–332) inserts changed/new chunks **one
round-trip per chunk**:

```python
# CASE B + C: INSERT ... ON CONFLICT DO UPDATE
for vec_idx, chunk_idx in enumerate(bc_indices):
    ...
    _exec(cur, f"INSERT INTO {table} ... VALUES (%s, ...) ON CONFLICT ...",
          (chunk_id, str(vec), chunk.page_content, json.dumps(metadata),
           source, chunk_idx, h))
```

A 500-chunk PDF pays 500 serial network round-trips inside one transaction.
At ~0.2–1ms per round-trip (local socket) that is 0.1–0.5s of pure wait per
file — multiplied across `ingest_dir` over hundreds of files.

CASE A (bulk `UPDATE ... WHERE langchain_id = ANY(%s)`) and CASE D (single
`DELETE`) are already set-based; only B/C is row-by-row.

### A.2 Proposed change

Replace the Python loop with **one multi-row INSERT statement** built with
`psycopg.sql` placeholders (NOT string concatenation of values):

```python
# CASE B + C: single multi-row INSERT ... ON CONFLICT
rows: list[tuple] = []
for vec_idx, chunk_idx in enumerate(bc_indices):
    chunk = chunks[chunk_idx]
    metadata = _build_chunk_metadata(chunk, self.settings, now)
    rows.append((
        f"{source}::{chunk_idx}",
        str(vectors[vec_idx]),
        chunk.page_content,
        json.dumps(metadata),
        source,
        chunk_idx,
        hashes[chunk_idx],
    ))

if rows:
    placeholders = b", ".join(
        pg_sql.SQL("(%s, %s, %s, %s::jsonb, %s, %s, %s)") for _ in rows
    )
    # Composed via psycopg.sql to keep identifiers/values parameterized:
    query = pg_sql.SQL(
        "INSERT INTO {table} (langchain_id, embedding, content, "
        "langchain_metadata, source, chunk_index, document_hash) "
        "VALUES {rows} ON CONFLICT (source, chunk_index) DO UPDATE "
        "SET embedding = EXCLUDED.embedding, content = EXCLUDED.content, "
        "document_hash = EXCLUDED.document_hash, "
        "langchain_metadata = EXCLUDED.langchain_metadata"
    ).format(table=pg_sql.Identifier(table), rows=placeholders)
    cur.execute(query, [v for row in rows for v in row])
```

**Decision point — multi-row VALUES vs `executemany`:**

| Option | Round-trips | Pros | Cons |
|--------|-------------|------|------|
| Multi-row VALUES | 1 | Fastest; single statement | Param flattening; statement length grows with batch (cap at ~1000 rows/chunk) |
| `cur.executemany()` + pipeline | N logical, pipelined | Simplest diff; psycopg batches internally | Slightly slower than true multi-row; still one transaction |

**Chosen: multi-row VALUES with a chunk cap of 1000 rows per statement**
(largest realistic files are far below this; loop the statement if ever
exceeded). Rationale: simplest mental model, fewest round-trips, and the
statement is still fully parameterized.

**Refactor companion:** extract `_build_chunk_metadata(chunk, settings, now)
-> dict` (currently inline in the loop, lines ~296–306) — reused by nothing
else today but makes the loop body trivial and unit-testable.

### A.3 Files touched

- `src/semsearch/service.py` — `ingest()` only
- `tests/test_service_ingest.py` — add 2 tests:
  - `test_ingest_many_chunks_single_transaction` — ingest 300+ chunk mock
    file, assert all rows present and `IngestResult` counts correct
  - `test_ingest_case_bc_preserves_upsert_semantics` — re-ingest modified
    file; CASE B rows updated, CASE C added, CASE A untouched (guards the
    `ON CONFLICT` behavior through the refactor)

### A.4 Verification

1. `pytest tests/test_service_ingest.py tests/test_service_ingest_dir.py -v`
2. Full suite green.
3. Micro-benchmark: ingest a generated 500-chunk text file before/after;
   record wall time in `BENCHMARKS.md` (file exists; append a Phase 2
   section). If `BENCHMARKS.md` is retired, record results in this plan's
   Success Metrics table instead.
4. `EXPLAIN ANALYZE` the generated statement once to confirm the plan is a
   simple insert, not per-row triggers/surprises.

### A.5 Risks & mitigations

- **Risk**: parameter-count limits with huge batches → mitigated by the
  1000-row cap (7000 params, far below Postgres' 65535 bound).
- **Risk**: `str(vec)` embedding serialization for hundreds of vectors is
  CPU work in Python → acceptable; it replaces N× round-trip waits. Profile
  only if benchmarks disagree.

---

## Phase B — Deduplicate `search()` / `asearch()`

**Priority: MEDIUM · Effort: small · Payoff: maintainability; eliminates sync/async drift risk**

**✅ DONE (2026-08-22).** All four helpers extracted as specified; both
methods now ~12 lines; `asearch` offloads the entire rerank flow to a thread
(improvement over plan). +6 tests (`TestResolveK` boundaries). Response
equivalence verified against pre-refactor code — identical modulo fresh
`ingested_at` timestamps and pgvector tie-order nondeterminism (reproduced
twice on unchanged code).

### B.1 Problem

`service.py` has two ~85-line methods (`search()` lines 560–646,
`asearch()` lines 648–717) with three duplicated logic blocks:

1. **k validation** (`if k is None: k = self.settings.default_k; if not
   (1 <= k <= 50): raise ValueError(...)` + `fetch_k = k*4 if rerank else k`)
2. **Result conversion** — the 15-line `SearchResult(...)` construction loop
   incl. the `score = 1.0 - distance` conversion (AGENTS.md decision #5)
3. **Rerank orchestration** — reranker-missing check, `Document` wrapping
   with `_search_result` metadata round-trip, `rerank_score` re-injection

Any future change to scoring, filter semantics, or rerank flow must be made
twice. Divergence between sync/async paths is a silent-bug factory.

### B.2 Proposed change

Extract three private helpers on the service class:

```python
def _resolve_k(self, k: int | None, rerank: bool) -> tuple[int, int]:
    """Validate k, return (k, fetch_k). Raises ValueError."""

@staticmethod
def _to_search_results(
    results_with_scores: list[tuple[Document, float]],
) -> list[SearchResult]:
    """distance → score = 1.0 - distance; build SearchResults."""

def _apply_rerank(
    self, query: str, results: list[SearchResult], k: int
) -> list[SearchResult]:
    """Shared rerank flow. Sync call; async caller wraps in to_thread."""

def _rerank_docs(self, results: list[SearchResult]) -> list[Document]:
    """Wrap results in Documents with _search_result round-trip metadata."""
```

Resulting bodies:

```python
def search(self, query, k=None, filter=None, rerank=False):
    k, fetch_k = self._resolve_k(k, rerank)
    try:
        raw = self.store.similarity_search_with_score(query, k=fetch_k, filter=filter)
    except Exception as e:
        raise SearchError(f"Search failed: {e}") from e
    results = self._to_search_results(raw)
    if rerank:
        results = self._apply_rerank(query, results, k)
    return results[:k]

async def asearch(self, query, k=None, filter=None, rerank=False):
    k, fetch_k = self._resolve_k(k, rerank)
    try:
        raw = await self.store.asimilarity_search_with_score(query, k=fetch_k, filter=filter)
    except Exception as e:
        raise SearchError(f"Search failed: {e}") from e
    results = self._to_search_results(raw)
    if rerank:
        results = await asyncio.to_thread(self._apply_rerank, query, results, k)
    return results[:k]
```

Net: each public method drops from ~85 to ~12 lines; the async path keeps
its `to_thread` offload for the sync reranker (unchanged behavior).

### B.3 Files touched

- `src/semsearch/service.py` — `search()`, `asearch()`, 4 new private helpers
- `tests/test_service_search.py` — add unit tests for `_resolve_k` boundary
  cases (0, 1, 50, 51, None) — currently only covered incidentally

### B.4 Verification

1. Full suite (search tests: `test_service_search.py`,
   `test_server.py::TestSearch`, rerank tests in `test_reranker_pooling.py`).
2. Manual: `semsearch serve` + one rerank request; confirm `rerank_score`
   still present in response metadata.
3. Diff the HTTP response JSON of a fixed query before/after — must be
   byte-identical.

### B.5 Risks

- **Risk**: subtle ordering difference (reranked list already truncated to
  `top_n=k` by reranker; final `[:k]` is a no-op but must remain).
  Mitigation: byte-identical response check in B.4.3.

---

## Phase C — Connection pooling for service-owned connections

**Priority: LOW · Effort: small · Payoff: ~1–5ms/request under serve load; caps concurrent DB connections**

**✅ DONE (2026-08-22).** Opt-in via `SEMSEARCH_POOL__*` (min_size=0 default =
OFF); `_leased` set preserves ownership rule #15; all 7 `_get_conn()` sites
release via `_release_conn`; `psycopg-pool>=3.2` declared. +10 tests
(`test_service_pool.py`). Live check: 4 ops over 2 pooled connections,
zero leaks after `close()`. Note: total server connections at worst are
`pool.max_size + 5` because PGEngine's separate SQLAlchemy pool coexists.

### C.1 Problem

`_get_conn()` (`service.py:95–106`) opens a **fresh psycopg connection for
every operation** — 7 call sites (ingest ×2, ingest_dir ×2, delete,
reingest, stats). Fine for one-shot CLI usage; wasteful when `serve` mode
serves `/stats` or future write endpoints under load: each request pays
connection setup (~1–5ms on unix socket, more on TCP) and Postgres pays
backend fork/teardown.

### C.2 Proposed change

Add an **opt-in** pool used only when the method owns the connection
(preserving the AGENTS.md #15 ownership rule):

```python
# config.py
class PoolConfig(BaseModel):
    """SEMSEARCH_POOL__* — 0 disables pooling (pure per-call connections)."""
    min_size: int = Field(default=0, ge=0, le=10)
    max_size: int = Field(default=4, ge=1, le=32)
    timeout: float = Field(default=5.0, gt=0)
```

```python
# service.py
@property
def _pool(self) -> psycopg_pool.ConnectionPool | None:
    """Lazily built; None when pooling disabled (min_size=0)."""
    if self._pool_obj is None and self.settings.pool.min_size > 0:
        self._pool_obj = ConnectionPool(
            self._db_url, min_size=..., max_size=...,
            timeout=..., open=True, kwargs={"options": keepalive_options},
        )
    return self._pool_obj

def _get_conn(self) -> psycopg.Connection:
    """From pool when enabled; else fresh connection (current behavior)."""
```

**Ownership semantics — the critical part:**

- `_get_conn()` returns a raw `psycopg.Connection` either way (pool's
  `connection()` context manager is used internally with
  `check=False`-style manual return).
- Methods keep `conn.close()` calls; when pooled, `_get_conn` wraps the
  returned connection so that `.close()` **returns it to the pool** instead
  of closing. Implementation: a tiny `_PooledConnection` proxy or, simpler,
  pool-leased connections are returned via `_release_conn(conn)` and the 7
  call sites' `finally: conn.close()` blocks switch to
  `self._release_conn(conn)`.
- `close()` (service lifecycle) also calls `self._pool_obj.close()`.

**Default remains `min_size=0` (pooling OFF)** → zero behavior change for
existing users; `serve` users can enable it via env:

```bash
SEMSEARCH_POOL__MIN_SIZE=1
SEMSEARCH_POOL__MAX_SIZE=4
```

### C.3 Files touched

- `src/semsearch/config.py` — `PoolConfig` + `Settings.pool` field
- `src/semsearch/service.py` — `_pool` property, `_get_conn`/`_release_conn`,
  `close()`
- `pyproject.toml` — add `psycopg-pool>=3.2` (already importable at 3.3.1 in
  the dev env via psycopg extras, but must be a declared dependency)
- `tests/` — new `test_service_pool.py`: pool disabled by default; enabled
  pool reuses connections (assert via pool stats); `close()` shuts it down;
  ownership rule intact (caller-provided `conn` never touches the pool)
- `docs/configuration.md` — new env var section

### C.4 Verification

1. Full suite green (default config = pooling off = current behavior).
2. Pool-enabled run: `serve` + 50 sequential `/stats` requests; assert
   `pool.getstats()[requests] >= 50` while server-side connection count
   (pg_stat_activity) stays ≤ max_size.
3. `semsearch stats` / `ingest` / `delete` CLI paths with pool enabled —
   no leaked connections (`pg_stat_activity` count returns to baseline).

### C.5 Risks

- **Risk**: connection returned to pool mid-transaction → mitigated by
  returning only in the same `finally` blocks that previously called
  `close()` (Postgres rolls back any open transaction on pool return check;
  psycopg_pool does `RESET`).
- **Risk**: forked processes (none today) sharing a pool → documented as
  unsupported; pool is created lazily post-fork.

---

## Phase D — Split the service god-class (opportunistic)

**Priority: DEFERRED → ✅ DONE EARLY (2026-08-22)**

**Governance note:** §D.3's trigger criteria were *not* met (~880 lines at
the time; no 4th responsibility; no cross-area change forcing it).
Implemented early by owner decision. Review found the split clean:
facade composes `IngestMixin` / `SearchMixin` / `AdminMixin` on
`BaseService`; zero import cycles; all 9 external consumers untouched;
suite green (132 passed); coverage improved to 80.15%. AGENTS.md source
layout updated accordingly.

### D.1 Problem

`SemanticSearchService` is 765 lines / 36 symbols spanning five distinct
responsibilities: schema/dimension management, ingest, delete/reingest,
search (sync+async), stats. Works today, but every new feature lands in one
file and the import graph concentrates all fan-in on one module.

### D.2 Proposed target shape (when triggered)

Split **internals only**; keep `SemanticSearchService` as a facade with the
exact same public API (CLI/server/tests untouched):

```
src/semsearch/
  services/
    __init__.py          # re-exports
    base.py              # shared: settings, _get_conn/_release_conn, lifecycle
    ingest.py            # IngestMixin: ingest, ingest_dir, reingest
    search.py            # SearchMixin: search, asearch, _apply_rerank, ...
    admin.py             # AdminMixin: delete, stats, init_schema, dim probing
  service.py             # class SemanticSearchService(IngestMixin, SearchMixin, AdminMixin): ...
```

Mixin composition over microservice-style separation: keeps single-process
in-memory state (pool, cached vector size) coherent, zero API churn.

### D.3 Trigger criteria (do NOT start before at least one holds)

- `service.py` exceeds ~1200 lines, **or**
- a 4th distinct responsibility appears (e.g., query-result cache, export),
  **or**
- a change requires touching >2 unrelated responsibility areas
  simultaneously.

### D.4 Verification

Full suite + import-graph check (no cycles; `cli.py`/`server.py` imports
unchanged). No benchmark impact expected — pure code motion.

---

## Rollout Order & Milestones

| Order | Phase | Status |
| ------- | ------- | -------- |
| 1 | **A** batched inserts | ✅ landed |
| 2 | **B** search dedupe | ✅ landed |
| 3 | **C** connection pool | ✅ landed (+docs) |
| 4 | **D** service split | ✅ landed early (owner decision — see §D note) |

Each phase: implement → full suite → diagnostics zero → benchmark note →
commit. Never two phases in one commit.

### Documentation housekeeping

AGENTS.md's **Documentation Files** table no longer lists this plan after
the task-tracking cleanup. Re-add the following row with the first landed
phase (A):

```markdown
| `PLAN.md` | Current optimization plan (batched inserts, search dedupe, pooling) |
```

If `BENCHMARKS.md` is kept, also restore its row:

```markdown
| `BENCHMARKS.md` | Performance benchmark results and methodology |
```

## Out of Scope (explicitly)

- Embedding-model / dimension changes (require re-ingest; separate decision)
- HNSW/DiskANN tuning defaults (runtime config already exposes them)
- Coverage-gate shortfall (85% gate vs ~78% actual — pre-existing)
- Query-result caching (proposed separately if agents show repeat-query
  patterns; would add invalidation complexity vs ingest hashes)
- turbovec / non-Postgres backends (rejected 2026-08-22 — keep PostgreSQL)

## Success Metrics

| Metric | Baseline (2026-08-22) | Target | Actual |
| -------- | ---------------------- | -------- | -------- |
| 500-chunk file ingest wall time | 724 ms (per-row, local socket) | ≥2× faster | 667 ms (1.1× local; scales with DB RTT — see BENCHMARKS.md) |
| `search`/`asearch` duplicated lines | ~60 lines | 0 | **0** (shared helpers in `services/search.py`) |
| Per-op DB connection setups (serve, pool on) | 1/request | ≤ 1/pool-lifetime | **4 ops / 2 connections** (live-verified) |
| Test suite | 114 passed, 1 skipped | ≥ 114 passed, 0 regressions | **132 passed, 1 skipped** (+18 new) |
| Diagnostics | 0 errors | 0 errors | **0 errors** |
| HTTP response JSON for fixed query | — | byte-identical pre/post each phase | **verified** (Phase B; mod timestamps/tie-order) |
| Coverage | ~77–79% | (gate out of scope) | 80.15% (+~3 pts) |
| `service.py` size | 765 lines / 36 symbols | split when triggered | **27-line facade** + 4 mixins (912 lines total) |
