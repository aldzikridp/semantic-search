# PLAN.md — Serve-Mode Performance Plan (Warmup, Keep-Alive, Pool Pre-Fill)

> **Previous plan (batched inserts, search dedupe, connection pooling,
> service split) is COMPLETE** — all four phases landed and verified on
> 2026-08-22; summary in the Appendix. This document covers the next unit of
> work: making the first request faster **and** every subsequent request
> faster — **without touching any paid API**.

---

## Phases at a Glance

| Phase | Item | Effort | Win | Depends on |
| ------- | ------ | -------- | ----- | ------------ |
| **E** | Raise HTTP `keepalive_expiry` 10 s → 300 s + reranker transport-retry | small | **~50–150 ms on nearly every request** (steady state) | — |
| **W** | Serve-mode startup warmup | small | removes ~5–30 ms local init from first request | — |
| **F** | Pool pre-fill inside `warmup()` | small | first concurrent burst doesn't serialize on connection setup | W |

Land order: **E → W → F** (E is independent and the biggest win; F extends
W's `warmup()`). One commit per phase.

---

## Phase E — Raise HTTP `keepalive_expiry` (10 s → 300 s)

**Priority: HIGH · Effort: small · Payoff: eliminates repeated TLS handshakes in steady state**

(Effort raised from "trivial" to "small": safe expiry requires one hardening
fix in the reranker — see E.5.)

### E.1 Problem

Both paid-API clients discard idle connections after **10 seconds**:

- `src/semsearch/embeddings.py:42–46` — `_HTTPX_LIMITS = _Limits(..., keepalive_expiry=10.0)`
  (module constant, shared by sync + async clients)
- `src/semsearch/reranker.py:78` — inline `_Limits(..., keepalive_expiry=10.0)`

Real agent workloads issue queries seconds-to-minutes apart, so the pooled
connection is almost always already expired → **nearly every request re-pays
TCP+TLS (~50–150 ms)** to the embedding API (and to the reranker endpoint on
reranked queries). This dwarfs all other per-request overhead we've optimized.

**Not configurable today**: verified — `config.py` only exposes DB TCP
keep-alives (`db_keepalive_idle/interval/count`, a different mechanism);
the HTTP limits are hardcoded in both files.

### E.2 Stale-connection analysis (why 300 s needs one companion fix)

Two flavors of "stale" exist once connections idle longer:

1. **Cleanly closed by server** (FIN received while idle — the common case;
   provider LBs idle-timeout at ~60–120 s): httpcore detects the EOF at
   checkout and silently discards the dead connection, opening a fresh one.
   **Fully transparent — handled.**
2. **Half-open / black-holed** (no FIN — NAT drop, network blip): the client
   cannot know; the request hangs until the read timeout (10 s, already
   configured). Component behavior then diverges:
   - **Embedding path**: protected — OpenAI SDK built with `max_retries=2`
     retries connection errors automatically → slower request, no failure.
   - **Reranker path**: ⚠️ **gap** — the retry loop in `rerank()` catches only
     `_HTTPStatusError` (429 backoff); transport errors propagate uncaught →
     `SearchError` → HTTP 500. With the current 10 s expiry this is nearly
     impossible; with 300 s it becomes occasionally possible.

Worst-case trade: today = every request pays a handshake, stale hits ~never;
with E = requests usually reuse, and a rare stale hit costs one transparent
retry (after E.4).

### E.3 Proposed change (Option A — better hardcoded default)

Change both sites to `keepalive_expiry=300.0` with a rationale comment.
Update the stale comment on embeddings.py line 45.

**Rejected alternative (Option B — expose as config)**: adding
`SEMSEARCH_*__HTTP_KEEPALIVE_EXPIRY` surface area for a value no deployment
plausibly needs to tune. Promoting to config later is trivial if ever needed.

### E.4 Required hardening: reranker transport-error retry

Extend the existing retry loop in `Reranker.rerank()` so connection-level
failures are retried like 429s instead of propagating:

```python
except _HTTPStatusError as e:
    if e.response.status_code == 429 and attempt < _MAX_RETRIES - 1:
        ...  # existing 429 backoff (unchanged)
    else:
        raise
except _HTTPXTransportError as e:   # ConnectError, ReadError, RemoteProtocolError, ...
    if attempt < _MAX_RETRIES - 1:
        logger.warning("Reranker connection error (%s), retrying (attempt %d/%d)",
                       e, attempt + 1, _MAX_RETRIES)
        continue                    # fresh pooled connection on next attempt
    raise SearchError(f"Rerank failed after retries: {e}") from e
```

Notes:

- `_HTTPXTransportError` = `httpx.TransportError` (or `httpx2` equivalent via
  the existing `_httpx_module` indirection) — covers ConnectError,
  ReadTimeout, ReadError, WriteError, RemoteProtocolError.
- A stale half-open hit costs one extra attempt (~read-timeout bound), then
  succeeds on the fresh connection — matching the embedding path's behavior.
- Non-429 HTTP status errors keep raising immediately (unchanged semantics).

### E.5 Files / tests / verification

- `src/semsearch/embeddings.py` — constant + comment (E.3)
- `src/semsearch/reranker.py` — inline limits (E.3) + transport-retry (E.4)
- `tests/test_embeddings.py` — assert `_HTTPX_LIMITS.keepalive_expiry == 300.0`
- `tests/test_reranker_pooling.py` —
  - assert the reranker client's limits carry `keepalive_expiry == 300.0`
    (expose limits on the instance if needed)
  - **new:** simulated stale connection — first `post()` raises
    `ConnectError`, second succeeds → `rerank()` returns results, exactly one
    retry logged, no exception escapes
  - **new:** transport error on all attempts → raises `SearchError` (not a
    raw httpx error) after `_MAX_RETRIES`
- Verify: full suite green.

---

## Phase W — Serve-mode startup warmup

**Priority: MEDIUM · Effort: small · Payoff: removes local lazy-init from first request**

### W.1 Background

Component lifecycle in serve mode:

| Component | Created | First physical connection |
| ----------- | --------- | -------------------------- |
| `PGEngine` (SQLAlchemy pool 5+10) | startup | first search (lazy checkout) |
| Embedder + OpenAI/async clients | startup | first search (TCP+TLS) |
| `PGVectorStore` | **first search** (lazy property) | same moment |
| `Reranker` + `httpx2.Client` | **first rerank request** (lazy property) | that request |
| Opt-in psycopg pool (Phase C) | **first service-owned op** | that operation |

The first `/search` pays `PGVectorStore.create_sync()` + first DB connection
setup; the first reranked search additionally pays `Reranker` construction.

### W.2 Explicitly rejected: warming the embedding/reranker APIs

1. **Availability coupling** — startup would fail or hang when a provider is
   down/rate-limited, even though everything local works.
2. **Zero net saving** — for any server that serves ≥1 search, the warmup
   call merely replaces the first real request's API call; it only wastes
   money when the server starts and never serves.
3. Verified: `Reranker.__init__` only stores config and creates an idle
   `httpx.Client` — **construction sends nothing** (httpx connects lazily),
   so building it at startup is free.

### W.3 Design: `warmup()` in `services/admin.py` (AdminMixin)

```python
def warmup(self) -> dict[str, bool]:
    """Pre-build lazily-initialized local resources.

    Safe at server startup: touches ONLY local resources (PGVectorStore,
    one DB round-trip, reranker client construction). Never contacts the
    embedding or reranker APIs. Fail-open: logs and continues on error so
    startup never depends on DB state or providers.
    """
    result = {"store": False, "db": False, "reranker": False}
    try:
        _ = self.store                      # build PGVectorStore
        result["store"] = True
    except Exception as e:
        logger.warning("Warmup: store init deferred (%s)", e)
    try:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            result["db"] = True
        finally:
            self._release_conn(conn)
    except Exception as e:
        logger.warning("Warmup: DB round-trip deferred (%s)", e)
    if self.settings.reranker is not None:
        try:
            _ = self.reranker               # construct client (no request)
            result["reranker"] = True
        except Exception as e:
            logger.warning("Warmup: reranker init deferred (%s)", e)
    return result
```

### W.4 Lifespan integration (`server.py::create_app`)

```python
svc = service or SemanticSearchService.from_settings(settings)
app.state.service = svc
# Warm local resources only — never touches paid APIs, so startup
# stays available even when providers are down.
warm = await asyncio.to_thread(svc.warmup)
logger.info("Semsearch service started (warmup: %s)", warm)
```

### W.5 Decision points (resolved)

| Question | Decision | Rationale |
| ---------- | ---------- | ----------- |
| Config flag to disable warmup? | **No** | Free, fail-open, ~20 ms; a flag is surface without a use case |
| Fail startup if warmup fails? | **No** — warn and continue | Availability-first; degrades to today's behavior |
| `stats()` vs `SELECT 1` for DB step? | **`SELECT 1`** | stats() runs 4+ queries; overkill for pool liveness |
| Embedding API ping? | **No** (see W.2) | Cost + availability coupling |

---

## Phase F — Pool pre-fill inside `warmup()`

**Priority: LOW · Effort: small · Payoff: first concurrent burst doesn't serialize on connection setup**

### F.1 Problem

Warmup (W) opens exactly **one** DB connection. The search path's SQLAlchemy
pool holds up to 5 (+10 overflow) connections that are still created lazily —
so the first burst of concurrent requests serializes on connection setup
(~1–5 ms each, plus asyncpg backend spawn server-side).

### F.2 Design

Extend `warmup()` with a pre-fill step: check out and return ~5 connections
against `PGEngine`'s underlying SQLAlchemy engine so the pool is at its
`pool_size` before the first request.

Implementation notes (verified against installed `langchain-postgres`):

- `PGEngine` keeps its engine **private** (factory builds it via
  `create_async_engine` and stores it on the instance); there is no public
  accessor. At implementation time, locate the attribute on the instance
  (e.g. `vars(pg_engine)` → the `AsyncEngine`), then drive N concurrent
  `SELECT 1` checkouts through it (or its `sync_engine`) and return them.
- Guarded like every other warmup step: failure → warning, no raise.
- Cap pre-fill at `min(pool_size=5, 5)` — never exceed the configured pool.

```python
try:
    await-or-run N concurrent SELECT 1 via the async engine   # details at impl
    result["pool_prefill"] = True
except Exception as e:
    logger.warning("Warmup: pool pre-fill deferred (%s)", e)
```

(If async-engine plumbing proves awkward, acceptable fallback: N sequential
checkouts — the goal is pool membership, not speed of the warmup itself.)

### F.3 Files / tests / verification

- `src/semsearch/services/admin.py` — extend `warmup()`
- Tests: `warmup()` returns `pool_prefill: True`; SQLAlchemy pool
  `checkedin() >= 5` (or engine-equivalent) after warmup; still green with
  Phase C psycopg pool enabled/disabled (orthogonal pools).
- Live check: warmup → `pg_stat_activity` shows ≥5 backends for the DB.

---

## Cross-Phase Files Touched

| File | E | W | F |
| ------ | --- | --- | --- |
| `src/semsearch/embeddings.py` | ✔ | | |
| `src/semsearch/reranker.py` | ✔ | | |
| `src/semsearch/services/admin.py` | | ✔ | ✔ |
| `src/semsearch/server.py` | | ✔ | |
| `tests/test_embeddings.py` | ✔ | | |
| `tests/test_reranker_pooling.py` | ✔ | | |
| `tests/test_server.py` | | ✔ | ✔ |
| `docs/api-reference.md` | | ✔ (startup warms local resources) | |

---

## Verification (per phase)

1. Full suite green via `nix develop` (baseline: **132 passed, 1 skipped**).
2. Diagnostics zero errors.
3. **E**: manual timing — two searches 30 s apart; second request should no
   longer include a TLS handshake (compare request timings before/after).
4. **W**: first `/search` wall time before/after (local-init portion ~5–30 ms
   disappears); bogus `collection_name` startup → warning + `/health` OK.
5. **W/F negative proof**: serve with unreachable provider endpoints →
   startup completes instantly (no provider dependency).
6. **F**: `pg_stat_activity` shows pre-filled backends right after startup.

---

## Risks & Mitigations

| Risk | Phase | Mitigation |
| ------ | ------- | ------------ |
| Provider closes idle sockets server-side | E | httpx reconnects transparently; no correctness impact |
| Warmup masks real DB problems | W | Warning logged with exception at startup |
| Table absent at first boot | W | Fail-open by design; warms on first successful path |
| Private PGEngine attribute changes upstream | F | Guarded try/except → graceful skip; pin langchain-postgres range |
| Pre-built services double-warmed in tests | W/F | Idempotent (properties cache) |

---

## Success Metrics

| Metric | Baseline | Target |
| -------- | ---------- | -------- |
| Steady-state request after >10 s idle gap | +50–150 ms TLS re-handshake | ~0 ms (E) |
| First `/search` local-init portion | ~5–30 ms | ~0 ms (W) |
| First concurrent burst connection setups | serialized | pre-filled (F) |
| Startup outbound API requests | 0 | **0 (guaranteed by test)** |
| Startup wall-time added | — | ≤ 50 ms |
| Test suite | 132 passed, 1 skipped | ≥ 132 passed, 0 regressions |
| Diagnostics | 0 errors | 0 errors |

---

## Deferred (not in this plan)

- **Query-embedding LRU cache** (embed once per unique query, search by
  vector): feasible — verified `PGVectorStore.asimilarity_search_with_score_by_vector`
  exists — and would save real API money on repeated agent queries, but it's
  a search-path refactor (sync + async). Revisit when repeat-query patterns
  are observed.
- **Full result cache**: needs ingest/delete invalidation; defer.
- **Exposing HTTP limits as config**: rejected for now (Option B, see E.2).

---

## Appendix — Previously Completed Work (2026-08-22)

All four phases of the prior codebase-optimization plan landed and were
reviewed/verified:

- **A — Batched CASE B/C inserts** (benchmarked: BENCHMARKS.md)
- **B — search/asearch dedupe** (response-equivalence verified)
- **C — Opt-in connection pool** (`SEMSEARCH_POOL__*`, live-verified)
- **D — Service split** (facade + `services/` mixins; early by owner decision)

Final suite at hand-off: **132 passed, 1 skipped · 0 diagnostics ·
coverage 80.15%**.
