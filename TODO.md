# TODO — Task Tracking

> Phase 1 Performance Optimization based on `semantic-search-performance-audit.md`

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Pending |
| 🔵 | In Progress |
| ✅ | Complete |
| ❌ | Blocked |

## Performance Phase 1 — Quick Wins + HTTP Server

| # | Task | Status | Depends On | Blocks | File |
|---|------|--------|------------|--------|------|
| 024 | Benchmark Harness | ✅ | — | 025, 026, 027, 028, 029 | `TASKS/TASK-024-benchmark-harness.md` |
| 025 | Cache Vector Size + DB Read in stats() | ✅ | 024 | 029 | `TASKS/TASK-025-cache-vector-size.md` |
| 026 | Expose HNSW Tuning in Config | ✅ | 024 | 029 | `TASKS/TASK-026-hnsw-tuning.md` |
| 027 | Persistent httpx.Client in Reranker with Retry | ✅ | 024 | 029 | `TASKS/TASK-027-reranker-httpx-pooling.md` |
| 028 | FastAPI HTTP Server (`semsearch serve`) | ✅ | 024 | 029 | `TASKS/TASK-028-fastapi-server.md` |
| 029 | Before/After Performance Benchmarks | ✅ | 025, 026, 027, 028 | — | `TASKS/TASK-029-before-after-benchmarks.md` |

## Dependency Graph

```
TASK-024 (Benchmark Harness)
    ├── TASK-025 (Cache Vector Size)
    ├── TASK-026 (HNSW Tuning)
    ├── TASK-027 (Reranker httpx)
    └── TASK-028 (FastAPI Server)
            │
            └── TASK-029 (Before/After Benchmarks)
```

## Implementation Order

```
1. TASK-024: Create benchmark harness
2. TASK-025: Cache vector size (simplest change)
3. TASK-027: Reranker httpx pooling
4. TASK-026: HNSW tuning
5. TASK-028: FastAPI server
6. TASK-029: Run before/after benchmarks
```

## Summary

| Metric | Value |
|--------|-------|
| Total tasks | 6 |
| Completed | 6 |
| In Progress | 0 |
| Pending | 0 |
| Blocked | 0 |

## Related Files

- `PLAN.md` — Full implementation plan
- `semantic-search-performance-audit.md` — Original audit findings
- `AGENTS.md` — Agent instructions (updated with new tasks)
