# TODO — Semantic Search Implementation

> **Plan**: [PLAN.md](PLAN.md) | **Spec**: [SPEC.md](SPEC.md)
> **Tasks Directory**: [TASKS/](TASKS/)

## Design Decisions

- **No local embeddings** — HuggingFace local embeddings removed; rely entirely on API providers (OpenAI, OpenRouter, OpenAI-compatible, Ollama).

## Status Overview

| Status | Count |
|--------|-------|
| ✅ Done | 22 |
| 🔄 In Progress | 0 |
| ⏳ Not Started | 1 |

---

## Task List

### Phase 1: Project Scaffolding
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-001](TASKS/TASK-001-project-scaffolding.md) | Project Scaffolding & Dependencies | ✅ | — | 002-009 |

### Phase 2-3: Configuration & Models
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-002](TASKS/TASK-002-configuration-layer.md) | Configuration Layer (config.py, errors.py) | ✅ | 001 | 006-009 |
| [TASK-003](TASKS/TASK-003-data-models.md) | Data Models (models.py) | ✅ | 001 | 008-009 |

### Phase 4-5: Document Processing
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-004](TASKS/TASK-004-document-loaders.md) | Document Loaders (loaders.py) | ✅ | 001 | 008 |
| [TASK-005](TASKS/TASK-005-text-splitter.md) | Text Splitter (splitter.py) | ✅ | 001 | 008 |

### Phase 6-7: Embeddings & Database
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-006](TASKS/TASK-006-embeddings-factory.md) | Embeddings Factory (embeddings.py) | ✅ | 002 | 007-008 |
| [TASK-007](TASKS/TASK-007-database-store.md) | Database Store (store.py) | ✅ | 002, 001 | 008 |

### Phase 8: Core Service (Split into Sub-tasks)
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-008](TASKS/TASK-008-service-skeleton.md) | Service Skeleton & Stats | ✅ | 002-007 | 009-014 |
| [TASK-009](TASKS/TASK-009-service-ingest.md) | Service — Ingest (Write Path) | ✅ | 008 | 012-014 |
| [TASK-010](TASKS/TASK-010-service-search.md) | Service — Search (Read Path) | ✅ | 008 | 014 |
| [TASK-011](TASKS/TASK-011-service-delete.md) | Service — Delete | ✅ | 008 | 012-014 |
| [TASK-012](TASKS/TASK-012-service-ingest-dir.md) | Service — Ingest Dir (Batch + Prune) | ✅ | 009, 011 | 014 |
| [TASK-013](TASKS/TASK-013-service-reingest.md) | Service — Reingest | ✅ | 009, 011 | 014 |

### Phase 9: CLI
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-014](TASKS/TASK-014-cli.md) | CLI Implementation (cli.py) | ✅ | 008-013 | 021 |

### Phase 10: Tests
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-015](TASKS/TASK-015-unit-tests.md) | Unit Tests — Loaders & Embeddings | ✅ | 004, 006 | 022 |
| [TASK-016](TASKS/TASK-016-integration-tests-ingest.md) | Integration Tests — Ingest | ✅ | 008, 009 | 022 |
| [TASK-017](TASKS/TASK-017-integration-tests-search.md) | Integration Tests — Search | ✅ | 008, 010 | 022 |
| [TASK-018](TASKS/TASK-018-integration-tests-delete.md) | Integration Tests — Delete | ✅ | 008, 011 | 022 |
| [TASK-019](TASKS/TASK-019-integration-tests-ingest-dir.md) | Integration Tests — Ingest Dir | ✅ | 008, 012 | 022 |
| [TASK-020](TASKS/TASK-020-integration-tests-provider.md) | Integration Tests — Provider & Edge Cases | ⏳ | 008, 006 | 022 |
| [TASK-021](TASKS/TASK-021-cli-tests.md) | CLI Tests | ✅ | 014 | 022 |

### Phase 11-12: Documentation & Verification
| # | Task | Status | Depends On | Blocks |
|---|------|--------|------------|--------|
| [TASK-022](TASKS/TASK-022-documentation.md) | Documentation & Polish | ⏳ | 001-021 | 023 |
| [TASK-023](TASKS/TASK-023-verification-acceptance.md) | Verification & Acceptance | ✅ | 001-022 | — |

---

## Dependency Graph

```
TASK-001 (Scaffolding)
  ├─→ TASK-002 (Config) ─────────┬─→ TASK-006 (Embeddings) ──┬─→ TASK-007 (Store) ──┐
  ├─→ TASK-003 (Models)          │                            │                       │
  ├─→ TASK-004 (Loaders)         │                            │                       │
  ├─→ TASK-005 (Splitter)        │                            │                       │
  │                               │                            │                       │
  │                               └────────────────────────────┴───────────────────────┤
  │                                                                                    │
  │                                                                                    ▼
  │                                                              TASK-008 (Service Skeleton)
  │                                                                ├─→ TASK-009 (Ingest)
  │                                                                ├─→ TASK-010 (Search)
  │                                                                └─→ TASK-011 (Delete)
  │                                                                      │       │
  │                                                                      ▼       ▼
  │                                                              TASK-012 (Ingest Dir)
  │                                                              TASK-013 (Reingest)
  │                                                                      │
  │                                                                      ▼
  │                                                              TASK-014 (CLI)
  │                                                                      │
  ▼                                                                      ▼
TASK-015 (Unit Tests)                                    TASK-021 (CLI Tests)
TASK-016 (Ingest Tests)                                          │
TASK-017 (Search Tests)                                          │
TASK-018 (Delete Tests)                                          │
TASK-019 (Ingest Dir Tests)                                      │
TASK-020 (Provider Tests)                                        │
  │                                                               │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
                      TASK-022 (Documentation)
                              │
                              ▼
                      TASK-023 (Verification)
```

---

## Critical Path

The critical path (longest dependency chain) is:

```
TASK-001 → TASK-002 → TASK-006 → TASK-007 → TASK-008 → TASK-009 → TASK-012 → TASK-014 → TASK-021 → TASK-022 → TASK-023
```

**Estimated critical path length**: ~5 working days

---

## How to Use This TODO

1. **Start with TASK-001** — No dependencies, creates project structure
2. **Follow the dependency graph** — Each task lists what it depends on and what it blocks
3. **Update status** — Change ⏳ to 🔄 when starting, ✅ when complete
4. **Check verification criteria** — Each task file has a "Verification" section
5. **Run tests frequently** — After each service task, run relevant tests
6. **Final sign-off** — TASK-023 has the complete acceptance checklist

---

## Quick Reference

### Files Created by Each Task

| Task | Primary File(s) |
|------|-----------------|
| 001 | `requirements.txt`, `pyproject.toml`, `.env.example`, `scripts/init_db.sql` |
| 002 | `src/semsearch/config.py`, `src/semsearch/errors.py` |
| 003 | `src/semsearch/models.py` |
| 004 | `src/semsearch/loaders.py` |
| 005 | `src/semsearch/splitter.py` |
| 006 | `src/semsearch/embeddings.py` |
| 007 | `src/semsearch/store.py` |
| 008 | `src/semsearch/service.py` (skeleton) |
| 009 | `src/semsearch/service.py` (add ingest) |
| 010 | `src/semsearch/service.py` (add search) |
| 011 | `src/semsearch/service.py` (add delete) |
| 012 | `src/semsearch/service.py` (add ingest_dir) |
| 013 | `src/semsearch/service.py` (add reingest) |
| 014 | `src/semsearch/cli.py` |
| 015 | `tests/test_loaders.py`, `tests/test_embeddings.py` |
| 016 | `tests/conftest.py`, `tests/test_service_ingest.py` |
| 017 | `tests/test_service_search.py` |
| 018 | `tests/test_service_delete.py` |
| 019 | `tests/test_service_ingest_dir.py` |
| 020 | `tests/test_service_provider.py` |
| 021 | `tests/test_cli.py` |
| 022 | `README.md` |
| 023 | (verification only) |

### Test Coverage Target

| Test File | Tests | Priority |
|-----------|-------|----------|
| `test_loaders.py` | U-1 to U-4 | High |
| `test_embeddings.py` | U-5 to U-9 | High |
| `test_service_ingest.py` | I-1, I-2, I-3, I-9, I-9b, I-10, I-10b, I-10c, I-10d, I-10e, I-11 | Critical |
| `test_service_search.py` | I-4, I-5, I-15 to I-20 | Critical |
| `test_service_delete.py` | I-6, I-7, I-8, I-12, I-13 | Critical |
| `test_service_ingest_dir.py` | I-29 to I-43 | High |
| `test_service_provider.py` | I-14, I-21 to I-28c | Medium |
| `test_cli.py` | C-1 to C-6 | Medium |

**Total**: 52 test cases | **Target**: ≥85% line coverage
