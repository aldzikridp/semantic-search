# TASK-003: Data Models

> **Phase**: 3 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-001
> **Blocks**: TASK-008, TASK-009

## Objective

Define all Pydantic models for the API contract in `models.py`. These models are the public-facing data structures returned by the service and CLI.

## File to Create

### `src/semsearch/models.py`

## Models (per SPEC §7.1)

### 1. `SearchResult`

```python
class SearchResult(BaseModel):
    id: str                               # langchain_id (TEXT, deterministic "{source}::{chunk_index}")
    content: str                          # chunk text
    score: float = Field(..., description="Cosine similarity in [-1, 1]; higher is better. "
                                          "Computed as 1.0 - cosine_distance")
    source: str | None = None             # top-level column
    chunk_index: int | None = None        # top-level column (0-based)
    page: int | None = None               # langchain_metadata.page (PDF only)
    row: int | None = None                # langchain_metadata.row (CSV only)
    doc_type: str | None = None            # langchain_metadata.doc_type
    metadata: dict[str, Any] = Field(default_factory=dict)  # full langchain_metadata blob
```

### 2. `DeleteResult`

```python
class DeleteResult(BaseModel):
    deleted_count: int                   # computed via SELECT COUNT(*) before delete
    filter: dict[str, Any]               # the filter dict passed to PGVectorStore.delete()
```

### 3. `IngestResult`

```python
class IngestResult(BaseModel):
    source: str
    chunks_added: int       # CASE C: newly embedded
    chunks_reused: int      # CASE A: content unchanged, reused embedding
    chunks_updated: int     # CASE B: content changed, re-embedded
    chunks_pruned: int = 0  # CASE D: stale tail chunks deleted
    ingested_at: datetime
```

### 4. `BatchAggregate`

```python
class BatchAggregate(BaseModel):
    """Aggregate counts across multiple files. No `source` field — it's per-batch."""
    chunks_added: int = 0
    chunks_reused: int = 0
    chunks_updated: int = 0
    chunks_pruned: int = 0
```

### 5. `BatchIngestResult`

```python
class BatchIngestResult(BaseModel):
    dir: str
    files_discovered: int
    files_skipped_unsupported: int
    files_attempted: int
    files_succeeded: int
    files_failed: int
    failed_files: list[dict] = Field(
        default_factory=list,
        description='List of {"path": str, "error": str} dicts',
    )
    aggregate: BatchAggregate
    elapsed_seconds: float
    pruned_sources: list[str] = Field(
        default_factory=list,
        description="Sources deleted by prune. Empty unless prune=True.",
    )
    pruned_chunks: int = Field(
        default=0,
        description="Total chunks deleted by prune. 0 if prune_dry_run=True.",
    )
```

## Critical Notes

1. **`SearchResult.score` is SIMILARITY, not distance** — The service converts `1.0 - cosine_distance`. Higher = more similar.

2. **`BatchAggregate` is separate from `IngestResult`** — Avoids the awkward `<batch:data/>` placeholder for `source`.

3. **`failed_files` contains `{"path": str, "error": str}` dicts** — Not tuples or custom objects.

4. **`deleted_count` in `DeleteResult` is computed BEFORE deletion** — Via `SELECT COUNT(*)` in the same transaction.

## Verification

- [ ] All models instantiate with required fields
- [ ] JSON serialization round-trips correctly
- [ ] `SearchResult` with `score=0.8731` serializes as `0.8731`
- [ ] `BatchIngestResult` with empty `failed_files=[]` serializes correctly
- [ ] Optional fields default to `None` or `[]` as specified
