# TASK-013: Core Service — Reingest Method

> **Phase**: 8.7 | **Priority**: High | **Status**: ✅ Done
> **Depends on**: TASK-009, TASK-011
> **Blocks**: TASK-014

## Objective

Implement the `reingest()` method — delete + ingest in one step for a single file.

## File to Modify

### `src/semsearch/service.py` (add `reingest` method)

## Implementation

### Method Signature

```python
def reingest(self, path: Path) -> IngestResult:
```

### Implementation

```python
def reingest(self, path: Path) -> IngestResult:
    """
    Delete all chunks for the source, then ingest fresh.

    NOT atomic: delete commits in its own transaction, then ingest runs
    in a separate transaction. If ingest fails after delete succeeded,
    the file's chunks are gone but not re-ingested.

    Args:
        path: file to reingest.

    Returns:
        IngestResult with chunks_added == total chunks (all are CASE C).

    Raises:
        FileIngestError: ingest failed after delete succeeded.
    """
    source = str(path)

    # Step 1: Delete all chunks for this source
    self.delete({"source": source})

    # Step 2: Ingest fresh (force re-embed since we just deleted everything)
    return self.ingest(path, reembed_unchanged=True)
```

## Critical Notes

### 1. NOT Atomic

Per SPEC §8.7, there's a design gap: `delete()` goes through `PGVectorStore.delete()` while `ingest()` uses a separate `engine.begin()` block. These are two different transaction paths that don't share a connection/transaction.

**Chosen approach**: "delete commits, then ingest runs in its own transaction." If ingest fails after delete, the file's chunks are gone but not re-ingested. This is documented clearly.

**Why not make it truly atomic?** Would require:
- Opening one `engine.begin()` block
- Passing the same connection through both delete and ingest
- Bypassing `PGVectorStore.delete()`'s own transaction handling
- Significant complexity for a CLI tool where the user can just re-run

### 2. Always Uses `reembed_unchanged=True`

Since we just deleted all chunks, there's nothing to reuse. Every chunk will be CASE C (new).

### 3. Result Interpretation

The returned `IngestResult` will have:
- `chunks_added` = total chunks (all are CASE C)
- `chunks_reused` = 0 (nothing to reuse)
- `chunks_updated` = 0 (nothing to update)
- `chunks_pruned` = 0 (nothing to prune)

## Verification

- [ ] `reingest(file)` produces same result as `delete + ingest`
- [ ] All chunks are CASE C (chunks_added = total)
- [ ] Zero chunks_reused and chunks_updated
- [ ] Works correctly when file hasn't changed (still re-embeds everything)
- [ ] Works correctly when file has changed (old content gone, new content ingested)
