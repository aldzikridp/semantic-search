# TASK-012: Core Service — Ingest Dir Method

> **Phase**: 8.6 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-009, TASK-011
> **Blocks**: TASK-014, TASK-019

## Objective

Implement the `ingest_dir()` method for recursive directory ingestion with optional pruning.

## File to Modify

### `src/semsearch/service.py` (add `ingest_dir` method)

## Implementation

### Method Signature

```python
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".csv", ".json")

def ingest_dir(
    self,
    dir_path: Path,
    *,
    glob: str = "**/*",
    exclude: list[str] | None = None,
    reembed_unchanged: bool = False,
    continue_on_error: bool = True,
    follow_symlinks: bool = False,
    prune: bool = False,
    prune_dry_run: bool = False,
) -> BatchIngestResult:
```

### Step-by-Step Flow

```python
def ingest_dir(self, dir_path: Path, **kwargs) -> BatchIngestResult:
    # Validate inputs
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    # Step 1: File discovery
    all_files = sorted(dir_path.glob(glob))

    # Filter files
    files = []
    skipped_unsupported = 0
    for f in all_files:
        # Skip directories
        if f.is_dir():
            continue
        # Skip hidden files
        if f.name.startswith("."):
            continue
        # Skip symlinks unless follow_symlinks
        if f.is_symlink() and not follow_symlinks:
            continue
        # Skip unsupported extensions
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped_unsupported += 1
            continue
        # Skip excluded patterns
        if exclude and any(fnmatch.fnmatch(str(f), pat) for pat in exclude):
            continue
        files.append(f)

    # Step 2: Ingest each file
    start_time = time.time()
    succeeded = 0
    failed = 0
    failed_files = []
    aggregate = BatchAggregate()
    ingested_sources = set()

    for file_path in files:
        try:
            result = self.ingest(file_path, reembed_unchanged=reembed_unchanged)
            succeeded += 1
            ingested_sources.add(str(file_path))
            aggregate.chunks_added += result.chunks_added
            aggregate.chunks_reused += result.chunks_reused
            aggregate.chunks_updated += result.chunks_updated
            aggregate.chunks_pruned += result.chunks_pruned
        except Exception as e:
            failed += 1
            failed_files.append({"path": str(file_path), "error": str(e)})
            if not continue_on_error:
                raise FileIngestError(f"Failed to ingest {file_path}: {e}") from e

    elapsed = time.time() - start_time

    # Step 3: Prune (if requested)
    pruned_sources = []
    pruned_chunks = 0

    if prune:
        # Query DB for all sources under dir_path
        dir_prefix = str(dir_path) + "/"
        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"SELECT DISTINCT source FROM {self.settings.collection_name} WHERE source LIKE :prefix"),
                {"prefix": dir_prefix + "%"}
            )
            db_sources = {row.source for row in result.fetchall()}

        # Find orphans
        orphan_sources = db_sources - ingested_sources

        for source in orphan_sources:
            pruned_sources.append(source)
            if not prune_dry_run:
                try:
                    delete_result = self.delete({"source": source})
                    pruned_chunks += delete_result.deleted_count
                except Exception as e:
                    logging.warning(f"Failed to prune {source}: {e}")

    return BatchIngestResult(
        dir=str(dir_path),
        files_discovered=len(all_files),
        files_skipped_unsupported=skipped_unsupported,
        files_attempted=len(files),
        files_succeeded=succeeded,
        files_failed=failed,
        failed_files=failed_files,
        aggregate=aggregate,
        elapsed_seconds=round(elapsed, 1),
        pruned_sources=pruned_sources,
        pruned_chunks=pruned_chunks,
    )
```

## Critical Implementation Details

### 1. File Discovery Rules

- Walk with `dir_path.glob(glob)` (default `**/*` = all files recursively)
- **Skip directories** (only process files)
- **Skip hidden files** (name starts with `.`)
- **Skip symlinks** unless `follow_symlinks=True`
- **Skip unsupported extensions** (not in `SUPPORTED_EXTENSIONS`)
- **Skip excluded patterns** (fnmatch)
- **Sort** for deterministic output

### 2. Per-file Isolation

Each file's ingest runs in its own transaction. A failure in one file does NOT roll back successful ingests of earlier files (when `continue_on_error=True`).

### 3. Prune Logic

Prune matches sources by **literal prefix**: `str(dir_path) + "/"`. This means:
- `ingest-dir data/` will only prune sources starting with `data/`
- Sources ingested from other directories are never touched
- Path consistency matters (see SPEC §9)

### 4. Prune Dry Run

When `prune_dry_run=True`:
- `pruned_sources` is populated (what WOULD be deleted)
- `pruned_chunks` is 0 (nothing actually deleted)
- Useful for inspecting before committing

### 5. Hidden Files and Symlinks

These are **independent** policies:
- Hidden file check: `f.name.startswith(".")`
- Symlink check: `f.is_symlink() and not follow_symlinks`
- A hidden symlink is skipped for both reasons

## Behavior Matrix

| Scenario | Plain `ingest-dir` | `--prune` | `--prune --dry-run` |
|----------|-------------------|-----------|---------------------|
| Unchanged dir | 0 embed calls | 0 embed calls, 0 deletes | 0 embed calls, 0 deletes |
| New file added | CASE C embeds | same + 0 deletes | same + 0 deletes |
| File edited | CASE B re-embeds | same + 0 deletes | same + 0 deletes |
| File deleted | Orphans remain | Orphans deleted | Orphans listed |
| File renamed | New added, old orphaned | New added, old deleted | New added, old listed |

## Verification (Integration Tests)

- [ ] I-29: Mixed files → correct `files_discovered`, `files_skipped_unsupported`, `files_attempted`
- [ ] I-30: `--glob "**/*.pdf"` → only PDFs ingested
- [ ] I-31: `--exclude "*/draft/*"` → draft subdir skipped
- [ ] I-32: Failing file + `continue_on_error=True` → `files_failed == 1`, others succeed
- [ ] I-33: Failing file + `--no-continue-on-error` → `FileIngestError` raised
- [ ] I-34: Hidden files skipped
- [ ] I-35: Symlinks skipped by default, followed with `--follow-symlinks`
- [ ] I-36: Idempotent re-run → zero embed calls
- [ ] I-37: New file added → only new file embedded
- [ ] I-38: `--prune` deletes orphans
- [ ] I-39: Without `--prune` → orphans remain
- [ ] I-40: `--prune --dry-run` → lists but doesn't delete
- [ ] I-41: File rename → old pruned, new embedded
- [ ] I-42: File move → same as rename
- [ ] I-43: Prune only affects `dir_path/` prefix
