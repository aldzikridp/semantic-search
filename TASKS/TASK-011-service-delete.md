# TASK-011: Core Service — Delete Method

> **Phase**: 8.5 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-008
> **Blocks**: TASK-012, TASK-013, TASK-014, TASK-018

## Objective

Implement the `delete()` method that removes chunks by filter, with special handling for empty filter (delete all).

## File to Modify

### `src/semsearch/service.py` (add `delete` method)

## Implementation

### Method Signature

```python
def delete(self, filter: dict) -> DeleteResult:
```

### Step-by-Step Flow

```python
def delete(self, filter: dict) -> DeleteResult:
    table = self.settings.collection_name

    try:
        with self.engine.begin() as conn:
            if not filter:
                # SPECIAL CASE: Empty filter = delete everything
                # PGVectorStore.delete(filter={}) may return False without deleting
                # Bypass with direct SQL

                # Count before
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()

                # Delete all
                conn.execute(text(f"DELETE FROM {table}"))
            else:
                # Non-empty filter: count before, delete, return count
                # Approach: count before + after in same transaction

                # Count before
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count_before = result.scalar()

                # Delegate to PGVectorStore
                self.store.delete(filter=filter)

                # Count after
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count_after = result.scalar()

                count = count_before - count_after

        return DeleteResult(deleted_count=count, filter=filter)

    except SemSearchError:
        raise
    except Exception as e:
        raise DeleteError(f"Delete failed: {e}") from e
```

## Critical Implementation Details

### 1. Empty Filter Special Case

`PGVectorStore.delete(filter=None)` or `filter={}` may return `False` without deleting anything. The spec mandates bypassing `PGVectorStore` for empty filters:

```python
if not filter:
    # Direct SQL bypass
    conn.execute(text(f"DELETE FROM {table}"))
```

The CLI wraps this with `--all --yes` confirmation.

### 2. deleted_count Computation

The spec says to compute `deleted_count` via `SELECT COUNT(*) ... WHERE <filter>` in the same transaction. However, translating the rich filter dict to SQL is complex and duplicates `PGVectorStore` internals.

**Recommended approach (Approach C from spec)**: Count before and after in the same transaction.

```python
# Count before
count_before = SELECT COUNT(*) FROM table

# Delete via PGVectorStore
store.delete(filter=filter)

# Count after
count_after = SELECT COUNT(*) FROM table

deleted_count = count_before - count_after
```

This is simple, correct, and doesn't require reimplementing filter translation.

### 3. Transaction Handling

The `SELECT COUNT(*)` + `PGVectorStore.delete()` + `SELECT COUNT(*)` should be in one transaction for consistency. However, `PGVectorStore.delete()` manages its own connection/transaction internally.

**Resolution**: Accept that the count is best-effort. The delete operation itself is atomic (PGVectorStore handles it). The count may have a small window of inconsistency, but for a single-user CLI tool, this is acceptable.

Alternative: Execute the delete in our own transaction using raw SQL, but this requires reimplementing filter translation.

### 4. Error Wrapping

```python
except SemSearchError:
    raise  # Re-raise semsearch errors as-is
except Exception as e:
    raise DeleteError(f"Delete failed: {e}") from e  # Wrap other errors
```

## Verification (Integration Tests)

- [ ] I-6: Delete by exact source → `deleted_count` correct, only that file's chunks removed
- [ ] I-7: Delete by metadata filter → correct chunks removed
- [ ] I-8: Delete with empty filter `{}` → all rows deleted, `deleted_count` correct
- [ ] I-12: `delete --all` without `--yes` blocked in CLI
- [ ] I-13: `delete --all --yes` wipes table
- [ ] `DeleteError` raised on DB failures
- [ ] Non-empty filter returns correct `deleted_count`
