# API Reference

Python API for programmatic use.

## Quick Start

```python
from semsearch.config import Settings
from semsearch.service import SemanticSearchService

# Load settings from .env
settings = Settings()

# Create service
with SemanticSearchService.from_settings(settings) as svc:
    # Initialize schema
    svc.init_schema()

    # Ingest a file
    result = svc.ingest(Path("docs/readme.md"))
    print(result.chunks_added)

    # Search
    results = svc.search("how to configure", k=5)
    for r in results:
        print(f"{r.score:.3f}: {r.content[:100]}")

    # Delete
    result = svc.delete({"source": "docs/readme.md"})
    print(result.deleted_count)
```

## Settings

### `Settings`

```python
from semsearch.config import Settings

settings = Settings()
# Loads from environment variables and .env file
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `database_url` | `str` | `postgresql+psycopg://...` | PostgreSQL connection string |
| `collection_name` | `str` | `semsearch_chunks` | Table name |
| `embedding_provider` | `EmbeddingProviderConfig` | HuggingFace | Provider config |
| `chunk_size` | `int` | `1000` | Chunk size in characters |
| `chunk_overlap` | `int` | `200` | Overlap between chunks |
| `default_k` | `int` | `5` | Default top-k for search |
| `recreate_collection_on_init` | `bool` | `False` | Safety flag |

### `EmbeddingProviderConfig`

```python
from semsearch.config import EmbeddingProviderConfig

config = EmbeddingProviderConfig(
    type="openai",
    model="text-embedding-3-small",
    api_key=SecretStr("sk-..."),
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `Literal[...]` | — | Provider type |
| `model` | `str` | — | Model name |
| `api_key` | `SecretStr \| None` | `None` | API key |
| `base_url` | `str \| None` | `None` | Custom base URL |
| `provider_order` | `list[str] \| None` | `None` | OpenRouter routing |
| `provider_allow_fallbacks` | `bool \| None` | `None` | OpenRouter fallbacks |
| `provider_ignore` | `list[str] \| None` | `None` | OpenRouter ignore |
| `provider_only` | `list[str] \| None` | `None` | OpenRouter whitelist |
| `provider_require_parameters` | `bool` | `False` | OpenRouter flag |
| `provider_data_collection` | `str \| None` | `None` | Data collection policy |
| `provider_max_price` | `dict \| None` | `None` | Price cap |

## SemanticSearchService

### Creating a Service

```python
from semsearch.service import SemanticSearchService
from semsearch.config import Settings

settings = Settings()
svc = SemanticSearchService.from_settings(settings)
```

Or with context manager:

```python
with SemanticSearchService.from_settings(settings) as svc:
    # Use service
    pass
# Automatically closed
```

### `init_schema(recreate=False)`

Create the database table.

```python
svc.init_schema()              # Create if not exists
svc.init_schema(recreate=True) # Drop and recreate
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `recreate` | `bool` | `False` | Drop table first |

**Raises:** `SchemaMismatchError` if table exists with wrong vector dimension.

### `ingest(path, reembed_unchanged=False)`

Ingest a single file.

```python
result = svc.ingest(Path("docs/readme.md"))
print(result.chunks_added)
print(result.chunks_reused)
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `Path` | — | File to ingest |
| `reembed_unchanged` | `bool` | `False` | Force re-embed all chunks |

**Returns:** `IngestResult`

**Raises:** `FileIngestError` on failure.

### `ingest_dir(dir_path, **kwargs)`

Ingest all supported files in a directory.

```python
result = svc.ingest_dir(
    Path("docs/"),
    glob="**/*.md",
    exclude=["*/draft/*"],
    prune=True,
)
print(result.files_succeeded)
print(result.aggregate.chunks_added)
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `dir_path` | `Path` | — | Directory to walk |
| `glob` | `str` | `"**/*"` | Glob pattern |
| `exclude` | `list[str] \| None` | `None` | fnmatch patterns to skip |
| `reembed_unchanged` | `bool` | `False` | Force re-embed |
| `continue_on_error` | `bool` | `True` | Continue on failure |
| `follow_symlinks` | `bool` | `False` | Follow symlinks |
| `prune` | `bool` | `False` | Delete orphaned chunks |
| `prune_dry_run` | `bool` | `False` | Preview prune |

**Returns:** `BatchIngestResult`

**Raises:** `FileIngestError` if `continue_on_error=False` and a file fails.

### `search(query, k=None, filter=None)`

Cosine similarity search.

```python
results = svc.search("how to configure", k=5)
for r in results:
    print(f"{r.score:.3f}: {r.source}")
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | — | Search query |
| `k` | `int \| None` | `None` | Top-k (default from settings) |
| `filter` | `dict \| None` | `None` | PGVectorStore filter |

**Returns:** `list[SearchResult]` sorted by score DESC.

**Raises:** `ValueError` if k out of range, `SearchError` on failure.

### `delete(filter)`

Delete chunks by filter.

```python
result = svc.delete({"source": "docs/old.md"})
print(result.deleted_count)
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `filter` | `dict` | — | Filter dict (empty = delete all) |

**Returns:** `DeleteResult`

**Raises:** `DeleteError` on failure.

### `stats()`

Get table statistics.

```python
stats = svc.stats()
print(stats["chunk_count"])
print(stats["source_count"])
```

**Returns:** `dict` with keys:
- `table`: Table name
- `embedding_provider`: Provider type
- `embedding_dim`: Vector dimension
- `chunk_count`: Total chunks
- `source_count`: Distinct sources
- `sources_by_count`: Top 20 sources by chunk count

### `reingest(path)`

Delete + ingest in one step.

```python
result = svc.reingest(Path("docs/readme.md"))
print(result.chunks_added)  # All chunks are CASE C
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `Path` | — | File to reingest |

**Returns:** `IngestResult` with `chunks_added == total chunks`.

## Models

### `SearchResult`

```python
class SearchResult(BaseModel):
    id: str                    # "{source}::{chunk_index}"
    content: str               # Chunk text
    score: float               # 1.0 - cosine_distance
    source: str | None         # File path
    chunk_index: int | None    # 0-based position
    page: int | None           # PDF page number
    row: int | None            # CSV row number
    doc_type: str | None       # "text", "pdf", "csv", "json"
    metadata: dict[str, Any]   # Full metadata blob
```

### `IngestResult`

```python
class IngestResult(BaseModel):
    source: str                # File path
    chunks_added: int          # CASE C count
    chunks_reused: int         # CASE A count
    chunks_updated: int        # CASE B count
    chunks_pruned: int         # CASE D count
    ingested_at: datetime      # Timestamp
```

### `BatchIngestResult`

```python
class BatchIngestResult(BaseModel):
    dir: str                   # Directory path
    files_discovered: int      # All files found
    files_skipped_unsupported: int  # Unsupported extensions
    files_attempted: int       # Files tried
    files_succeeded: int       # Successful
    files_failed: int          # Failed
    failed_files: list[dict]   # Error details
    aggregate: BatchAggregate  # Sum of chunk counts
    elapsed_seconds: float     # Duration
    pruned_sources: list[str]  # Pruned sources
    pruned_chunks: int         # Pruned chunk count
```

### `DeleteResult`

```python
class DeleteResult(BaseModel):
    deleted_count: int         # Chunks deleted
    filter: dict[str, Any]     # Filter used
```

## Errors

```python
from semsearch.errors import (
    SemSearchError,        # Base exception
    FileIngestError,       # Ingest failure
    SearchError,           # Search failure
    DeleteError,           # Delete failure
    SchemaMismatchError,   # Wrong vector dimension
    ProviderConfigError,   # Missing credentials
)
```

## Examples

### Custom Settings

```python
from semsearch.config import Settings, EmbeddingProviderConfig
from pydantic import SecretStr

settings = Settings(
    database_url="postgresql://user:pass@host:5432/db",
    embedding_provider=EmbeddingProviderConfig(
        type="openai",
        model="text-embedding-3-small",
        api_key=SecretStr("sk-..."),
    ),
    chunk_size=500,
    chunk_overlap=100,
)
```

### Search with Filter

```python
# Search only PDFs
results = svc.search("query", filter={"doc_type": "pdf"})

# Search with prefix
results = svc.search("query", filter={"source": {"$ilike": "docs/%"}})

# Combined filter
results = svc.search("query", filter={
    "$and": [
        {"doc_type": "text"},
        {"source": {"$ilike": "TASKS/%"}}
    ]
})
```

### Batch Operations

```python
# Ingest directory with prune
result = svc.ingest_dir(
    Path("docs/"),
    glob="**/*.md",
    exclude=["*/draft/*"],
    prune=True,
)

# Check results
if result.files_failed > 0:
    for f in result.failed_files:
        print(f"Failed: {f['path']}: {f['error']}")
```

### Error Handling

```python
from semsearch.errors import FileIngestError, SearchError

try:
    result = svc.ingest(Path("bad.pdf"))
except FileIngestError as e:
    print(f"Ingest failed: {e}")

try:
    results = svc.search("query")
except SearchError as e:
    print(f"Search failed: {e}")
```
