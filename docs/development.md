# Development Guide

This guide covers contributing to semsearch: setup, testing, debugging, and code conventions.

## Setup

### NixOS

```bash
git clone <repo-url> && cd semantic-search
nix develop  # installs everything
```

### Other Systems

```bash
git clone <repo-url> && cd semantic-search
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/semsearch/
├── __init__.py       # Package marker + version
├── cli.py            # Typer CLI (8 commands)
├── config.py         # Settings + EmbeddingProviderConfig
├── embeddings.py     # Provider dispatch (openai, ollama, openrouter, openai_compatible)
├── errors.py         # Exception hierarchy
├── loaders.py        # File type dispatch (txt, pdf, csv, json)
├── models.py         # Pydantic models
├── service.py        # SemanticSearchService (main orchestration)
├── splitter.py       # Text chunking
└── store.py          # PostgreSQL + pgvector

tests/
├── conftest.py       # Shared fixtures (MockEmbeddings, service)
├── test_loaders.py   # Loader unit tests
├── test_embeddings.py # Embeddings tests
├── test_service_*.py # Integration tests
└── test_cli.py       # CLI tests

docs/                 # This documentation
```

## Running Tests

### All Tests

```bash
# With local PostgreSQL
export TEST_DATABASE_URL="postgresql://semsearch:test@/semsearch?host=/path/to/.pgsocket"
pytest -v

# With Docker
export TEST_DATABASE_URL="postgresql://test:test@localhost:5432/test"
pytest -v
```

### Specific Tests

```bash
# Unit tests only (no DB)
pytest tests/test_loaders.py tests/test_embeddings.py -v

# Integration tests
pytest tests/test_service_ingest.py -v

# CLI tests
pytest tests/test_cli.py -v
```

### With Coverage

```bash
pytest --cov=semsearch --cov-report=term-missing
```

## Test Fixtures

### MockEmbeddings

Deterministic mock embeddings (no API calls):

```python
@pytest.fixture
def mock_embeddings():
    return MockEmbeddings(dim=128)
```

### Service

Fresh service with mock embeddings:

```python
@pytest.fixture
def service(pg_url, mock_embeddings):
    # Creates fresh table, returns service
    ...
```

### Sample Files

```python
@pytest.fixture
def sample_txt(tmp_path):
    # Returns Path to ~3 chunk text file
    ...

@pytest.fixture
def sample_csv(tmp_path):
    # Returns Path to 10-row CSV
    ...
```

## Code Conventions

### Imports

```python
# Standard library
import hashlib
from pathlib import Path
from typing import Any, Self

# Third-party
from pydantic import BaseModel
from sqlalchemy import text

# Local
from semsearch.config import Settings
from semsearch.errors import FileIngestError
```

Do **not** use `from __future__ import annotations` — annotations are
eagerly evaluated (Python ≥3.11 target). For forward references, quote the
name (`-> "ClassName"`) or use `typing.Self` for methods returning the
enclosing class.

### Type Hints

All public functions must have type hints:

```python
def ingest(self, path: Path, *, reembed_unchanged: bool = False) -> IngestResult:
    ...
```

### Docstrings

Google style:

```python
def search(self, query: str, k: int | None = None) -> list[SearchResult]:
    """Cosine similarity search over the chunks table.

    Args:
        query: Free-text query.
        k: Top-k results. Defaults to settings.default_k.

    Returns:
        List of SearchResult sorted by score DESC.

    Raises:
        ValueError: k out of range.
        SearchError: Search failed.
    """
```

### Error Handling

```python
from semsearch.errors import FileIngestError

try:
    result = svc.ingest(path)
except FileIngestError as e:
    logger.error(f"Failed to ingest {path}: {e}")
    raise
```

## Debugging

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check SQL Queries

```python
# In service.py, add logging
import logging
logger = logging.getLogger(__name__)

logger.debug(f"Executing: {query}")
```

### Inspect Database

```bash
# Connect to database
psql -d semsearch

# Check table
\d semsearch_chunks

# Count chunks
SELECT COUNT(*) FROM semsearch_chunks;

# Check sources
SELECT source, COUNT(*) FROM semsearch_chunks GROUP BY source;

# Check metadata
SELECT langchain_metadata FROM semsearch_chunks LIMIT 5;
```

### Debug Embeddings

```python
from semsearch.config import Settings
from semsearch.embeddings import build_embedder

settings = Settings()
embedder = build_embedder(settings)

# Test embedding
result = embedder.embed_query("test")
print(f"Dimension: {len(result)}")
print(f"First 5 values: {result[:5]}")
```

## Adding a New Provider

1. **Add to config.py:**

```python
# In EmbeddingProviderConfig.type Literal
type: Literal["openai", "ollama", "openai_compatible", "openrouter", "new_provider"]
```

1. **Add to embeddings.py:**

```python
if cfg.type == "new_provider":
    from new_package import NewEmbeddings
    return NewEmbeddings(
        model=cfg.model,
        api_key=cfg.api_key.get_secret_value(),
    )
```

1. **Add to pyproject.toml:**

```toml
[project.optional-dependencies]
new_provider = [
    "new-package>=1.0.0",
]
```

1. **Add tests:**

```python
def test_new_provider_builds():
    settings = Settings(embedding_provider=EmbeddingProviderConfig(
        type="new_provider",
        model="test",
    ))
    embedder = build_embedder(settings)
    assert isinstance(embedder, NewEmbeddings)
```

1. **Update docs:**

- `docs/providers.md` — Add provider section
- `docs/configuration.md` — Add env var example
- `.env.example` — Add commented example

## Adding a New Loader

1. **Add to loaders.py:**

```python
_DOC_TYPE_BY_EXT = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "json",
    ".new": "new_type",  # Add here
}
```

1. **Add to pick_loader:**

```python
elif suffix == ".new":
    from new_package import NewLoader
    loader = NewLoader(str(path))
```

1. **Add tests:**

```python
def test_new_loader(tmp_path):
    f = tmp_path / "test.new"
    f.write_text("content")
    loader = pick_loader(f)
    docs = loader()
    assert len(docs) > 0
```

## Pull Request Checklist

- [ ] Code follows conventions (type hints, docstrings)
- [ ] Tests pass (`pytest -v`)
- [ ] Coverage ≥85% (`pytest --cov`)
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Error handling added
- [ ] Logging added for debugging

## Common Issues

### "No module named 'langchain_openai'"

Install provider extras:

```bash
pip install -e ".[openai]"
```

### "libstdc++.so.6: cannot open shared object file"

Missing system library. Install via:

```bash
# Ubuntu/Debian
sudo apt install libstdc++6

# NixOS
# Add to flake.nix buildInputs
```

### "libz.so.1: cannot open shared object file"

Missing zlib. Install via:

```bash
# Ubuntu/Debian
sudo apt install zlib1g

# NixOS
# Add pkgs.zlib to flake.nix buildInputs
```

### Tests Skip with "Docker not available"

Integration tests need Docker for testcontainers:

```bash
# Start Docker
sudo systemctl start docker

# Or use local PostgreSQL
export TEST_DATABASE_URL="postgresql://semsearch:test@/semsearch?host=/path/to/.pgsocket"
```

## Performance Tips

### Batch Embeddings

The service batches embeddings for efficiency:

```python
# Bad: One API call per chunk
for chunk in chunks:
    embed(chunk)

# Good: Single API call for all chunks
embed_documents([c.page_content for c in chunks])
```

### Connection Reuse in Batch Operations

`ingest_dir()` reuses a single connection for all files:

```python
# This is what ingest_dir() does internally:
conn = self._get_conn()
try:
    for file in files:
        self.ingest(file, conn=conn)  # Reuse connection
finally:
    conn.close()
```

This reduces TCP + auth overhead from 2N to 1 for N files.

### Connection Reuse

Service methods accept an optional `conn` parameter for connection reuse:

```python
# Standalone call — method creates and closes its own connection
result = svc.ingest(Path("file.txt"))  # 1 connection

# Batch — one connection for all files
conn = svc._get_conn()
try:
    for path in files:
        svc.ingest(path, conn=conn)  # Reuses same connection
finally:
    conn.close()
```

`ingest_dir()` and `reingest()` use this pattern automatically:

- `ingest_dir(N files)` → 1 connection for the batch (not 2N)
- `reingest()` → 1 connection for delete + ingest (not 2)

For standalone calls (`ingest()`, `delete()`, `stats()`), each call creates its own connection. This is fine for CLI single-command usage.

### Lazy Store

PGVectorStore is lazy-initialized:

```python
# Store created only when first accessed
@property
def store(self):
    if self._store is None:
        self._store = build_store(...)
    return self._store
```
