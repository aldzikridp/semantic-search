# TASK-016: Integration Tests — Ingest

> **Phase**: 10.3 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-008, TASK-009
> **Blocks**: TASK-022

## Objective

Write integration tests for the ingest method covering all four cases (A/B/C/D) and idempotency.

## Files to Create/Modify

### 1. `tests/conftest.py`

Test fixtures using testcontainers:

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg

@pytest.fixture(scope="session")
def settings(pg_container):
    from semsearch.config import Settings, EmbeddingProviderConfig
    return Settings(
        database_url=pg_container.get_connection_url(driver="psycopg"),
        embedding_provider=EmbeddingProviderConfig(
            type="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

@pytest.fixture
def service(settings):
    from semsearch.service import SemanticSearchService
    with SemanticSearchService.from_settings(settings) as svc:
        svc.init_schema(recreate=True)  # Fresh table per test
        yield svc

@pytest.fixture
def sample_txt(tmp_path):
    """Create a text file with enough content for ~3 chunks."""
    content = "This is a test paragraph. " * 100  # ~2500 chars
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file

@pytest.fixture
def sample_pdf(tmp_path):
    """Create or provide a small PDF for testing."""
    # Use a known small PDF file from fixtures
    # Or generate one with reportlab/fpdf
    pass

@pytest.fixture
def sample_csv(tmp_path):
    """Create a CSV with 10 rows."""
    import csv
    file = tmp_path / "sample.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        for i in range(10):
            writer.writerow([f"item_{i}", f"value_{i}"])
    return file
```

### 2. `tests/test_service_ingest.py`

```python
import pytest
from pathlib import Path
from unittest.mock import patch

class TestIngestText:
    def test_ingest_text_file(self, service, sample_txt):  # I-1
        """Ingesting a text file produces chunks with correct counts"""
        result = service.ingest(sample_txt)
        assert result.chunks_added >= 1
        assert result.chunks_reused == 0
        assert result.source == str(sample_txt)

class TestIngestPDF:
    def test_ingest_pdf(self, service, sample_pdf):  # I-2
        """Ingesting a PDF produces chunks with page metadata"""
        result = service.ingest(sample_pdf)
        assert result.chunks_added >= 5
        # Verify page metadata in DB
        # ...

class TestIngestCSV:
    def test_ingest_csv(self, service, sample_csv):  # I-3
        """Ingesting a CSV produces one chunk per row"""
        result = service.ingest(sample_csv)
        assert result.chunks_added == 10

class TestIdempotency:
    def test_reingest_unchanged(self, service, sample_txt):  # I-9
        """Re-ingesting unchanged file reuses all chunks"""
        service.ingest(sample_txt)
        result = service.ingest(sample_txt)
        assert result.chunks_added == 0
        assert result.chunks_reused > 0
        assert result.chunks_updated == 0

    def test_reingest_zero_embed_calls(self, service, sample_txt):  # I-9b
        """Re-ingesting unchanged file makes zero embedding API calls"""
        service.ingest(sample_txt)
        with patch.object(service.embedder, "embed_documents") as mock_embed:
            service.ingest(sample_txt)
            mock_embed.assert_not_called()

    def test_reingest_partial_change(self, service, tmp_path):  # I-10
        """Re-ingesting with 1 changed chunk re-embeds only that chunk"""
        # Create file with enough content for 5 chunks
        file = tmp_path / "test.txt"
        file.write_text(("chunk content " * 100 + "\n") * 5)
        service.ingest(file)

        # Modify one chunk
        content = file.read_text()
        lines = content.split("\n")
        lines[2] = "MODIFIED CHUNK " * 100
        file.write_text("\n".join(lines))

        result = service.ingest(file)
        assert result.chunks_reused == 4
        assert result.chunks_updated == 1
        assert result.chunks_added == 0

    def test_embed_call_count_on_partial_change(self, service, tmp_path):  # I-10b
        """Partial change calls embed_documents once with list of length 1"""
        file = tmp_path / "test.txt"
        file.write_text(("chunk " * 100 + "\n") * 5)
        service.ingest(file)

        content = file.read_text()
        lines = content.split("\n")
        lines[2] = "CHANGED " * 100
        file.write_text("\n".join(lines))

        with patch.object(service.embedder, "embed_documents", wraps=service.embedder.embed_documents) as mock:
            service.ingest(file)
            assert mock.call_count == 1
            assert len(mock.call_args[0][0]) == 1  # Only 1 chunk re-embedded

    def test_case_d_stale_tail_pruned(self, service, tmp_path):  # I-10d
        """File shortening deletes stale tail chunks"""
        file = tmp_path / "test.txt"
        file.write_text(("chunk " * 100 + "\n") * 5)
        service.ingest(file)

        # Shorten file to 3 chunks
        content = file.read_text()
        lines = content.split("\n")
        file.write_text("\n".join(lines[:3]))

        result = service.ingest(file)
        assert result.chunks_pruned == 2
        # Total rows should be 3
        stats = service.stats()
        assert stats["chunk_count"] == 3

    def test_case_d_no_prune_when_growing(self, service, tmp_path):  # I-10e
        """File growing does not prune anything"""
        file = tmp_path / "test.txt"
        file.write_text(("chunk " * 100 + "\n") * 3)
        service.ingest(file)

        # Grow file to 5 chunks
        file.write_text(("chunk " * 100 + "\n") * 5)
        result = service.ingest(file)
        assert result.chunks_pruned == 0

class TestForceReembed:
    def test_force_reembeds_everything(self, service, sample_txt):  # I-11
        """--force re-embeds all chunks even if unchanged"""
        service.ingest(sample_txt)
        with patch.object(service.embedder, "embed_documents", wraps=service.embedder.embed_documents) as mock:
            result = service.ingest(sample_txt, reembed_unchanged=True)
            assert result.chunks_reused == 0
            assert result.chunks_updated > 0
            assert mock.call_count == 1
```

## Test Data Requirements

Create `tests/fixtures/` with:
- `sample.txt` — Multi-paragraph text (~3000 chars, ~3 chunks)
- `sample.pdf` — 5+ page PDF
- `sample.csv` — 10 rows with headers
- `sample.json` — JSON array with 5+ elements

## Critical Notes

1. **Each test gets fresh table** — `svc.init_schema(recreate=True)` in fixture
2. **CASE D test requires file modification** — Use `tmp_path` fixture
3. **Mock `embed_documents`** — To verify call count for idempotency
4. **HuggingFace model loads once** — Session-scoped container, model cached

## Verification

- [ ] All ingest tests pass
- [ ] CASE A (reuse) makes zero embed calls
- [ ] CASE B (update) makes exactly 1 embed call with changed chunks only
- [ ] CASE D (prune) deletes stale tail rows
- [ ] `--force` bypasses content-hash cache
