"""Shared fixtures for semsearch tests."""

import os
import subprocess
import pytest
from pydantic import SecretStr

from semsearch.config import Settings, EmbeddingProviderConfig


class MockEmbeddings:
    """Deterministic mock embeddings for testing."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._call_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: str) -> list[list[float]]:
        return self.embed_documents(texts)

    def _embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        raw = (h * ((self.dim // 32) + 1))[:self.dim]
        return [b / 255.0 for b in raw]


def _get_database_url() -> str | None:
    """Get database URL from environment or local PostgreSQL."""
    # Check environment first
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url

    # Try local PostgreSQL socket
    pgsocket = os.path.expanduser("~/Project/semantic-search/.pgsocket")
    if os.path.exists(pgsocket):
        return f"postgresql+psycopg://semsearch:test@/{pgsocket}/.s.PGSQL.5432?host={pgsocket}&dbname=semsearch"

    # Try testcontainers as fallback
    try:
        import subprocess
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if result.returncode == 0:
            from testcontainers.community.postgres import PostgresContainer
            # Can't yield here, need to handle differently
            pass
    except Exception:
        pass

    return None


@pytest.fixture(scope="session")
def pg_url():
    """Database URL for testing."""
    url = _get_database_url()
    if url is None:
        pytest.skip("No PostgreSQL available for testing")
    return url


@pytest.fixture(scope="session")
def settings(pg_url):
    """Settings using test database."""
    return Settings(
        database_url=pg_url,
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key-for-mocking"),
        ),
    )


@pytest.fixture
def mock_embeddings():
    """Deterministic mock embeddings."""
    return MockEmbeddings(dim=128)


@pytest.fixture
def service(pg_url, mock_embeddings):
    """SemanticSearchService with fresh table and mock embeddings."""
    from semsearch.service import SemanticSearchService
    from semsearch.store import build_engine, init_schema

    settings = Settings(
        database_url=pg_url,
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
    )

    engine = build_engine(settings)

    # Initialize schema first (creates table)
    init_schema(settings, engine, mock_embeddings.dim, recreate=True)

    # Create service (store will be lazy-initialized)
    svc = SemanticSearchService(settings, engine, mock_embeddings)
    yield svc
    svc.close()


@pytest.fixture
def sample_txt(tmp_path):
    """Text file with ~3 chunks worth of content."""
    content = ("This is a test paragraph with enough content. " * 50 + "\n") * 3
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file


@pytest.fixture
def sample_csv(tmp_path):
    """CSV file with 10 rows."""
    import csv

    file = tmp_path / "sample.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value", "description"])
        for i in range(10):
            writer.writerow([f"item_{i}", str(i * 100), f"Description for item {i}" * 5])
    return file
