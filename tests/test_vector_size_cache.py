"""Tests for vector size caching and DB read (TASK-025)."""

import pytest
from pydantic import SecretStr
from unittest.mock import MagicMock, patch

from semsearch.config import Settings, EmbeddingProviderConfig
from semsearch.service import SemanticSearchService
from tests.conftest import MockEmbeddings


# ------------------------------------------------------------------
# Unit tests (no DB required)
# ------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    """Build a minimal Settings object with a fake DB URL."""
    defaults = dict(
        database_url="postgresql+psycopg://fake:fake@localhost/fake",
        collection_name="semsearch_chunks",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestVectorSizeCaching:
    """Test _get_vector_size() caching behaviour."""

    def _make_service(self, dim: int = 128) -> tuple[SemanticSearchService, MockEmbeddings]:
        """Build a service with a spy-able embedder."""
        embedder = MockEmbeddings(dim=dim)
        # Wrap embed_query to count calls
        original = embedder.embed_query
        embedder._query_calls = 0

        def counting_query(text: str):
            embedder._query_calls += 1
            return original(text)

        embedder.embed_query = counting_query  # type: ignore[method-assign]

        settings = _make_settings()
        engine = MagicMock()
        svc = SemanticSearchService(settings, engine, embedder)
        return svc, embedder

    def test_first_call_probes_embedder(self):
        svc, embedder = self._make_service(dim=64)
        size = svc._get_vector_size()
        assert size == 64
        assert embedder._query_calls == 1

    def test_second_call_uses_cache(self):
        svc, embedder = self._make_service(dim=64)
        svc._get_vector_size()
        svc._get_vector_size()
        assert embedder._query_calls == 1  # only the first call

    def test_force_probe_bypasses_cache(self):
        svc, embedder = self._make_service(dim=64)
        svc._get_vector_size()
        svc._get_vector_size(force_probe=True)
        assert embedder._query_calls == 2

    def test_cache_stored_on_instance(self):
        svc, _ = self._make_service(dim=256)
        assert svc._cached_vector_size is None
        svc._get_vector_size()
        assert svc._cached_vector_size == 256


# ------------------------------------------------------------------
# Integration tests (require DB)
# ------------------------------------------------------------------


@pytest.fixture
def svc_with_table(pg_url):
    """Service with a real table created (for DB-read tests)."""
    embedder = MockEmbeddings(dim=128)
    settings = Settings(
        database_url=pg_url,
        collection_name="semsearch_chunks_cache_test",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
    )
    from semsearch.store import build_engine, init_schema

    engine = build_engine(settings)
    init_schema(settings, engine, embedder.dim, recreate=True)
    svc = SemanticSearchService(settings, engine, embedder)
    yield svc
    svc.close()


class TestVectorSizeFromDB:
    """Test _get_vector_size_from_db() with a real database."""

    def test_reads_dim_from_existing_table(self, svc_with_table):
        dim = svc_with_table._get_vector_size_from_db()
        assert dim == 128

    def test_reads_dim_with_reused_conn(self, svc_with_table):
        """_get_vector_size_from_db reuses an existing connection."""
        conn = svc_with_table._get_conn()
        try:
            dim = svc_with_table._get_vector_size_from_db(conn)
            assert dim == 128
        finally:
            conn.close()

    def test_returns_none_when_table_missing(self, pg_url):
        settings = Settings(
            database_url=pg_url,
            collection_name="nonexistent_table_xyz",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
        )
        embedder = MockEmbeddings(dim=64)
        from semsearch.store import build_engine

        engine = build_engine(settings)
        svc = SemanticSearchService(settings, engine, embedder)
        try:
            dim = svc._get_vector_size_from_db()
            assert dim is None
        finally:
            svc.close()

    def test_stats_uses_db_read_no_api_call(self, svc_with_table):
        """stats() should get dim from DB, not by embedding a probe."""
        # Patch embed_query to track if it's called
        with patch.object(svc_with_table.embedder, 'embed_query', wraps=svc_with_table.embedder.embed_query) as mock_embed:
            result = svc_with_table.stats()

            assert result["embedding_dim"] == 128
            # DB path used → no embed_query call needed
            mock_embed.assert_not_called()
