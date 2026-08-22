"""Tests for HNSW tuning configuration (TASK-026)."""

import psycopg
import pytest
from pydantic import SecretStr

from semsearch.config import Settings, EmbeddingProviderConfig, HnswConfig
from semsearch.service import SemanticSearchService
from semsearch.store import build_engine, init_schema
from tests.conftest import MockEmbeddings


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------


class TestHnswConfig:
    """Test HnswConfig model validation."""

    def test_defaults(self):
        cfg = HnswConfig()
        assert cfg.m == 16
        assert cfg.ef_construction == 200
        assert cfg.ef_search == 80

    def test_custom_values(self):
        cfg = HnswConfig(m=32, ef_construction=300, ef_search=120)
        assert cfg.m == 32
        assert cfg.ef_construction == 300
        assert cfg.ef_search == 120

    def test_validation_lower_bound(self):
        with pytest.raises(Exception):
            HnswConfig(m=1)  # min is 2

    def test_validation_upper_bound(self):
        with pytest.raises(Exception):
            HnswConfig(ef_construction=1001)  # max is 1000


class TestSettingsHnsw:
    """Test that Settings includes HnswConfig with defaults."""

    def test_default_hnsw_config(self):
        settings = Settings(
            database_url="postgresql+psycopg://fake:fake@localhost/fake",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
        )
        assert isinstance(settings.hnsw, HnswConfig)
        assert settings.hnsw.m == 16
        assert settings.hnsw.ef_construction == 200
        assert settings.hnsw.ef_search == 80

    def test_env_var_override(self, monkeypatch):
        """HNSW config can be overridden via env vars."""
        monkeypatch.setenv("SEMSEARCH_HNSW__M", "32")
        monkeypatch.setenv("SEMSEARCH_HNSW__EF_CONSTRUCTION", "300")
        monkeypatch.setenv("SEMSEARCH_HNSW__EF_SEARCH", "120")
        settings = Settings(
            database_url="postgresql+psycopg://fake:fake@localhost/fake",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
        )
        assert settings.hnsw.m == 32
        assert settings.hnsw.ef_construction == 300
        assert settings.hnsw.ef_search == 120


# ------------------------------------------------------------------
# Integration tests (require DB)
# ------------------------------------------------------------------


@pytest.fixture
def hnsw_settings(pg_url):
    """Settings with custom HNSW parameters."""
    return Settings(
        database_url=pg_url,
        collection_name="semsearch_chunks_hnsw_test",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
        hnsw=HnswConfig(m=24, ef_construction=250, ef_search=100),
    )


@pytest.fixture
def hnsw_service(pg_url, hnsw_settings):
    """Service with HNSW-tuned schema."""
    embedder = MockEmbeddings(dim=128)
    engine = build_engine(hnsw_settings)
    init_schema(hnsw_settings, engine, embedder.dim, recreate=True)
    svc = SemanticSearchService(hnsw_settings, engine, embedder)
    yield svc
    svc.close()


class TestHnswIndexCreation:
    """Test that HNSW index uses config values."""

    def test_index_created_with_config_params(self, hnsw_service, hnsw_settings):
        """Verify the HNSW index was created with m and ef_construction from config."""
        table = hnsw_settings.collection_name
        db_url = hnsw_settings.database_url.replace("+psycopg", "")
        conn = psycopg.connect(db_url)
        try:
            with conn.cursor() as cur:
                # Check that the HNSW index exists
                cur.execute(
                    "SELECT am.amname "
                    "FROM pg_class c "
                    "JOIN pg_index i ON c.oid = i.indexrelid "
                    "JOIN pg_am am ON c.relam = am.oid "
                    "WHERE i.indrelid = %s::regclass "
                    "AND c.relname = %s",
                    (table, f"{table}_hnsw_idx"),
                )
                row = cur.fetchone()
                # Index might be DiskANN if vectorscale is installed — skip in that case
                if row is None:
                    pytest.skip("HNSW index not created (likely DiskANN or >2000 dim)")
                assert row[0] == "hnsw"

                # Verify index options via pg_get_indexdef
                cur.execute(
                    "SELECT pg_get_indexdef(c.oid) "
                    "FROM pg_class c "
                    "JOIN pg_index i ON c.oid = i.indexrelid "
                    "WHERE i.indrelid = %s::regclass "
                    "AND c.relname = %s",
                    (table, f"{table}_hnsw_idx"),
                )
                index_def = cur.fetchone()[0]
                assert "m='24'" in index_def
                assert "ef_construction='250'" in index_def
        finally:
            conn.close()

    def test_ef_search_set_on_table(self, hnsw_service, hnsw_settings):
        """Verify table-level ef_search was set."""
        db_url = hnsw_settings.database_url.replace("+psycopg", "")
        conn = psycopg.connect(db_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT reloptions FROM pg_class WHERE relname = %s",
                    (hnsw_settings.collection_name,),
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    pytest.skip("No reloptions found (table may not use HNSW)")
                reloptions = row[0]
                assert any("hnsw.ef_search=100" in opt for opt in reloptions), (
                    f"Expected hnsw.ef_search=100 in reloptions, got {reloptions}"
                )
        finally:
            conn.close()


class TestDefaultHnswConfig:
    """Test that default HNSW config is applied when no custom config is given."""

    def test_defaults_applied(self, pg_url):
        """Default config should use ef_construction=200, ef_search=80."""
        settings = Settings(
            database_url=pg_url,
            collection_name="semsearch_chunks_hnsw_defaults",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            # No hnsw= override — uses defaults
        )
        embedder = MockEmbeddings(dim=128)
        engine = build_engine(settings)
        init_schema(settings, engine, embedder.dim, recreate=True)
        svc = SemanticSearchService(settings, engine, embedder)
        try:
            table = settings.collection_name
            db_url = settings.database_url.replace("+psycopg", "")
            conn = psycopg.connect(db_url)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_get_indexdef(c.oid) "
                        "FROM pg_class c "
                        "JOIN pg_index i ON c.oid = i.indexrelid "
                        "WHERE i.indrelid = %s::regclass "
                        "AND c.relname = %s",
                        (table, f"{table}_hnsw_idx"),
                    )
                    row = cur.fetchone()
                    if row is None:
                        pytest.skip("HNSW index not created (likely DiskANN)")
                    index_def = row[0]
                    assert "ef_construction='200'" in index_def
            finally:
                conn.close()
        finally:
            svc.close()
