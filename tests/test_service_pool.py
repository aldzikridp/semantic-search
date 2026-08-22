"""Tests for the opt-in psycopg connection pool (Phase C, PLAN.md).

Pooling is OFF by default (min_size=0); these tests verify both modes and
the AGENTS.md #15 ownership rule (caller-provided conns never touch the pool).
"""

from pathlib import Path
from typing import Any

# pi-lens-ignore: import-not-found
import pytest
# pi-lens-ignore: import-not-found
from pydantic import SecretStr

from semsearch.config import EmbeddingProviderConfig, PoolConfig, Settings


def _make_service(pg_url: str, mock_embeddings: Any, pool: PoolConfig | None = None) -> Any:
    """Build a SemanticSearchService with fresh table, optional pool config."""
    from semsearch.service import SemanticSearchService
    from semsearch.store import build_engine, init_schema

    settings = Settings(
        database_url=pg_url,
        collection_name="semsearch_chunks_test",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
        pool=pool or PoolConfig(),
    )
    engine = build_engine(settings)
    init_schema(settings, engine, mock_embeddings.dim, recreate=True)
    return SemanticSearchService(settings, engine, mock_embeddings)


def _pool(svc: Any) -> Any:
    """Return the service's built pool, asserting it exists."""
    pool = svc._pool
    assert pool is not None, "expected pooling to be enabled and built"
    return pool


class TestPoolDisabled:
    def test_pool_is_none_by_default(self, pg_url: str, mock_embeddings: Any) -> None:
        """Default config (min_size=0) never builds a pool."""
        svc = _make_service(pg_url, mock_embeddings)
        try:
            assert svc.settings.pool.min_size == 0
            assert svc._pool is None
            # Operations still work with fresh connections.
            assert svc.stats()["chunk_count"] == 0
        finally:
            svc.close()

    def test_release_conn_closes_unleased(self, pg_url: str, mock_embeddings: Any) -> None:
        """_release_conn closes connections that did not come from the pool."""
        import psycopg

        svc = _make_service(pg_url, mock_embeddings)
        try:
            conn = psycopg.connect(svc._db_url)
            assert not conn.closed
            svc._release_conn(conn)
            assert conn.closed
        finally:
            svc.close()


class TestPoolEnabled:
    def test_pool_lazily_built(self, pg_url: str, mock_embeddings: Any) -> None:
        """With min_size>0, _pool builds lazily on first access."""
        svc = _make_service(pg_url, mock_embeddings, PoolConfig(min_size=1, max_size=2))
        try:
            assert svc._pool_obj is None  # not built until first use
            assert svc._pool is not None  # property builds it
            assert svc._pool is svc._pool_obj
        finally:
            svc.close()

    def test_connections_reused_across_operations(self, pg_url: str, mock_embeddings: Any, sample_txt: Path) -> None:
        """Sequential operations lease/return through one small pool."""
        svc = _make_service(pg_url, mock_embeddings, PoolConfig(min_size=1, max_size=2))
        try:
            svc.ingest(sample_txt)
            svc.stats()
            svc.stats()
            stats = _pool(svc).get_stats()
            assert stats["requests_num"] >= 3  # ingest + 2x stats
            assert stats["pool_available"] >= 1  # all released, none leaked
            assert stats.get("returns_bad", 0) == 0
        finally:
            svc.close()

    def test_close_shuts_down_pool(self, pg_url: str, mock_embeddings: Any) -> None:
        """close() terminates the pool so nothing outlives the service."""
        svc = _make_service(pg_url, mock_embeddings, PoolConfig(min_size=1, max_size=2))
        svc.stats()  # force pool creation
        assert svc._pool_obj is not None
        svc.close()
        assert svc._pool_obj is None
        assert not svc._leased

    def test_caller_conn_never_touches_pool(self, pg_url: str, mock_embeddings: Any, sample_txt: Path) -> None:
        """Ownership rule: caller-provided conn bypasses the pool entirely."""
        import psycopg

        svc = _make_service(pg_url, mock_embeddings, PoolConfig(min_size=1, max_size=2))
        try:
            svc.ingest(sample_txt)  # warm the pool
            requests_before = _pool(svc).get_stats()["requests_num"]

            caller_conn = psycopg.connect(svc._db_url)
            try:
                result = svc.stats(conn=caller_conn)
                assert result["chunk_count"] >= 1
                # Caller still owns it — service must not have closed it.
                assert not caller_conn.closed
            finally:
                caller_conn.close()

            # No pool traffic from the caller-owned connection.
            assert _pool(svc).get_stats()["requests_num"] == requests_before
        finally:
            svc.close()

    def test_pool_survives_operation_errors(self, pg_url: str, mock_embeddings: Any, tmp_path: Path) -> None:
        """Failed ingests roll back and still return their connection."""
        svc = _make_service(pg_url, mock_embeddings, PoolConfig(min_size=1, max_size=2))
        try:
            bad = tmp_path / "bad.pdf"
            bad.write_bytes(b"not a pdf")
            with pytest.raises(Exception):  # FileIngestError
                svc.ingest(bad)
            stats = _pool(svc).get_stats()
            assert stats["pool_available"] >= 1  # connection returned, not leaked
            assert stats.get("returns_bad", 0) == 0  # returned in clean state
        finally:
            svc.close()


class TestPoolConfigValidation:
    def test_min_size_zero_disables(self) -> None:
        assert PoolConfig().min_size == 0

    def test_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            PoolConfig(min_size=-1)
        with pytest.raises(ValueError):
            PoolConfig(max_size=33)
        with pytest.raises(ValueError):
            PoolConfig(timeout=0)

    def test_env_var_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEMSEARCH_POOL__MIN_SIZE", "2")
        monkeypatch.setenv("SEMSEARCH_POOL__MAX_SIZE", "8")
        s = Settings(database_url="postgresql://x")
        assert (s.pool.min_size, s.pool.max_size) == (2, 8)
