"""Tests for the FastAPI HTTP server (TASK-028).

The server is read-only (search + stats). Data modification uses CLI.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from langchain_core.documents import Document

# Support both httpx and httpx2 (depending on OpenAI SDK version)
try:
    from httpx2 import ASGITransport, AsyncClient
except ImportError:
    from httpx import ASGITransport, AsyncClient

from semsearch.config import Settings, EmbeddingProviderConfig, RerankerProviderConfig
from semsearch.server import create_app
from semsearch.service import SemanticSearchService
from semsearch.store import build_engine, init_schema
from tests.conftest import MockEmbeddings


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def svc(pg_url):
    """SemanticSearchService with mock embeddings and a fresh table."""
    embedder = MockEmbeddings(dim=128)
    settings = Settings(
        database_url=pg_url,
        collection_name="semsearch_chunks_test",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
        reranker=None,  # Explicitly disable reranker for most tests
    )
    engine = build_engine(settings)
    init_schema(settings, engine, embedder.dim, recreate=True)
    svc = SemanticSearchService(settings, engine, embedder)
    yield svc
    svc.close()


@pytest.fixture
def app(svc, pg_url):
    """FastAPI app wired to the mock service."""
    settings = Settings(
        database_url=pg_url,
        collection_name="semsearch_chunks_test",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
        reranker=None,
    )
    # Create app with the service
    application = create_app(settings, service=svc)
    # Manually set the service on app.state since ASGITransport doesn't run lifespan
    application.state.service = svc
    return application


@pytest.fixture
async def client(app):
    """Async HTTP client for testing the app."""
    # pi-lens-ignore: reportArgumentType (httpx2/httpx union vs AsyncBaseTransport;
    # structurally compatible at runtime)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:  # type: ignore[arg-type]
        yield c


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


class TestHealth:
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


class TestSearch:
    async def test_search_returns_results(self, client):
        r = await client.post("/search", json={"query": "test", "k": 3})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert data["query"] == "test"
        assert data["k"] == 3

    async def test_search_with_rerank_not_configured(self, client):
        """Rerank returns 500 when reranker is not configured."""
        r = await client.post("/search", json={"query": "test", "k": 2, "rerank": True})
        assert r.status_code == 500
        assert "Reranker not configured" in r.json()["detail"]

    async def test_search_with_rerank(self, svc, pg_url):
        """Rerank works when reranker is configured."""
        # First ingest something to search over
        from pathlib import Path
        import tempfile
        tmp = tempfile.mkdtemp()
        f = Path(tmp) / "rerank_doc.txt"
        f.write_text("Hello world. " * 100)
        svc.ingest(f)

        # Mock the reranker — must preserve _search_result in metadata
        def mock_rerank(query, documents, top_n=None):
            """Simulate reranking by returning docs with rerank_score."""
            result = []
            for doc in documents[: (top_n or len(documents))]:
                doc.metadata["rerank_score"] = 0.95
                result.append(doc)
            return result

        svc._reranker = MagicMock()
        svc._reranker.rerank.side_effect = mock_rerank

        # Create app with the service that has a reranker
        settings = Settings(
            database_url=pg_url,
            collection_name="semsearch_chunks_test",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            reranker=None,
        )
        app = create_app(settings, service=svc)
        app.state.service = svc
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as client:  # type: ignore[arg-type]
            r = await client.post("/search", json={"query": "hello world", "k": 2, "rerank": True})
            assert r.status_code == 200
            data = r.json()
            assert data["reranked"] is True
            assert len(data["results"]) > 0
            # Verify rerank_score is in the results
            assert data["results"][0].get("metadata", {}).get("rerank_score") == 0.95
            # Verify the reranker was called
            svc._reranker.rerank.assert_called_once()

    async def test_search_with_filter(self, client):
        r = await client.post(
            "/search",
            json={"query": "test", "k": 5, "filter": {"doc_type": "txt"}},
        )
        assert r.status_code == 200

    async def test_search_k_zero_is_422(self, client):
        r = await client.post("/search", json={"query": "test", "k": 0})
        assert r.status_code == 422

    async def test_search_k_over_50_is_422(self, client):
        r = await client.post("/search", json={"query": "test", "k": 51})
        assert r.status_code == 422

    async def test_search_missing_query(self, client):
        r = await client.post("/search", json={"k": 5})
        assert r.status_code == 422


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------


class TestStats:
    async def test_stats(self, client):
        r = await client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "chunk_count" in data
        assert "source_count" in data
        assert "table" in data
        assert isinstance(data["chunk_count"], int)
        assert isinstance(data["source_count"], int)


# ------------------------------------------------------------------
# OpenAPI docs
# ------------------------------------------------------------------


class TestDocs:
    async def test_openapi_docs_available(self, client):
        """OpenAPI docs should be available at /docs."""
        r = await client.get("/docs")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    async def test_openapi_json_available(self, client):
        """OpenAPI JSON spec should be available at /openapi.json."""
        r = await client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert "openapi" in data
        assert data["info"]["title"] == "semsearch"


# ------------------------------------------------------------------
# Warmup (PLAN.md Phase W)
# ------------------------------------------------------------------


class TestWarmup:
    """warmup() pre-builds lazy local resources; fail-open, no paid-API calls."""

    def test_warmup_statuses_without_reranker(self, svc) -> None:
        result = svc.warmup()
        assert result == {
            "store": True,
            "db": True,
            "reranker": False,
            "pool_prefill": True,
        }

    def test_warmup_is_idempotent(self, svc) -> None:
        """Properties cache — double-warm (pre-built services) is harmless."""
        first = svc.warmup()
        second = svc.warmup()
        assert first == second == {
            "store": True,
            "db": True,
            "reranker": False,
            "pool_prefill": True,
        }

    def test_warmup_makes_no_embedding_calls(self, svc) -> None:
        """PLAN.md §W.2 guarantee: warmup never contacts paid APIs.

        Regression guard for the availability-coupling rejection: startup
        must not depend on (or spend money at) the embedding provider.
        """
        result = svc.warmup()
        assert result["store"] and result["db"]
        assert svc.embedder._call_count == 0

    def test_warmup_with_reranker_configured(self, pg_url) -> None:
        from semsearch.reranker import Reranker

        embedder = MockEmbeddings(dim=128)
        settings = Settings(
            database_url=pg_url,
            collection_name="semsearch_chunks_test",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            reranker=RerankerProviderConfig(
                base_url="https://example.com/rerank",
                model="test-rerank-model",
                api_key=SecretStr("test-key"),
            ),
        )
        engine = build_engine(settings)
        svc = SemanticSearchService(settings, engine, embedder)
        try:
            result = svc.warmup()
            assert result["reranker"] is True
            assert isinstance(svc._reranker, Reranker)
        finally:
            svc.close()

    def test_warmup_fail_open_on_missing_table(self, pg_url) -> None:
        """Bogus collection_name → store step deferred, DB step OK, no raise."""
        embedder = MockEmbeddings(dim=128)
        settings = Settings(
            database_url=pg_url,
            collection_name="no_such_table_warmup_check",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            reranker=None,
        )
        engine = build_engine(settings)
        svc = SemanticSearchService(settings, engine, embedder)
        try:
            result = svc.warmup()
            assert result["store"] is False
            assert result["db"] is True  # SELECT 1 is table-independent
        finally:
            svc.close()

    def test_lifespan_warms_up_and_health_ok(self, pg_url, caplog) -> None:
        """Startup runs warmup; /health responds even with unreachable provider.

        Negative proof (PLAN.md verification 5): the OpenAI endpoint is never
        contacted at startup — a dummy key and unreachable provider don't
        block startup or /health.
        """
        import logging
        import os

        from fastapi.testclient import TestClient

        # Point the provider at a surely-dead endpoint to prove no API call.
        os.environ["SEMSEARCH_EMBEDDING_PROVIDER__TYPE"] = "openai_compatible"
        try:
            settings = Settings(
                database_url=pg_url,
                collection_name="semsearch_chunks_test",
                embedding_provider=EmbeddingProviderConfig(
                    type="openai_compatible",
                    model="test-model",
                    api_key=SecretStr("dummy"),
                    base_url="http://127.0.0.1:1/v1",  # nothing listens here
                ),
                reranker=None,
            )
            app = create_app(settings)
            with caplog.at_level(logging.INFO, logger="semsearch.server"):
                with TestClient(app) as client:
                    assert client.get("/health").status_code == 200
                    svc = app.state.service
                    assert svc._store is not None  # warmed at startup
            warm_logs = [r for r in caplog.records if "warmup" in r.getMessage()]
            assert warm_logs, "expected a warmup log line at startup"
            # Service is closed by lifespan shutdown.
        finally:
            os.environ.pop("SEMSEARCH_EMBEDDING_PROVIDER__TYPE", None)



class TestPoolPrefill:
    """Phase F: warmup() pre-fills PGEngine's SQLAlchemy connection pool."""

    def test_pool_prefill_fills_pool(self, svc) -> None:
        """After warmup, >= 5 connections sit checked-in and ready."""
        result = svc.warmup()
        assert result["pool_prefill"] is True

        sa_pool = svc.engine._pool.sync_engine.pool
        assert sa_pool.size() == 5
        assert sa_pool.checkedin() >= 5

    def test_engine_usable_after_prefill(self, svc) -> None:
        """Pre-filled connections are loop-affine to PGEngine's loop — reuse OK."""
        svc.warmup()
        stats = svc.stats()  # exercises PGEngine checkout post-prefill
        assert "chunk_count" in stats

    def test_prefill_orthogonal_to_psycopg_pool(self, pg_url) -> None:
        """Phase C psycopg pool (enabled) and SQLAlchemy pre-fill coexist."""
        from semsearch.config import PoolConfig

        embedder = MockEmbeddings(dim=128)
        settings = Settings(
            database_url=pg_url,
            collection_name="semsearch_chunks_test",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            reranker=None,
            pool=PoolConfig(min_size=1, max_size=2),
        )
        engine = build_engine(settings)
        init_schema(settings, engine, embedder.dim, recreate=True)
        svc = SemanticSearchService(settings, engine, embedder)
        try:
            result = svc.warmup()
            assert result["store"] is True
            assert result["db"] is True
            assert result["pool_prefill"] is True
            # Both pools populated independently.
            assert svc.engine._pool.sync_engine.pool.checkedin() >= 5
            psycopg_pool = svc._pool
            assert psycopg_pool is not None  # pooling enabled via PoolConfig
            assert psycopg_pool.get_stats()["pool_available"] >= 1
        finally:
            svc.close()
