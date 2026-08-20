"""Tests for the FastAPI HTTP server (TASK-028).

The server is read-only (search + stats). Data modification uses CLI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from httpx2 import ASGITransport, AsyncClient
from langchain_core.documents import Document

from semsearch.config import Settings, EmbeddingProviderConfig
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
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
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
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
