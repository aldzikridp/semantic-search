"""Tests for persistent httpx.Client and retry logic in Reranker (TASK-027)."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from semsearch.config import RerankerProviderConfig
from semsearch.errors import SearchError
from semsearch.reranker import (
    Reranker,
    _INITIAL_BACKOFF,
    _MAX_RETRIES,
    _httpx_module,
    _Timeout,
    _Limits,
    _HTTPStatusError,
    _HTTPError,
    build_reranker,
)
from semsearch.config import Settings, EmbeddingProviderConfig


def _make_config(**overrides: Any) -> RerankerProviderConfig:
    defaults: dict[str, Any] = dict(
        base_url="https://example.com/rerank",
        model="test-rerank-model",
        api_key=SecretStr("test-key"),
        top_n=3,
    )
    defaults.update(overrides)
    return RerankerProviderConfig(**defaults)


def _make_docs(n: int = 5) -> list[Document]:
    return [
        Document(page_content=f"Document {i}", metadata={"source": f"doc_{i}.txt"})
        for i in range(n)
    ]


def _mock_success_response(results: list[dict] | None = None) -> Any:
    """Build a mock 200 response with reranker results."""
    if results is None:
        results = [
            {"index": 0, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.80},
            {"index": 1, "relevance_score": 0.60},
        ]
    resp = MagicMock(spec=_httpx_module.Response)
    resp.status_code = 200
    resp.json.return_value = {"results": results}
    resp.raise_for_status.return_value = None
    return resp


def _mock_429_response() -> Any:
    """Build a mock 429 rate-limit response."""
    resp = MagicMock(spec=_httpx_module.Response)
    resp.status_code = 429
    resp.text = "Too Many Requests"
    resp.raise_for_status.side_effect = _HTTPStatusError(
        "429", request=MagicMock(), response=resp
    )
    return resp


# ------------------------------------------------------------------
# Unit tests — Client lifecycle
# ------------------------------------------------------------------


class TestRerankerClient:
    """Test persistent _httpx_module.Client usage."""

    def test_client_created_on_init(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        assert isinstance(reranker._client, _httpx_module.Client)
        reranker.close()

    def test_close_releases_client(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        reranker.close()
        # After close, the client should be closed (no error on double-close)
        reranker.close()

    def test_context_manager(self):
        config = _make_config()
        with Reranker(config, "test-key") as reranker:
            assert isinstance(reranker._client, _httpx_module.Client)
        # Client is closed after exiting context

    def test_client_headers_contain_auth(self):
        config = _make_config()
        reranker = Reranker(config, "my-secret-key")
        assert reranker._client.headers["Authorization"] == "Bearer my-secret-key"
        reranker.close()


# ------------------------------------------------------------------
# Unit tests — Successful rerank
# ------------------------------------------------------------------


class TestRerankSuccess:
    """Test successful rerank flow."""

    def test_rerank_returns_reordered_docs(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(5)

        with patch.object(reranker._client, "post", return_value=_mock_success_response()):
            result = reranker.rerank("test query", docs, top_n=3)

        assert len(result) == 3
        assert result[0].page_content == "Document 0"
        assert result[0].metadata["rerank_score"] == 0.95
        assert result[1].page_content == "Document 2"
        reranker.close()

    def test_rerank_empty_docs(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        result = reranker.rerank("query", [])
        assert result == []
        reranker.close()

    def test_rerank_uses_persistent_client(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        mock_post = MagicMock(return_value=_mock_success_response())
        with patch.object(reranker._client, "post", mock_post):
            reranker.rerank("query", docs)
            reranker.rerank("query", docs)

        assert mock_post.call_count == 2
        # Verify the same client was used (post called on the same object)
        reranker.close()


# ------------------------------------------------------------------
# Unit tests — Retry logic
# ------------------------------------------------------------------


class TestRerankRetry:
    """Test retry with exponential backoff on 429s."""

    def test_retries_on_429_then_succeeds(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        resp_429 = _mock_429_response()
        resp_ok = _mock_success_response()

        mock_post = MagicMock(side_effect=[resp_429, resp_ok])
        with patch.object(reranker._client, "post", mock_post):
            with patch("semsearch.reranker.time.sleep") as mock_sleep:
                result = reranker.rerank("query", docs)

        assert len(result) == 3
        assert mock_post.call_count == 2
        # First retry should sleep _INITIAL_BACKOFF (0.5s)
        mock_sleep.assert_called_once_with(_INITIAL_BACKOFF)
        reranker.close()

    def test_retries_twice_on_consecutive_429s(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        resp_429 = _mock_429_response()
        resp_ok = _mock_success_response()

        mock_post = MagicMock(side_effect=[resp_429, resp_429, resp_ok])
        with patch.object(reranker._client, "post", mock_post):
            with patch("semsearch.reranker.time.sleep") as mock_sleep:
                result = reranker.rerank("query", docs)

        assert len(result) == 3
        assert mock_post.call_count == 3
        # Backoff: 0.5s, 1.0s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(_INITIAL_BACKOFF)           # 0.5
        mock_sleep.assert_any_call(_INITIAL_BACKOFF * 2)       # 1.0
        reranker.close()

    def test_raises_after_max_retries(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        resp_429 = _mock_429_response()
        mock_post = MagicMock(return_value=resp_429)

        with patch.object(reranker._client, "post", mock_post):
            with patch("semsearch.reranker.time.sleep"):
                with pytest.raises(SearchError, match="Rerank failed"):
                    reranker.rerank("query", docs)

        assert mock_post.call_count == _MAX_RETRIES
        reranker.close()

    def test_non_429_raises_immediately(self):
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        resp_500 = MagicMock(spec=_httpx_module.Response)
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"
        resp_500.raise_for_status.side_effect = _HTTPStatusError(
            "500", request=MagicMock(), response=resp_500
        )

        mock_post = MagicMock(return_value=resp_500)
        with patch.object(reranker._client, "post", mock_post):
            with patch("semsearch.reranker.time.sleep") as mock_sleep:
                with pytest.raises(SearchError, match="Rerank failed"):
                    reranker.rerank("query", docs)

        # No retry for non-429
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()
        reranker.close()

    def test_keepalive_expiry_is_300s(self):
        """Phase E: pooled connections survive agent-scale idle gaps."""
        config = _make_config()
        reranker = Reranker(config, "test-key")
        assert reranker._limits.keepalive_expiry == 300.0
        reranker.close()

    def test_stale_connection_retried_then_succeeds(self, caplog):
        """Phase E: a half-open pooled connection costs one transparent retry.

        First post() raises ConnectError (stale keep-alive hit), second
        succeeds → results returned, exactly one retry logged, no exception.
        """
        import logging

        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        stale = _httpx_module.ConnectError("stale keep-alive connection")
        mock_post = MagicMock(side_effect=[stale, _mock_success_response()])
        with patch.object(reranker._client, "post", mock_post):
            with caplog.at_level(logging.WARNING, logger="semsearch.reranker"):
                result = reranker.rerank("query", docs)

        assert len(result) == 3
        assert mock_post.call_count == 2  # exactly one retry
        retry_warnings = [
            r for r in caplog.records if "connection error" in r.getMessage()
        ]
        assert len(retry_warnings) == 1
        reranker.close()

    def test_transport_error_exhausts_retries_raises_searcherror(self):
        """Phase E: persistent transport failure surfaces as SearchError
        (not a raw httpx error) after _MAX_RETRIES attempts."""
        config = _make_config()
        reranker = Reranker(config, "test-key")
        docs = _make_docs(3)

        mock_post = MagicMock(
            side_effect=_httpx_module.ConnectError("Connection refused")
        )
        with patch.object(reranker._client, "post", mock_post):
            with pytest.raises(SearchError, match="Rerank failed after retries"):
                reranker.rerank("query", docs)

        assert mock_post.call_count == _MAX_RETRIES
        reranker.close()


# ------------------------------------------------------------------
# Unit tests — build_reranker factory
# ------------------------------------------------------------------


class TestBuildReranker:
    """Test build_reranker() factory function."""

    def test_returns_none_when_not_configured(self):
        settings = Settings(
            database_url="postgresql+psycopg://fake:fake@localhost/fake",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            reranker=None,
        )
        assert build_reranker(settings) is None

    def test_returns_reranker_when_configured(self):
        settings = Settings(
            database_url="postgresql+psycopg://fake:fake@localhost/fake",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("test-key"),
            ),
            reranker=RerankerProviderConfig(
                base_url="https://example.com/rerank",
                model="test-model",
            ),
        )
        reranker = build_reranker(settings)
        assert isinstance(reranker, Reranker)
        assert reranker.model == "test-model"
        reranker.close()

    def test_uses_embedding_key_as_fallback(self):
        settings = Settings(
            database_url="postgresql+psycopg://fake:fake@localhost/fake",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                api_key=SecretStr("embedding-key"),
            ),
            reranker=RerankerProviderConfig(
                base_url="https://example.com/rerank",
                model="test-model",
                # No api_key — should fall back to embedding provider key
            ),
        )
        reranker = build_reranker(settings)
        assert isinstance(reranker, Reranker)
        assert reranker.api_key == "embedding-key"
        reranker.close()

    def test_raises_when_no_key_available(self):
        settings = Settings(
            database_url="postgresql+psycopg://fake:fake@localhost/fake",
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
                # No api_key
            ),
            reranker=RerankerProviderConfig(
                base_url="https://example.com/rerank",
                model="test-model",
                # No api_key
            ),
        )
        with pytest.raises(SearchError, match="Reranker requires an API key"):
            build_reranker(settings)
