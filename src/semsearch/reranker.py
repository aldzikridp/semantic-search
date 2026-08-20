"""Generic reranker for any OpenAI-compatible reranker endpoint (spec §7.x).

Works with OpenRouter, Jina, Cohere, and any other endpoint that uses:
    POST {base_url}
    {"model": "...", "query": "...", "documents": [...], "top_n": N}

Response:
    {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document
from pydantic import SecretStr

from semsearch.config import RerankerProviderConfig, Settings
from semsearch.errors import SearchError

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 0.5  # seconds


def _detect_httpx():
    """Detect which httpx version is available.

    OpenAI SDK v3 uses httpx2, v2 uses httpx.
    Returns (httpx_module, Timeout, Limits, HTTPStatusError, HTTPError).
    """
    try:
        import openai
        major_version = int(openai.__version__.split(".")[0])
        if major_version >= 3:
            import httpx2
            return httpx2, httpx2.Timeout, httpx2.Limits, httpx2.HTTPStatusError, httpx2.HTTPError
    except (ImportError, AttributeError, ValueError):
        pass

    import httpx
    return httpx, httpx.Timeout, httpx.Limits, httpx.HTTPStatusError, httpx.HTTPError


# Detect at module load time
_httpx_module, _Timeout, _Limits, _HTTPStatusError, _HTTPError = _detect_httpx()


class Reranker:
    """Generic reranker that works with any compatible endpoint.

    Uses a persistent HTTP client for connection pooling and
    retries 429 rate-limit responses with exponential backoff.

    Usage::

        reranker = Reranker(config, api_key)
        reranked_docs = reranker.rerank("query", documents, top_n=5)
        reranker.close()
    """

    def __init__(
        self,
        config: RerankerProviderConfig,
        api_key: str,
    ) -> None:
        self.base_url = config.base_url
        self.model = config.model
        self.api_key = api_key
        self.default_top_n = config.top_n

        # Persistent client — connection pool survives across rerank() calls
        self._client = _httpx_module.Client(
            timeout=_Timeout(connect=5.0, read=10.0, write=5.0, pool=2.0),
            limits=_Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        self._client.close()

    def __enter__(self) -> Reranker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query.
            documents: Documents from vector search.
            top_n: Number of results to return. Defaults to config.top_n.

        Returns:
            Reranked documents with rerank_score in metadata.

        Raises:
            SearchError: If the rerank API call fails after retries.
        """
        if not documents:
            return []

        n = top_n or self.default_top_n
        texts = [doc.page_content for doc in documents]
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": n,
        }

        # Retry with exponential backoff for 429s
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except _HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < _MAX_RETRIES - 1:
                    wait = _INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Reranker rate-limited (429), retrying in %.1fs (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise SearchError(f"Rerank failed: {e}") from e
            except _HTTPError as e:
                raise SearchError(f"Rerank failed: {e}") from e

        results = data.get("results", [])

        # Map indices back to original documents (preserving metadata)
        reranked: list[Document] = []
        for r in results:
            idx = r["index"]
            score = r["relevance_score"]
            doc = documents[idx]
            doc.metadata["rerank_score"] = score
            reranked.append(doc)

        return reranked


def build_reranker(settings: Settings) -> Reranker | None:
    """Build a Reranker from settings, or None if not configured.

    Falls back to embedding provider's API key if reranker.api_key is not set.

    Args:
        settings: Application settings.

    Returns:
        Reranker instance, or None if reranker is not configured.
    """
    if settings.reranker is None:
        return None

    # Get API key: reranker key first, then embedding provider key
    api_key = None
    if settings.reranker.api_key:
        api_key = settings.reranker.api_key.get_secret_value()
    elif settings.embedding_provider.api_key:
        api_key = settings.embedding_provider.api_key.get_secret_value()

    if not api_key:
        raise SearchError(
            "Reranker requires an API key. Set SEMSEARCH_RERANKER__API_KEY "
            "or SEMSEARCH_EMBEDDING_PROVIDER__API_KEY"
        )

    return Reranker(settings.reranker, api_key)
