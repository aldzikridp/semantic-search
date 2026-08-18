"""Generic reranker for any OpenAI-compatible reranker endpoint (spec §7.x).

Works with OpenRouter, Jina, Cohere, and any other endpoint that uses:
    POST {base_url}
    {"model": "...", "query": "...", "documents": [...], "top_n": N}

Response:
    {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.documents import Document
from pydantic import SecretStr

from semsearch.config import RerankerProviderConfig, Settings
from semsearch.errors import SearchError


class Reranker:
    """Generic reranker that works with any compatible endpoint.

    Usage::

        reranker = Reranker(config, api_key)
        reranked_docs = reranker.rerank("query", documents, top_n=5)
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
            SearchError: If the rerank API call fails.
        """
        if not documents:
            return []

        n = top_n or self.default_top_n
        texts = [doc.page_content for doc in documents]

        try:
            response = httpx.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": n,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
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
