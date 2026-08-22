"""Search mixin: sync/async similarity search and rerank orchestration.

Part of ``semsearch.services`` (PLAN.md Phase D) — implementation detail of
``semsearch.service.SemanticSearchService``. ``PGVectorStore`` stays
read-only per AGENTS.md decision #2.
"""

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document

from semsearch.errors import SearchError
from semsearch.models import SearchResult
from semsearch.reranker import Reranker, build_reranker

from .base import BaseService

logger = logging.getLogger(__name__)


class SearchMixin(BaseService):
    """Cosine similarity search (sync + async) with optional reranking."""

    @property
    def reranker(self) -> Reranker | None:
        """Lazy-init reranker (only created if configured)."""
        if self._reranker is None and self.settings.reranker is not None:
            self._reranker = build_reranker(self.settings)
        return self._reranker

    def search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Cosine similarity search over the chunks table.

        Args:
            query: Free-text query.
            k: Top-k results. Defaults to settings.default_k. Must be 1-50.
            filter: Optional PGVectorStore filter dict.
            rerank: If True, rerank results using configured reranker.

        Returns:
            List of SearchResult sorted by score DESC (or rerank_score DESC).

        Raises:
            ValueError: k out of range.
            SearchError: Search or rerank failed.
        """
        import time as _time
        k, fetch_k = self._resolve_k(k, rerank)

        try:
            t0 = _time.monotonic()
            raw = self.store.similarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter,
            )
            logger.debug("similarity_search_with_score took %.1fms", (_time.monotonic() - t0) * 1000)
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

        results = self._to_search_results(raw)
        if rerank:
            results = self._apply_rerank(query, results, k)
        return results[:k]

    async def asearch(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Async cosine similarity search over the chunks table.

        Uses async versions of embedding and DB calls to avoid blocking.
        """
        k, fetch_k = self._resolve_k(k, rerank)

        try:
            raw = await self.store.asimilarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter,
            )
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

        results = self._to_search_results(raw)
        if rerank:
            # Reranker is sync — run the whole rerank flow in a thread.
            results = await asyncio.to_thread(self._apply_rerank, query, results, k)
        return results[:k]

    def _resolve_k(self, k: int | None, rerank: bool) -> tuple[int, int]:
        """Validate *k* and return ``(k, fetch_k)``. Raises ValueError."""
        if k is None:
            k = self.settings.default_k
        if not (1 <= k <= 50):
            raise ValueError(f"k must be between 1 and 50, got {k}")
        # When reranking, fetch more candidates for better reranking quality
        fetch_k = k * 4 if rerank else k
        return k, fetch_k

    @staticmethod
    def _to_search_results(
        results_with_scores: list[tuple[Document, float]],
    ) -> list[SearchResult]:
        """Convert (Document, cosine_distance) pairs to SearchResults.

        Score conversion per spec: score = 1.0 - cosine_distance.
        """
        results = []
        for doc, distance in results_with_scores:
            score = 1.0 - distance
            results.append(
                SearchResult(
                    id=doc.metadata.get("langchain_id", ""),
                    content=doc.page_content,
                    score=score,
                    source=doc.metadata.get("source"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    page=doc.metadata.get("page"),
                    row=doc.metadata.get("row"),
                    doc_type=doc.metadata.get("doc_type"),
                    metadata=doc.metadata,
                )
            )
        return results

    @staticmethod
    def _rerank_docs(results: list[SearchResult]) -> list[Document]:
        """Wrap SearchResults in Documents carrying the result via metadata.

        The original SearchResult round-trips in ``_search_result`` so the
        reranked order can be mapped back without rebuilding from scratch.
        """
        return [
            Document(page_content=r.content, metadata={**r.metadata, "_search_result": r})
            for r in results
        ]

    def _apply_rerank(
        self, query: str, results: list[SearchResult], k: int
    ) -> list[SearchResult]:
        """Shared rerank flow: validate config, rerank, re-inject scores.

        Sync call — async callers wrap in ``asyncio.to_thread``.
        """
        reranker = self.reranker
        if reranker is None:
            raise SearchError(
                "Reranker not configured. Set SEMSEARCH_RERANKER__BASE_URL "
                "and SEMSEARCH_RERANKER__MODEL in .env"
            )

        docs = self._rerank_docs(results)
        reranked_docs = reranker.rerank(query, docs, top_n=k)

        # Build new results from reranked docs
        reranked_results = []
        for doc in reranked_docs:
            sr = doc.metadata.pop("_search_result")
            sr.metadata["rerank_score"] = doc.metadata.get("rerank_score")
            reranked_results.append(sr)
        return reranked_results
