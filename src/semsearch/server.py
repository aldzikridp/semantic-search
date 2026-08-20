"""FastAPI HTTP server for semsearch (read-only query mode).

Usage:
    semsearch serve --host 0.0.0.0 --port 8383

Endpoints are read-only. Use CLI commands for data modification:
    semsearch ingest <file>
    semsearch ingest-dir <dir>
    semsearch delete --filter '{"source": "..."}'
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from semsearch.config import Settings
from semsearch.errors import SemSearchError
from semsearch.service import SemanticSearchService

logger = logging.getLogger("semsearch.server")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter: dict[str, Any] | None = None
    rerank: bool = False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings,
    *,
    service: SemanticSearchService | None = None,
) -> FastAPI:
    """Create the FastAPI app with a shared SemanticSearchService.

    The server is read-only — use CLI commands for data modification.

    Sync service methods (search, stats) are run in a thread pool to avoid
    blocking the async event loop during network calls (embedding API,
    reranker, database).

    Args:
        settings: Application settings.
        service: Optional pre-built service (for testing). When provided the
            lifespan skips creating a new service and uses this one instead.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = service or SemanticSearchService.from_settings(settings)
        app.state.service = svc
        logger.info("Semsearch service started")
        yield
        # Only close if we created it (caller owns pre-built services)
        if service is None:
            # Run close in thread pool since it uses asyncio.new_event_loop()
            await asyncio.to_thread(svc.close)
            logger.info("Semsearch service stopped")

    app = FastAPI(
        title="semsearch",
        description="Semantic search HTTP API (read-only query mode)",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _get_svc(request: Request) -> SemanticSearchService:
        return request.app.state.service  # type: ignore[no-any-return]

    # ---- Health ----

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---- Search ----

    @app.post("/search")
    async def search(req: SearchRequest, request: Request):
        svc = _get_svc(request)
        start_time = time.monotonic()
        try:
            # Use async search to avoid sync/async deadlock with asyncpg
            results = await svc.asearch(
                query=req.query,
                k=req.k,
                filter=req.filter,
                rerank=req.rerank,
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Search completed: query=%r, k=%d, rerank=%s, results=%d, elapsed=%.1fms",
                req.query[:50], req.k, req.rerank, len(results), elapsed_ms,
            )
            return {
                "query": req.query,
                "k": req.k,
                "filter": req.filter,
                "reranked": req.rerank,
                "results": [r.model_dump() for r in results],
            }
        except ValueError as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning("Search validation error: %s (%.1fms)", e, elapsed_ms)
            raise HTTPException(status_code=422, detail=str(e))
        except SemSearchError as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Search failed: %s (%.1fms)\n%s",
                e, elapsed_ms, traceback.format_exc(),
            )
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Unexpected error during search: %s (%.1fms)\n%s",
                e, elapsed_ms, traceback.format_exc(),
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    # ---- Stats ----

    @app.get("/stats")
    async def stats(request: Request):
        svc = _get_svc(request)
        try:
            # Run sync service method in thread pool
            return await asyncio.to_thread(svc.stats)
        except SemSearchError as e:
            logger.error("Stats failed: %s\n%s", e, traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error("Unexpected error during stats: %s\n%s", e, traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
