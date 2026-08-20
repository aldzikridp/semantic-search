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
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from semsearch.config import Settings
from semsearch.errors import SemSearchError
from semsearch.service import SemanticSearchService


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
        yield
        # Only close if we created it (caller owns pre-built services)
        if service is None:
            # Run close in thread pool since it uses asyncio.new_event_loop()
            await asyncio.to_thread(svc.close)

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
        try:
            # Run sync service method in thread pool to avoid blocking event loop
            # during network calls (embedding API, reranker, database)
            results = await asyncio.to_thread(
                svc.search,
                query=req.query,
                k=req.k,
                filter=req.filter,
                rerank=req.rerank,
            )
            return {
                "query": req.query,
                "k": req.k,
                "filter": req.filter,
                "reranked": req.rerank,
                "results": [r.model_dump() for r in results],
            }
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Stats ----

    @app.get("/stats")
    async def stats(request: Request):
        svc = _get_svc(request)
        try:
            # Run sync service method in thread pool
            return await asyncio.to_thread(svc.stats)
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
