# TASK-028: FastAPI HTTP Server (`semsearch serve`)

> **Status**: Complete ✅  
> **Phase**: Performance Phase 1  
> **Depends on**: TASK-024 (benchmark harness)  
> **Blocks**: TASK-029 (before/after benchmarks)  

## Objective

Add a FastAPI HTTP server that keeps the `SemanticSearchService` warm between requests, eliminating cold-start overhead for AI agent tool calling.

## Problem

Every CLI command spins up a fresh `PGEngine` + embedder + connections (~500ms–1.5s cold start). For AI agents making repeated tool calls, this is wasteful. A `repl` command was considered but only helps humans — not AI agents doing tool calling via HTTP.

## Solution

Add `semsearch serve --host 0.0.0.0 --port 8383` that starts a FastAPI HTTP server. The `SemanticSearchService` instance lives for the lifetime of the server — zero cold-start on every request.

## Endpoints

| Method | Path | Body | Response | Description |
|--------|------|------|----------|-------------|
| `POST` | `/search` | `{"query": "...", "k": 5, "filter": {}, "rerank": false}` | `{"results": [...]}` | Similarity search |
| `GET` | `/stats` | — | `{"table": "...", "chunk_count": N, ...}` | Table statistics |
| `GET` | `/health` | — | `{"status": "ok"}` | Health check |

**Note:** The server is read-only. Use CLI commands for data modification:
```bash
semsearch ingest <file>
semsearch ingest-dir <dir>
semsearch delete --filter '{"source": "..."}'
```

## Files to Create / Modify

### New: `src/semsearch/server.py`

FastAPI app with all endpoints using `lifespan` for clean startup/shutdown.

**Key design:**
- `create_app(settings)` factory function
- `asynccontextmanager` lifespan creates/closes `SemanticSearchService`
- Pydantic request models for validation
- HTTPException for error responses
- All endpoints mirror CLI commands

```python
"""FastAPI HTTP server for semsearch.

Usage:
    semsearch serve --host 0.0.0.0 --port 8383
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semsearch.config import Settings
from semsearch.errors import SemSearchError
from semsearch.service import SemanticSearchService


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter: dict[str, Any] | None = None
    rerank: bool = False


class IngestRequest(BaseModel):
    path: str
    force: bool = False


class IngestDirRequest(BaseModel):
    dir_path: str
    glob: str = "**/*"
    exclude: list[str] | None = None
    prune: bool = False
    prune_dry_run: bool = False
    continue_on_error: bool = True
    follow_symlinks: bool = False
    force: bool = False


class DeleteRequest(BaseModel):
    filter: dict[str, Any] | None = None
    all: bool = False


def create_app(settings: Settings) -> FastAPI:
    """Create the FastAPI app with a shared SemanticSearchService."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = SemanticSearchService.from_settings(settings)
        app.state.service = svc
        yield
        svc.close()

    app = FastAPI(
        title="semsearch",
        description="Semantic search HTTP API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _get_svc(request) -> SemanticSearchService:
        return request.app.state.service

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/search")
    async def search(req: SearchRequest, request):
        svc = _get_svc(request)
        try:
            results = svc.search(
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

    @app.post("/ingest")
    async def ingest(req: IngestRequest, request):
        svc = _get_svc(request)
        try:
            result = svc.ingest(Path(req.path), reembed_unchanged=req.force)
            return result.model_dump()
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest-dir")
    async def ingest_dir(req: IngestDirRequest, request):
        svc = _get_svc(request)
        try:
            result = svc.ingest_dir(
                Path(req.dir_path),
                glob=req.glob,
                exclude=req.exclude,
                reembed_unchanged=req.force,
                continue_on_error=req.continue_on_error,
                follow_symlinks=req.follow_symlinks,
                prune=req.prune,
                prune_dry_run=req.prune_dry_run,
            )
            return result.model_dump()
        except (ValueError, SemSearchError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.delete("/delete")
    async def delete(req: DeleteRequest, request):
        svc = _get_svc(request)
        if not req.all and req.filter is None:
            raise HTTPException(status_code=422, detail="Provide filter or set all=true")
        filter_dict = {} if req.all else req.filter
        try:
            result = svc.delete(filter_dict)
            return result.model_dump()
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def stats(request):
        svc = _get_svc(request)
        try:
            return svc.stats()
        except SemSearchError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
```

### Modify: `src/semsearch/cli.py`

Add `serve` command:

```python
@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8383, help="Bind port"),
) -> None:
    """Start the HTTP server (keeps service warm between requests)."""
    import uvicorn
    from semsearch.server import create_app

    settings = get_settings(_config_path)
    application = create_app(settings)
    uvicorn.run(application, host=host, port=port)
```

### Modify: `pyproject.toml`

Add FastAPI and uvicorn to dependencies:

```toml
dependencies = [
    # ... existing deps ...
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
]
```

### New: `tests/test_server.py`

HTTP endpoint tests using `httpx.AsyncClient` with `ASGITransport`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from semsearch.server import create_app


@pytest.fixture
def app(pg_url, mock_embeddings):
    """Create test app with mock embeddings."""
    from semsearch.config import Settings, EmbeddingProviderConfig
    from semsearch.store import build_engine, init_schema
    from pydantic import SecretStr

    settings = Settings(
        database_url=pg_url,
        collection_name="semsearch_chunks_test",
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("test-key"),
        ),
    )
    engine = build_engine(settings)
    init_schema(settings, engine, mock_embeddings.dim, recreate=True)
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.json() == {"status": "ok"}


async def test_search(client):
    r = await client.post("/search", json={"query": "test", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert data["query"] == "test"
    assert data["k"] == 3


async def test_stats(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "chunk_count" in data
    assert "table" in data


async def test_search_validation_error(client):
    r = await client.post("/search", json={"query": "test", "k": 0})
    assert r.status_code == 422


async def test_delete_requires_filter(client):
    r = await client.request("DELETE", "/delete", json={})
    assert r.status_code == 422
```

## Server Lifecycle

```
semsearch serve --port 8383
  └── uvicorn starts
       └── lifespan: SemanticSearchService.from_settings(settings)
            ├── PGEngine (connection pool)
            ├── OpenAIEmbeddings (HTTP client)
            └── Reranker (if configured)
       └── Requests:
            ├── POST /search  → svc.search()  (no cold start)
            ├── POST /ingest  → svc.ingest()
            ├── GET /stats    → svc.stats()
            └── ...
       └── Shutdown: svc.close()
```

## AI Agent Integration

```python
import httpx

# Agent makes repeated calls — server stays warm
client = httpx.Client(base_url="http://localhost:8383")

# Search
results = client.post("/search", json={"query": "how to deploy", "k": 5}).json()

# Ingest
client.post("/ingest", json={"path": "/docs/new.md"})

# Stats
stats = client.get("/stats").json()
```

## Acceptance Criteria

- [x] `src/semsearch/server.py` exists with `create_app()` factory
- [x] `create_app()` accepts optional `service` parameter for testing
- [x] `semsearch serve --host 0.0.0.0 --port 8383` starts the server
- [x] `curl http://localhost:8383/health` returns `{"status": "ok"}`
- [x] `POST /search` returns search results
- [x] `POST /ingest` ingests a file
- [x] `POST /ingest-dir` ingests a directory
- [x] `DELETE /delete` deletes by filter
- [x] `GET /stats` returns table statistics
- [x] OpenAPI docs available at `http://localhost:8383/docs`
- [x] `pyproject.toml` includes `fastapi>=0.115.0` and `uvicorn[standard]>=0.30.0`
- [x] `tests/test_server.py` passes all endpoint tests
- [x] Tests use `create_app(service=svc)` pattern
- [x] Tests verify OpenAPI docs and JSON spec
- [x] All existing tests pass: `pytest -v`
