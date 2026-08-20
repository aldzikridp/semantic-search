"""CLI entry point for semsearch (spec §8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import SecretStr

from semsearch.config import Settings, get_settings
from semsearch.errors import SemSearchError
from semsearch.models import BatchIngestResult, DeleteResult, IngestResult
from semsearch.service import SemanticSearchService

app = typer.Typer(
    name="semsearch",
    help="Semantic search over local documents.",
    add_completion=False,
)

# Store config path globally for commands to use
_config_path: Path | None = None


@app.callback(invoke_without_command=True)
def main(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (default: .env)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Semantic search over local documents."""
    global _config_path
    _config_path = config


def _apply_provider_overrides(
    settings: Settings,
    provider: Optional[str],
    model: Optional[str],
    order: Optional[str],
    allow_fallbacks: Optional[bool],
    ignore: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> Settings:
    """Apply CLI provider overrides to settings."""
    if provider:
        settings.embedding_provider.type = provider  # type: ignore[assignment]
    if model:
        settings.embedding_provider.model = model
    if order:
        settings.embedding_provider.provider_order = json.loads(order)
    if allow_fallbacks is not None:
        settings.embedding_provider.provider_allow_fallbacks = allow_fallbacks
    if ignore:
        settings.embedding_provider.provider_ignore = json.loads(ignore)
    if base_url:
        settings.embedding_provider.base_url = base_url
    if api_key:
        settings.embedding_provider.api_key = SecretStr(api_key)
    return settings


@app.command()
def init(
    recreate: bool = typer.Option(False, "--recreate", help="Drop and recreate table"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation for --recreate"),
) -> None:
    """Create / migrate the chunks table."""
    settings = get_settings(_config_path)
    if recreate and not yes:
        if not typer.confirm("This will DELETE all data. Continue?"):
            raise typer.Exit(1)
    with SemanticSearchService.from_settings(settings) as svc:
        svc.init_schema(recreate=recreate)
        typer.echo("Schema initialized.")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File to ingest"),
    force: bool = typer.Option(False, "--force", help="Force re-embed unchanged chunks"),
    provider: Optional[str] = typer.Option(None, help="Override embedding provider type"),
    provider_model: Optional[str] = typer.Option(None, help="Override embedding model"),
    provider_order: Optional[str] = typer.Option(None, help='OpenRouter provider.order (JSON)'),
    provider_allow_fallbacks: Optional[bool] = typer.Option(None, help="OpenRouter allow_fallbacks"),
    provider_ignore: Optional[str] = typer.Option(None, help='OpenRouter provider.ignore (JSON)'),
    provider_base_url: Optional[str] = typer.Option(None, help="Override base URL"),
    provider_api_key: Optional[str] = typer.Option(None, help="Override API key"),
) -> None:
    """Ingest a single file."""
    settings = _apply_provider_overrides(
        get_settings(_config_path), provider, provider_model, provider_order,
        provider_allow_fallbacks, provider_ignore, provider_base_url, provider_api_key,
    )
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.ingest(path, reembed_unchanged=force)
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


@app.command(name="ingest-dir")
def ingest_dir(
    dir_path: Path = typer.Argument(..., help="Directory to walk"),
    glob: str = typer.Option("**/*", help="Glob pattern for file discovery"),
    exclude: list[str] = typer.Option([], help="fnmatch patterns to skip (repeatable)"),
    prune: bool = typer.Option(False, "--prune", help="Delete orphaned chunks"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what prune would delete"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--no-continue-on-error"),
    follow_symlinks: bool = typer.Option(False, "--follow-symlinks", help="Follow symlinks"),
    force: bool = typer.Option(False, "--force", help="Force re-embed unchanged chunks"),
) -> None:
    """Recursively ingest all supported files in a directory."""
    settings = get_settings(_config_path)
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.ingest_dir(
            dir_path,
            glob=glob,
            exclude=exclude,
            reembed_unchanged=force,
            continue_on_error=continue_on_error,
            follow_symlinks=follow_symlinks,
            prune=prune,
            prune_dry_run=dry_run,
        )
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, help="Top-k results"),
    filter: Optional[str] = typer.Option(None, help="PGVectorStore filter dict (JSON)"),
    rerank: bool = typer.Option(False, "--rerank", help="Rerank results using configured reranker"),
) -> None:
    """Run a similarity search."""
    settings = get_settings(_config_path)
    filter_dict = json.loads(filter) if filter else None
    with SemanticSearchService.from_settings(settings) as svc:
        results = svc.search(query, k=k, filter=filter_dict, rerank=rerank)
        output = {
            "query": query,
            "k": k,
            "filter": filter_dict,
            "reranked": rerank,
            "results": [r.model_dump() for r in results],
        }
        typer.echo(json.dumps(output, indent=2, default=str))


@app.command()
def delete(
    filter: Optional[str] = typer.Option(None, help="Filter dict (JSON)"),
    all: bool = typer.Option(False, "--all", help="Delete everything"),
    yes: bool = typer.Option(False, "--yes", help="Confirm --all"),
) -> None:
    """Delete chunks matching a filter (or --all)."""
    if all and not yes:
        typer.echo("Use --yes to confirm deletion of ALL data.", err=True)
        raise typer.Exit(1)

    filter_dict = {} if all else (json.loads(filter) if filter else None)
    if filter_dict is None:
        typer.echo("Provide --filter or --all", err=True)
        raise typer.Exit(1)

    settings = get_settings(_config_path)
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.delete(filter_dict)
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


@app.command()
def stats() -> None:
    """Show table stats."""
    settings = get_settings(_config_path)
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.stats()
        typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def reingest(
    path: Path = typer.Argument(..., help="File to reingest"),
) -> None:
    """Delete + ingest in one step."""
    settings = get_settings(_config_path)
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.reingest(path)
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


@app.command()
def version() -> None:
    """Show the semsearch version."""
    from semsearch import __version__

    typer.echo(f"semsearch {__version__}")


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


def main() -> None:
    """Entry point with error handling."""
    try:
        app()
    except SemSearchError as e:
        typer.echo(json.dumps({"error": str(e), "type": type(e).__name__}), err=True)
        raise typer.Exit(1)
    except Exception:
        raise


if __name__ == "__main__":
    main()
