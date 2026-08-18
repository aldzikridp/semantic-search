# TASK-014: CLI Implementation

> **Phase**: 9 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-008, TASK-009, TASK-010, TASK-011, TASK-012, TASK-013
> **Blocks**: TASK-021

## Objective

Implement all `typer` subcommands for the `semsearch` CLI entry point.

## File to Create

### `src/semsearch/cli.py`

## Implementation

### App Setup

```python
import typer
import json
from pathlib import Path
from typing import Optional

app = typer.Typer(help="Semantic search over local documents")
```

### 1. `init` Command

```bash
semsearch init [--recreate] [--yes]
```

```python
@app.command()
def init(
    recreate: bool = typer.Option(False, "--recreate", help="Drop and recreate table"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation for --recreate"),
):
    """Create / migrate the chunks table."""
    settings = Settings()
    if recreate and not yes:
        if not typer.confirm("This will DELETE all data. Continue?"):
            raise typer.Exit(1)
    with SemanticSearchService.from_settings(settings) as svc:
        svc.init_schema(recreate=recreate)
        typer.echo("Schema initialized.")
```

### 2. `ingest` Command

```bash
semsearch ingest <path> [--force] [--provider TYPE] [--provider-model MODEL] ...
```

```python
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
):
    """Ingest a single file."""
    settings = _apply_provider_overrides(
        Settings(), provider, provider_model, provider_order,
        provider_allow_fallbacks, provider_ignore, provider_base_url, provider_api_key
    )
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.ingest(path, reembed_unchanged=force)
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
```

### 3. `ingest-dir` Command

```bash
semsearch ingest-dir <dir_path> [--glob PATTERN] [--exclude PATTERN]... [--prune] [--dry-run] [--no-continue-on-error] [--follow-symlinks] [--force]
```

```python
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
):
    """Recursively ingest all supported files in a directory."""
    settings = Settings()
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.ingest_dir(
            dir_path, glob=glob, exclude=exclude,
            reembed_unchanged=force, continue_on_error=continue_on_error,
            follow_symlinks=follow_symlinks, prune=prune, prune_dry_run=dry_run,
        )
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
```

### 4. `search` Command

```bash
semsearch search <query> [--k N] [--filter JSON]
```

```python
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, help="Top-k results"),
    filter: Optional[str] = typer.Option(None, help="PGVectorStore filter dict (JSON)"),
):
    """Run a similarity search."""
    settings = Settings()
    filter_dict = json.loads(filter) if filter else None
    with SemanticSearchService.from_settings(settings) as svc:
        results = svc.search(query, k=k, filter=filter_dict)
        output = {
            "query": query,
            "k": k,
            "filter": filter_dict,
            "results": [r.model_dump() for r in results],
        }
        typer.echo(json.dumps(output, indent=2, default=str))
```

### 5. `delete` Command

```bash
semsearch delete [--filter JSON] [--all] [--yes]
```

```python
@app.command()
def delete(
    filter: Optional[str] = typer.Option(None, help="Filter dict (JSON)"),
    all: bool = typer.Option(False, "--all", help="Delete everything"),
    yes: bool = typer.Option(False, "--yes", help="Confirm --all"),
):
    """Delete chunks matching a filter (or --all)."""
    if all and not yes:
        typer.echo("Use --yes to confirm deletion of ALL data.", err=True)
        raise typer.Exit(1)

    filter_dict = {} if all else (json.loads(filter) if filter else None)
    if filter_dict is None:
        typer.echo("Provide --filter or --all", err=True)
        raise typer.Exit(1)

    settings = Settings()
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.delete(filter_dict)
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
```

### 6. `stats` Command

```bash
semsearch stats
```

```python
@app.command()
def stats():
    """Show table stats."""
    settings = Settings()
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.stats()
        typer.echo(json.dumps(result, indent=2, default=str))
```

### 7. `reingest` Command

```bash
semsearch reingest <path>
```

```python
@app.command()
def reingest(
    path: Path = typer.Argument(..., help="File to reingest"),
):
    """Delete + ingest in one step."""
    settings = Settings()
    with SemanticSearchService.from_settings(settings) as svc:
        result = svc.reingest(path)
        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
```

### Error Handling Wrapper

```python
def main():
    try:
        app()
    except SemSearchError as e:
        typer.echo(json.dumps({"error": str(e), "type": type(e).__name__}), err=True)
        raise typer.Exit(1)
    except Exception:
        raise  # Let unexpected exceptions propagate (exit code 2)
```

### Provider Override Helper

```python
def _apply_provider_overrides(settings, provider, model, order, allow_fallbacks, ignore, base_url, api_key):
    """Apply CLI provider overrides to settings."""
    if provider:
        settings.embedding_provider.type = provider
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
```

## Critical Notes

1. **All output is JSON to stdout** — Errors go to stderr
2. **`--filter` accepts JSON string** — Must parse with `json.loads`
3. **`--all` requires `--yes`** — Safety mechanism for destructive operations
4. **Provider overrides are optional** — Only applied when provided
5. **Context manager used everywhere** — Ensures connection cleanup

## Verification (CLI Tests C-1 to C-6)

- [ ] C-1: `semsearch init` x2 — both succeed
- [ ] C-2: `semsearch ingest ./sample.txt` — stdout is valid JSON
- [ ] C-3: `semsearch search "test" --k 3` — stdout has results array
- [ ] C-4: `semsearch delete --filter '{"source": "..."}'` — stdout has deleted_count
- [ ] C-5: `semsearch delete --all` — exits non-zero (missing --yes)
- [ ] C-6: `semsearch stats` — stdout matches expected schema
