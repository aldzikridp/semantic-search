"""CLI entry point for semsearch (TASK-014 will flesh this out)."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="semsearch",
    help="Semantic search service over local documents.",
    add_completion=False,
)


@app.command()
def version() -> None:
    """Show the semsearch version."""
    from semsearch import __version__

    typer.echo(f"semsearch {__version__}")


@app.command(name="help")
def help_cmd() -> None:
    """Show this help message."""
    typer.echo(app.get_help())


if __name__ == "__main__":
    app()
