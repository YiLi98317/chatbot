from __future__ import annotations

import typer
from rich import print as rprint
from pathlib import Path

from chatbot.config import get_settings
from chatbot.retrieval.lexical import normalize_query, lexical_lookup

app = typer.Typer(add_completion=False)


def _resolve_db_uri() -> str | None:
    settings = get_settings()
    if settings.db_uri:
        return settings.db_uri
    default_sqlite = Path(settings.data_dir) / "chinook" / "Chinook_Sqlite.sqlite"
    if default_sqlite.exists():
        return f"sqlite:///{default_sqlite}"
    return None


@app.command()
def run(
    query: str = typer.Argument(..., help="Query string for lexical lookup"),
    limit: int = typer.Option(5, "--limit", "-k", help="Max results"),
):
    db_uri = _resolve_db_uri()
    if not db_uri:
        rprint("[red]No DB URI configured and no default Chinook SQLite found.[/red]")
        raise typer.Exit(code=1)

    normalized = normalize_query(query)
    rprint(f"[bold]Normalized[/bold]: {normalized!r}")

    lq = query.lower()
    if "album" in lq:
        preferred_tables = ["Album", "Track", "Artist"]
    elif "artist" in lq or "band" in lq:
        preferred_tables = ["Artist", "Track", "Album"]
    else:
        preferred_tables = ["Track", "Album", "Artist"]

    hits = lexical_lookup(db_uri=db_uri, candidate=normalized, preferred_tables=preferred_tables, limit=limit)
    if not hits:
        rprint("[yellow]No lexical hits. Would fall back to vector search.[/yellow]")
        return
    for i, h in enumerate(hits, 1):
        rprint(
            f"[dim]{i}.[/dim] table=[cyan]{h.table}[/cyan] pk={h.pk} "
            f"title={h.title_value!r} conf={h.confidence:.2f}"
        )
    decision = "skip Qdrant" if max(h.confidence for h in hits) >= 0.8 else "fallback to Qdrant"
    rprint(f"[bold]Decision[/bold]: {decision}")


def main():
    app()


if __name__ == "__main__":
    main()


