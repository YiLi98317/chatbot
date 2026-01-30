from __future__ import annotations

import typer
from rich import print as rprint
from pathlib import Path

from chatbot.config import get_settings
from chatbot.retrieval.entity_resolver import resolve_entity, rebuild_fts_index, ensure_fts_index

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
    query: str = typer.Argument(..., help="Query to test entity resolver"),
    top: int = typer.Option(5, "--top", "-k", help="Top candidates to show"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild FTS index before running"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Print resolver debug logs"),
):
    """
    Run the entity resolver against a single query and print decision and top hits.
    """
    db_uri = _resolve_db_uri()
    if not db_uri:
        rprint("[red]No DB URI configured and no default Chinook SQLite found.[/red]")
        raise typer.Exit(code=1)
    if rebuild:
        rprint("[dim]Rebuilding FTS index...[/dim]")
        rebuild_fts_index(db_uri)
    else:
        ensure_fts_index(db_uri)

    res = resolve_entity(db_uri=db_uri, query=query, limit=50, debug=debug)
    decision = res.get("decision")
    hits = res.get("hits", [])[:top]
    rprint(f"[bold]Decision[/bold]: {decision}")
    if not hits:
        rprint("[yellow]No candidates; would fall back to Qdrant.[/yellow]")
        return
    rprint("[bold]Top candidates[/bold]:")
    for i, h in enumerate(hits, 1):
        rprint(
            f"[dim]{i}.[/dim] {h['entity_type']}:{h['entity_id']}  "
            f"{h['name']!r}  fuzzy={h['fuzzy']:.1f}  bm25={h['bm25']:.3f}  final={h['final']:.2f}"
        )


@app.command("acceptance")
def acceptance_suite():
    """
    Run acceptance tests A–E and print decisions/scores.
    """
    db_uri = _resolve_db_uri()
    if not db_uri:
        rprint("[red]No DB URI configured and no default Chinook SQLite found.[/red]")
        raise typer.Exit(code=1)
    ensure_fts_index(db_uri)

    tests = [
        ("A", 'Do you know the song "Fly Me To The Moon"?'),
        ("B", "Is there a song named fly somebody to the mooon?"),
        ("C", "Fly Me To The Moon"),
        ("D", "avril lavin songs"),
        ("E", "asdzxcqwe not in db"),
    ]
    for label, q in tests:
        rprint(f"\n[bold cyan]{label}.[/bold cyan] {q}")
        res = resolve_entity(db_uri=db_uri, query=q, limit=50, debug=True)
        decision = res.get("decision")
        hits = res.get("hits", [])[:5]
        rprint(f"[bold]Decision[/bold]: {decision}")
        if hits:
            for i, h in enumerate(hits, 1):
                rprint(
                    f"[dim]{i}.[/dim] {h['entity_type']}:{h['entity_id']}  "
                    f"{h['name']!r}  fuzzy={h['fuzzy']:.1f}  final={h['final']:.2f}"
                )
        else:
            rprint("[dim]No candidates[/dim]")


def main():
    app()


if __name__ == "__main__":
    main()


