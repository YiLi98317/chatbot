from __future__ import annotations

import typer
from rich import print as rprint
from sqlalchemy import create_engine
from tqdm import tqdm
from qdrant_client import QdrantClient

from chatbot.config import get_settings
from chatbot.embeddings.provider import embed_text
from chatbot.sql.reader import select_rows
from chatbot.sql.row_to_doc import row_to_text, doc_id
from chatbot.vectorstore.qdrant_store import QdrantStore
from chatbot.retrieval.normalize import detect_lang

app = typer.Typer(add_completion=False)


@app.command()
def ingest_sql(
    table: str = typer.Option(None, "--table", "-t", help="Source table name"),
    since: str = typer.Option(None, "--since", help="ISO date/time for incremental sync"),
    where: str = typer.Option(None, "--where", help="Additional SQL WHERE clause (use carefully)"),
    limit: int = typer.Option(None, "--limit", help="Max rows to process"),
    collection: str = typer.Option(None, "--collection", "-c", help="Qdrant collection name"),
    embed_model: str = typer.Option(
        "nomic-embed-text", "--embed-model", help="Ollama embedding model"
    ),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate collection before ingest"),
    pk_col: str = typer.Option(
        None, "--pk-col", help="Primary key column name (overrides SQL_PK from .env)"
    ),
    updated_at_col: str = typer.Option(
        None, "--updated-at-col", help="Updated-at column (overrides SQL_UPDATED_AT from .env)"
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size for Qdrant upsert"),
):
    """
    Ingest rows from a SQL database into Qdrant using deterministic IDs and incremental sync.
    """
    settings = get_settings()
    db_uri = settings.db_uri
    if not db_uri:
        raise typer.BadParameter(
            "DB_URI is not set. Configure it in your .env (e.g. mysql+pymysql://user:pass@host:3306/db), "
            "or set MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD so DB_URI can be derived."
        )

    table_name = table or settings.sql_table
    if not table_name:
        raise typer.BadParameter("Table name is required (--table) or set SQL_TABLE in .env.")

    collection_name = collection or settings.default_collection
    pk_name = pk_col or settings.sql_pk
    updated_at_name = updated_at_col or settings.sql_updated_at
    rprint(
        f"[bold]SQL ingest[/bold] from [cyan]{db_uri}[/cyan] table [bold]{table_name}[/bold] into [bold]{collection_name}[/bold]"
    )
    rprint(f"[dim]PK: {pk_name}   Updated-at: {updated_at_name or '(none)'}[/dim]")

    engine = create_engine(db_uri)
    rows = list(
        select_rows(
            engine=engine,
            table=table_name,
            pk_col=pk_name,
            updated_at_col=updated_at_name,
            since=since,
            where=where,
            limit=limit,
        )
    )
    if not rows:
        rprint("[yellow]No rows matched the criteria.[/yellow]")
        raise typer.Exit(code=0)

    texts, embeddings, metadatas, ids = [], [], [], []
    for row in tqdm(rows, desc="Embedding rows"):
        text = row_to_text(table_name, row)
        lang = detect_lang(text)
        emb = embed_text(
            text,
            provider=settings.embed_provider,
            model=embed_model,
            ollama_base_url=settings.ollama_base_url,
        )
        texts.append(text)
        embeddings.append(emb)
        pk_value = row[pk_name]
        updated_at_value = row.get(updated_at_name, "")
        ids.append(doc_id(table_name, pk_value, updated_at_value))
        metadatas.append(
            {
                "table": table_name,
                "pk": pk_value,
                "updated_at": str(updated_at_value),
                "source": f"db:{table_name}:{pk_value}",
                "lang": lang,
                "source_type": "sql",
            }
        )

    vector_size = len(embeddings[0])
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=120.0)
    store = QdrantStore(client, base_url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if recreate:
        try:
            if hasattr(client, "collection_exists") and client.collection_exists(
                collection_name=collection_name
            ):  # type: ignore[attr-defined]
                client.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
        # Ensure payload indexes for filters like "table" and "lang"
        try:
            store.ensure_default_payload_indexes(collection_name)
        except Exception:
            pass
    else:
        store.ensure_collection(collection_name, vector_size)
        try:
            store.ensure_default_payload_indexes(collection_name)
        except Exception:
            pass
    rprint("Uploading to Qdrant...")
    store.upsert_texts(
        collection_name,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
        batch_size=batch_size,
    )
    rprint("[green]SQL ingestion completed.[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
