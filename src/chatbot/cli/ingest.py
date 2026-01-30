from __future__ import annotations

import typer
from rich import print as rprint
from tqdm import tqdm
from qdrant_client import QdrantClient

from chatbot.config import get_settings
from chatbot.embeddings.provider import embed_texts
from chatbot.ingest.loader import load_and_chunk
from chatbot.vectorstore.qdrant_store import QdrantStore

app = typer.Typer(add_completion=False)


@app.command()
def ingest(
    collection: str = typer.Option(None, "--collection", "-c", help="Qdrant collection name"),
    chunk_size: int = typer.Option(800, "--chunk-size", help="Characters per chunk"),
    chunk_overlap: int = typer.Option(
        150, "--chunk-overlap", help="Characters overlap between chunks"
    ),
    model: str = typer.Option("nomic-embed-text", "--model", "-m", help="Ollama embedding model"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate collection before ingest"),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size for Qdrant upsert"),
):
    """
    Ingest files into Qdrant: load -> chunk -> embed -> upsert.
    """
    settings = get_settings()
    collection_name = collection or settings.default_collection
    data_dir = settings.data_dir
    rprint(f"[bold]Ingesting from[/bold] {data_dir} into collection [bold]{collection_name}[/bold]")

    # Load and chunk documents
    docs = load_and_chunk(data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not docs:
        rprint("[yellow]No documents found to ingest.[/yellow]")
        raise typer.Exit(code=0)

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    # Embed (provider-based: Ollama or SentenceTransformers)
    rprint(f"Embedding {len(texts)} chunks with model: {model} (provider={settings.embed_provider})")
    embeddings = embed_texts(
        texts,
        provider=settings.embed_provider,
        model=model,
        ollama_base_url=settings.ollama_base_url,
        batch_size=32,
    )
    vector_size = len(embeddings[0])

    # Qdrant setup
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
        try:
            store.ensure_default_payload_indexes(collection_name)
        except Exception:
            pass
    else:
        store.ensure_collection(collection_name, vector_size)

    # Upsert in batches
    rprint("Uploading to Qdrant...")
    store.upsert_texts(
        collection_name,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        batch_size=batch_size,
    )
    rprint("[green]Ingestion completed.[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
