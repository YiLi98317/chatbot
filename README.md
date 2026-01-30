## Chatbot (CLI-first) — Qdrant + Ollama

Deeper architecture + workflow diagrams live in `intro.md`.

**New machine?** See **[INSTALL.md](INSTALL.md)** for a clean setup from clone to run.

---

### Tech stack

- **Language/runtime**: Python (venv)
- **CLI**: Typer + Rich
- **Vector DB**: Qdrant (`qdrant-client`)
- **Models**:
  - **Embeddings**: Ollama HTTP API (`POST /api/embeddings`) *or* local SentenceTransformers (`EMBED_PROVIDER=sentence_transformers`)
  - **Generation**: Ollama HTTP API (`POST /api/generate`)
- **SQL**: SQLAlchemy (SQLite by default; MySQL supported via PyMySQL)
- **Lexical matching**: SQLite FTS5 + RapidFuzz
- **Utilities**: requests, tqdm, python-dotenv

---

### How to run

#### Requirements

- Python 3.10+ and `make`
- Ollama running locally (default `OLLAMA_BASE_URL=http://localhost:11434`)
- Qdrant (local Docker or cloud)

#### Setup

1. Create `.env` in the project root:

- `QDRANT_URL=http://localhost:6333` (or your cloud URL)
- `QDRANT_API_KEY=` (empty for local)
- `OLLAMA_BASE_URL=http://localhost:11434`
- `QDRANT_COLLECTION=chatbot_docs`
- `EMBED_PROVIDER=ollama` (or `sentence_transformers`)
- `EMBED_MODEL=nomic-embed-text`
- `CHAT_MODEL=llama3.1`
- `TOP_K=4`
- (optional for SQL ingest / resolver) `DB_URI=sqlite:///data/knowledge.db`

Chinese support (recommended settings):

- `EMBED_PROVIDER=sentence_transformers`
- `EMBED_MODEL=BAAI/bge-m3`
- `CHAT_MODEL=deepseek-chat` (via Ollama, optional but recommended for Chinese)
- `ZH_CHUNK_SIZE=1200`
- `ZH_CHUNK_OVERLAP=150`

2. Install:

```bash
make install
```

3. Pull models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

If using DeepSeek for Chinese generation:

```bash
ollama pull deepseek-chat
```

4. Start services:

```bash
# Qdrant (local)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Ollama
ollama serve
```

#### Quickstart (Chinook sample)

```bash
. ./.venv/bin/activate
python reingest_chinook.py
make chat collection=chatbot_docs
```

Debug mode:

```bash
make chat collection=chatbot_docs DEBUG=1
```

One-off query:

```bash
make query q="Do you know the song Fly Me To The Moon?" collection=chatbot_docs
```

One-off query (debug):

```bash
make query q="Do you know the song Fly Me To The Moon?" collection=chatbot_docs args="--debug"
```

#### Ingest your own files

Put `.md` / `.txt` files under `data/` (or set `DATA_DIR`), then:

```bash
make ingest collection=chatbot_docs
```

#### Ingest from SQL

Set `DB_URI` (and optionally `SQL_TABLE`, `SQL_PK`, `SQL_UPDATED_AT`), then:

```bash
make ingest-sql args="--table knowledge --since 2025-01-01 --limit 10000 --collection chatbot_docs"
```

#### Useful checks

```bash
python tests/qdrantClientConnectionTest.py
```

---

### Publishing this repo to GitHub

From the project root:

```bash
# First time only: init and add remote
git init
git remote add origin https://github.com/YiLi98317/chatbot.git

# Stage, check, commit, push
git add .
git status   # confirm only source/config; no .venv, .env, data/, *.sqlite, traces/
git commit -m "Initial commit: chatbot CLI, Qdrant + Ollama"
git branch -M main
git push -u origin main
```

If the repo is already initialized and has `origin`, skip the `git init` and `git remote add` lines. Later pushes: `git add . && git commit -m "Your message" && git push`.
