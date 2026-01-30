# Install on a New Machine

Use this guide to get the chatbot running from a clean clone. The repo is set up so you only commit source code and config templates; local data, venvs, and DBs stay out of git.

---

## Prerequisites

- **Python 3.10+**
- **make**
- **Docker** (optional, for local Qdrant)
- **Ollama** (for embeddings and chat)

---

## 1. Clone the repo

```bash
git clone https://github.com/YiLi98317/chatbot.git
cd chatbot
```

---

## 2. Create virtual environment and install dependencies

The Makefile uses system Python for the venv when possible (avoids issues when your shell is in a conda environment).

```bash
make install
```

This will:

- Create `.venv/` with system Python (or fallback to `python3`)
- Upgrade pip and install packages from `requirements.txt`

**If `make install` fails** (e.g. “No such file or directory” for pip):

- Remove any partial venv: `rm -rf .venv`
- Create the venv with system Python, then run make again:

  ```bash
  /usr/bin/python3 -m venv .venv   # macOS; on Linux try: python3 -m venv .venv
  make install
  ```

---

## 3. Configure environment

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `QDRANT_URL` – e.g. `http://localhost:6333` for local Qdrant
- `QDRANT_COLLECTION` – e.g. `chatbot_docs`
- `OLLAMA_BASE_URL` – e.g. `http://localhost:11434`
- `EMBED_PROVIDER` – `ollama` or `sentence_transformers`
- `EMBED_MODEL` – e.g. `nomic-embed-text`
- `CHAT_MODEL` – e.g. `llama3.1`

See `.env.example` and the main [README.md](README.md) for full options (e.g. Chinese support, SQL ingest).

---

## 4. Start services (if running locally)

**Qdrant (Docker):**

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

**Ollama:**

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1
```

---

## 5. Quick test (Chinook sample)

```bash
. ./.venv/bin/activate
python reingest_chinook.py
make chat collection=chatbot_docs
```

For more commands (ingest, query, debug), see [README.md](README.md).

---

## What is not in the repo (kept local by .gitignore)

- `.venv/` – virtual environment
- `.env` – your secrets and local URLs
- `data/` – local documents
- `*.sqlite`, `traces/` – local DBs and metrics
- `__pycache__/`, build artifacts, IDE config

So a clone is source-only; each machine creates its own venv and config.
