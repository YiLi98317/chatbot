# Install on a New Machine

Use this guide to get the chatbot running from a clean clone. The repo is set up so you only commit source code and config templates; local data, venvs, and DBs stay out of git.

---

## Prerequisites

- **Python 3.10+** (required for torch 2.6+ and sentence_transformers). On macOS: `brew install python@3.11` then use `python3.11`; the Makefile will prefer `python3.12`, `python3.11`, or `python3.10` when creating the venv.
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

The Makefile prefers Python 3.10+ (`python3.12`, `python3.11`, or `python3.10`) when creating the venv, then falls back to system `python3`. This avoids conda’s Python and ensures torch 2.6+ can be installed.

```bash
make install
```

This will:

- Create `.venv/` with system Python (or fallback to `python3`)
- Upgrade pip and install packages from `requirements.txt`

**Optional — PyTorch 2.6 for sentence_transformers:** Default PyPI only has torch up to 2.2.x. If you use `EMBED_PROVIDER=sentence_transformers` and hit a `torch.load` CVE error when loading models, install PyTorch 2.6+ after `make install`:

```bash
. .venv/bin/activate
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
```

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

## 4. Install Ollama (if not already installed)

Ollama runs embeddings and chat models locally. Install it before starting services.

**macOS — Option A: Download**

1. Go to [ollama.com/download](https://ollama.com/download).
2. Download **Ollama for macOS**, open the `.dmg`, and drag Ollama into Applications.
3. Open Ollama from Applications (or Spotlight). It runs in the menu bar; the server is at `http://localhost:11434` by default.

**macOS — Option B: Homebrew**

```bash
brew install ollama
```

**Other platforms:** See [ollama.com/download](https://ollama.com/download) for Linux and Windows.

**Pull models** (after Ollama is installed and running):

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

For Chinese chat you can also: `ollama pull deepseek-r1`.

---

## 5. Start services (if running locally)

**Qdrant (Docker):**

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

**Ollama:** Start the server (if not already running from the app). Pull models in step 4 if you haven’t yet.

```bash
ollama serve
```

---

## 6. Quick test (Chinook sample)

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
