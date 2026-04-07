# RAG + MCP Assistant

A production-ready RAG (Retrieval-Augmented Generation) assistant with a browser-based chat UI.  
Paste any URL or upload a document → ask questions → get accurate, sourced answers powered by OpenAI.

**Live demo (once deployed):** `https://<your-app>.up.railway.app/gotoassistant/`

---

## What it does

1. **Crawl & ingest any website** — paste a URL (including JavaScript-rendered docs like IBM Docs, React/Angular SPAs) and the system learns the entire site automatically using a headless Chromium crawler.
2. **Upload files** — PDF, DOCX, TXT, Markdown, HTML.
3. **Ask questions** — the RAG pipeline retrieves the most relevant chunks and uses GPT-4o-mini to generate a grounded answer with source references.
4. **Streaming answers** — responses stream word-by-word in the chat UI.
5. **MCP Server** — exposes all tools/resources to Claude Desktop and other MCP clients.

---

## Deploy on Railway (free, public URL in ~8 minutes)

Railway gives you a permanent public URL like `https://rag-mcp-assistant-production.up.railway.app/gotoassistant/` at no cost.

### Step 1 — Sign up on Railway

Go to **railway.app** → click **"Start a New Project"** → sign in with GitHub.

### Step 2 — Create a new project from your GitHub repo

1. Click **"New Project"** → **"Deploy from GitHub repo"**
2. Select `paresh53/rag-mcp-assistant`
3. Railway detects the `Dockerfile` automatically and starts building (takes ~6-8 minutes first time)

### Step 3 — Add environment variables

While the build runs, click your service → **"Variables"** tab → add these one by one:

| Variable | Value | Required? |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` (your key from platform.openai.com) | ✅ Yes |
| `LLM_PROVIDER` | `openai` | ✅ Yes |
| `LLM_MODEL` | `gpt-4o-mini` | ✅ Yes |
| `EMBEDDING_PROVIDER` | `huggingface` | ✅ Yes |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | ✅ Yes |
| `CHROMA_PERSIST_DIR` | `/app/data/chroma_db` | ✅ Yes |
| `API_HOST` | `0.0.0.0` | ✅ Yes |
| `LLM_TEMPERATURE` | `0.0` | Optional |
| `RETRIEVAL_STRATEGY` | `mmr` | Optional |
| `RERANK_ENABLED` | `true` | Optional |
| `QUERY_EXPANSION_ENABLED` | `true` | Optional |

> **Note:** Do NOT set `PORT` — Railway injects it automatically. Do NOT set `API_PORT` on Railway.

After saving variables, Railway will redeploy automatically.

### Step 4 — Generate a public domain

1. In your service → **"Settings"** → **"Networking"**
2. Click **"Generate Domain"**
3. Copy your URL, e.g. `https://rag-mcp-assistant-production.up.railway.app`

**Your chat UI is live at:** `https://<your-domain>/gotoassistant/`

### Step 5 (recommended) — Add a volume for persistent storage

Without a volume, your ingested documents are lost on every redeploy.

1. In Railway dashboard → your service → **"Add Volume"**
2. Mount path: `/app/data`
3. This keeps ChromaDB data across restarts and redeployments

---

## Using the Chat UI

Open `https://<your-domain>/gotoassistant/` in any browser.

### Add knowledge (ingest documents)

**From the sidebar → "Add Knowledge" panel:**

1. **Crawl a website** — paste any URL (e.g. `https://www.ibm.com/docs/en/filenet-p8-platform/5.7.0`) → set max pages → click **"Crawl"**. Works on JavaScript SPAs (IBM Docs, React/Angular sites).

2. **Single URL** — paste a direct page URL → click **"Add"**

3. **Upload a file** — drag a PDF, DOCX, or TXT file onto the page or use the file picker (via API, see below)

### Ask questions

Type your question in the chat box and press **Enter** or click **Send**.

**Sidebar controls** (affect answer quality):
- **Strategy** — `MMR` (diverse results), `Similarity` (most relevant), `Hybrid` (BM25 + dense)
- **k** — how many document chunks to retrieve (3–10)
- **Query Expansion** — generates sub-queries to improve recall
- **Reranking** — reorders retrieved chunks for better precision
- **Compression** — strips irrelevant sentences from chunks before sending to LLM

### Example starter questions

After crawling IBM FileNet P8 docs:
- *"What are the system requirements for FileNet P8 5.7.0?"*
- *"How do I configure Content Engine security?"*
- *"What changed in the 5.7.0 release notes?"*

---

## Run locally

### Prerequisites

- Python 3.11+
- An OpenAI API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Install

```bash
git clone https://github.com/paresh53/rag-mcp-assistant.git
cd rag-mcp-assistant
pip install --upgrade setuptools pip
pip install -e ".[dev]"
pip install playwright && python -m playwright install chromium
```

### Configure

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
```

### Start

```bash
python main.py
```

Open:
- **Chat UI** → http://localhost:9000/gotoassistant/
- **API docs** → http://localhost:9000/gotoassistant/docs
- **Health check** → http://localhost:9000/gotoassistant/health

### Ingest via command line

```bash
# Ingest a folder of documents
python scripts/ingest_documents.py data/documents/

# Run a quick demo
python scripts/demo_rag.py --question "What is RAG?"
```

---

## Run with Docker (local)

```bash
# Copy and edit your env file first
cp .env.example .env   # set OPENAI_API_KEY

docker-compose up --build
```

Services:
- **Chat UI** → http://localhost:9000/gotoassistant/
- **MCP Server** → http://localhost:9001
- **ChromaDB** → http://localhost:9002

---

## API Reference

All endpoints are prefixed with `/gotoassistant`. Interactive docs at `/gotoassistant/docs`.

### Chat / RAG
| Method | Path | Description |
|---|---|---|
| POST | `/gotoassistant/rag/query` | Ask a question (full response) |
| POST | `/gotoassistant/rag/query/stream` | Ask a question (streaming SSE) |
| POST | `/gotoassistant/rag/query/conversational` | Multi-turn chat |
| GET | `/gotoassistant/rag/search` | Raw vector search (no LLM) |

### Documents
| Method | Path | Description |
|---|---|---|
| POST | `/gotoassistant/documents/ingest/url` | Ingest a single URL |
| POST | `/gotoassistant/documents/ingest/crawl` | Crawl an entire website (SSE progress stream) |
| POST | `/gotoassistant/documents/ingest/file` | Upload and ingest a file |
| GET | `/gotoassistant/documents/stats` | Vector store statistics |
| DELETE | `/gotoassistant/documents/collection` | Clear a collection |

### Health
| Method | Path | Description |
|---|---|---|
| GET | `/gotoassistant/health` | Liveness |
| GET | `/gotoassistant/ready` | Readiness |

### Example: ingest a URL via curl

```bash
curl -X POST https://<your-domain>/gotoassistant/documents/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/library/asyncio.html"}'
```

### Example: ask a question via curl

```bash
curl -X POST https://<your-domain>/gotoassistant/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does asyncio work?", "collection": ""}'
```

---

## Use with Claude Desktop (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-assistant": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/rag-mcp-assistant",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

Claude Desktop gains access to all 8 RAG tools and 5 resources.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                           │
│  Documents → Loader → Splitter → Embeddings → ChromaDB       │
│                                      ↓                       │
│  Query → [Query Expansion] → Retriever → [Reranker] →        │
│          [Contextual Compressor] → LLM → Answer + Sources    │
└──────────────────────────────────────────────────────────────┘
                          ↑ ↓
┌──────────────────────────────────────────────────────────────┐
│                       MCP Server                             │
│  Tools: search_documents, ask_question, ingest_document      │
│  Resources: documents://list, collection://stats             │
└──────────────────────────────────────────────────────────────┘
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Get from platform.openai.com |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_TEMPERATURE` | `0.0` | Response creativity (0=factual, 1=creative) |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` \| `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (free) |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector DB storage path |
| `RETRIEVAL_STRATEGY` | `mmr` | `similarity` \| `mmr` \| `hybrid` |
| `RETRIEVAL_K` | `5` | Number of chunks to retrieve |
| `RERANK_ENABLED` | `true` | Enable FlashRank reranking |
| `RERANK_TOP_N` | `3` | Chunks kept after reranking |
| `QUERY_EXPANSION_ENABLED` | `true` | Enable multi-query expansion |
| `API_PORT` | `9000` | Port (overridden by `PORT` on Railway/Render/Fly.io) |
| `MCP_PORT` | `9001` | MCP server port |

---

## License

MIT
