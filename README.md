# RAG + MCP Assistant

A production-ready Python project demonstrating **all key concepts** of:
- **RAG** (Retrieval-Augmented Generation)
- **MCP** (Model Context Protocol)

Built with FastAPI, LangChain, ChromaDB, and the official Anthropic MCP SDK.

> **Chat UI** is available at `http://localhost:9000/gotoassistant/` — ask questions, see answers with syntax-highlighted code and source references, all from your browser.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                           │
│                                                              │
│  Documents → Loader → Splitter → Embeddings → Vector Store  │
│                                      ↓                       │
│  Query → [Query Expansion] → Retriever → [Reranker] →        │
│          [Contextual Compressor] → LLM → Answer + Sources   │
└──────────────────────────────────────────────────────────────┘
                          ↑ ↓
┌──────────────────────────────────────────────────────────────┐
│                       MCP Server                             │
│                                                              │
│  Tools     → search_documents, ask_question, ingest_document │
│  Resources → documents://list, collection://stats           │
│  Prompts   → rag_answer, document_analysis, qa_evaluation   │
│  Sampling  → request LLM inference via MCP client           │
└──────────────────────────────────────────────────────────────┘
```

---

## RAG Concepts Covered

| Concept | File | Description |
|---|---|---|
| Document Loading | `src/rag/document_loader.py` | PDF, TXT, DOCX, HTML, Markdown, Web URLs |
| Text Splitting | `src/rag/text_splitter.py` | Recursive, Token-based, Semantic (embedding-aware) |
| Embeddings | `src/rag/embeddings.py` | HuggingFace (local) + OpenAI, cosine similarity |
| Vector Store | `src/rag/vector_store.py` | ChromaDB with persistence, batch indexing, metadata filters |
| Retrieval Strategies | `src/rag/retriever.py` | Similarity (k-NN), MMR (diversity), Hybrid (BM25 + dense) |
| Query Expansion | `src/rag/query_expansion.py` | Multi-Query + HyDE (Hypothetical Document Embeddings) |
| Re-ranking | `src/rag/reranker.py` | FlashRank (local cross-encoder) + Cohere API |
| Contextual Compression | `src/rag/context_compression.py` | EmbeddingsFilter, LLMChainExtractor, Pipeline compressor |
| Full Pipeline | `src/rag/pipeline.py` | Sync, async, streaming, conversational (multi-turn) |
| Prompt Templates | `src/llm/prompt_templates.py` | RAG answer, condense-question, summarise, Q&A generation |

## MCP Concepts Covered

| Concept | File | Description |
|---|---|---|
| MCP Server | `src/mcp/server.py` | FastMCP with lifespan, stdio & SSE transports |
| Tools | `src/mcp/tools.py` | 8 tools with annotations (readOnly, destructive, idempotent) |
| Resources | `src/mcp/resources.py` | Static + dynamic (URI template) resources |
| Prompts | `src/mcp/prompts.py` | 4 reusable prompt templates with arguments |
| MCP Client | `src/mcp/client.py` | Stdio + SSE client; list/call tools, read resources, get prompts |
| Sampling | `src/mcp/server.py` | Server requests LLM inference from the MCP client |
| Roots | `src/mcp/server.py` | Filesystem boundary declaration |
| Resource Subscriptions | `src/mcp/client.py` | Subscribe to resource change notifications |
| Logging | `src/mcp/server.py` | Structured logs forwarded to MCP client |
| Tool Annotations | `src/mcp/tools.py` | readOnly, destructive, idempotent flags |

---

## Project Structure

```
rag-mcp-assistant/
├── main.py                          # Entry point
├── pyproject.toml                   # Dependencies
├── docker-compose.yml               # Docker setup
├── .env.example                     # Environment variables template
│
├── src/
│   ├── config.py                    # Pydantic settings
│   ├── rag/
│   │   ├── document_loader.py       # Multi-format document loading
│   │   ├── text_splitter.py         # Recursive / Token / Semantic splitting
│   │   ├── embeddings.py            # HuggingFace + OpenAI embeddings
│   │   ├── vector_store.py          # ChromaDB vector store
│   │   ├── retriever.py             # Similarity / MMR / Hybrid retrieval
│   │   ├── query_expansion.py       # Multi-Query + HyDE
│   │   ├── reranker.py              # FlashRank + Cohere reranking
│   │   ├── context_compression.py   # Contextual compression strategies
│   │   └── pipeline.py              # Full RAG pipeline orchestrator
│   ├── llm/
│   │   ├── base.py                  # LLM factory (OpenAI / Anthropic / Ollama)
│   │   └── prompt_templates.py      # Centralised prompt templates
│   ├── mcp/
│   │   ├── server.py                # FastMCP server (stdio + SSE)
│   │   ├── tools.py                 # 8 MCP tools
│   │   ├── resources.py             # 5 MCP resources
│   │   ├── prompts.py               # 4 MCP prompt templates
│   │   └── client.py                # MCP client (stdio + SSE)
│   └── api/
│       ├── app.py                   # FastAPI application factory (root: /gotoassistant)
│       ├── static/
│       │   └── index.html           # Chat UI (served at /gotoassistant/)
│       └── routes/
│           ├── health.py            # /health, /ready
│           ├── documents.py         # /documents/* (ingest, stats)
│           ├── rag.py               # /rag/* (query, stream, conversational)
│           └── mcp.py               # /mcp/* (tools, resources, prompts)
│
├── data/
│   └── documents/                   # Sample documents for ingestion
│
├── scripts/
│   ├── ingest_documents.py          # CLI ingestion tool
│   └── demo_rag.py                  # End-to-end demo script
│
└── tests/
    ├── conftest.py                  # Shared fixtures
    ├── test_api.py                  # FastAPI route tests
    ├── test_rag/
    │   └── test_pipeline.py         # RAG component tests
    └── test_mcp/
        └── test_server.py           # MCP tools/resources/prompts tests
```

---

## Quick Start

### 1. Clone and install

> **Python 3.12+** is supported. If you get a `BackendUnavailable` error, run `pip install --upgrade setuptools pip` first.

```bash
git clone https://github.com/paresh53/rag-mcp-assistant.git
cd rag-mcp-assistant
pip install --upgrade setuptools pip
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (get one at https://platform.openai.com/api-keys)
```

### 3. Start the API server

```bash
python main.py
```

Then open:
- **Chat UI** → `http://localhost:9000/gotoassistant/`
- **Swagger docs** → `http://localhost:9000/gotoassistant/docs`
- **Health check** → `http://localhost:9000/gotoassistant/health`

### 4. Ingest documents

Indexing a URL (e.g. IBM FileNet docs):
```bash
curl -X POST http://localhost:9000/gotoassistant/documents/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.ibm.com/docs/en/filenet-p8-platform/5.2.0?topic=documents-working"}'
```

Or from the command line:
```bash
python scripts/ingest_documents.py data/documents/
```

### 5. Run a quick demo

```bash
python scripts/demo_rag.py --question "What is the Model Context Protocol?"
```

### 6. Start the MCP server (separate terminal)

```bash
# Over SSE (HTTP):
MCP_TRANSPORT=sse python -m src.mcp.server

# Over stdio (for Claude Desktop):
MCP_TRANSPORT=stdio python -m src.mcp.server
```

---

## Using with Claude Desktop

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

Claude Desktop will then have access to all 8 tools and 5 resources.

---

## Chat UI

A built-in browser-based chat interface is served at `GET /gotoassistant/`.

**Features:**
- Ask questions in natural language and get answers with formatted markdown
- Syntax-highlighted code blocks with one-click **Copy** button
- Source chips showing which documents each answer came from
- Sidebar controls: retrieval strategy (MMR / Similarity / Hybrid), number of docs (k), query expansion, re-ranking, and compression toggles
- Live connection status indicator
- Example starter questions on the welcome screen

---

## API Endpoints

All endpoints are prefixed with `/gotoassistant`. Full interactive docs at `/gotoassistant/docs`.

### UI
| Method | Path | Description |
|--------|------|-------------|
| GET | `/gotoassistant/` | Chat UI |

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/gotoassistant/health` | Liveness check |
| GET | `/gotoassistant/ready` | Readiness check |

### Documents
| Method | Path | Description |
|--------|------|-------------|
| POST | `/gotoassistant/documents/ingest/url` | Ingest a URL |
| POST | `/gotoassistant/documents/ingest/file` | Upload and ingest a file |
| GET | `/gotoassistant/documents/stats` | Vector store statistics |
| DELETE | `/gotoassistant/documents/collection` | Clear a collection |

### RAG
| Method | Path | Description |
|--------|------|-------------|
| POST | `/gotoassistant/rag/query` | Full RAG query (all strategies) |
| POST | `/gotoassistant/rag/query/stream` | Streaming RAG (SSE) |
| POST | `/gotoassistant/rag/query/conversational` | Multi-turn conversational RAG |
| GET | `/gotoassistant/rag/search` | Raw vector search (no generation) |

### MCP Bridge
| Method | Path | Description |
|--------|------|-------------|
| GET | `/gotoassistant/mcp/tools` | List all MCP tools |
| POST | `/gotoassistant/mcp/tools/call` | Call any MCP tool |
| GET | `/gotoassistant/mcp/resources` | List all MCP resources |
| GET | `/gotoassistant/mcp/resources/read` | Read a resource by URI |
| GET | `/gotoassistant/mcp/prompts` | List all MCP prompts |
| POST | `/gotoassistant/mcp/prompts/get` | Get a filled prompt |

---

## Running Tests

```bash
pytest -v
# With coverage:
pytest --cov=src --cov-report=html
```

---

## Docker

```bash
docker-compose up --build
```

Services:
- **api** → http://localhost:9000 (Chat UI at `/gotoassistant/`)
- **mcp_server** → http://localhost:9001
- **chromadb** → http://localhost:9002

---

## Configuration Reference

All settings are read from environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` \| `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model name |
| `RETRIEVAL_STRATEGY` | `mmr` | `similarity` \| `mmr` \| `hybrid` |
| `RERANK_ENABLED` | `true` | Enable FlashRank reranking |
| `RERANK_TOP_N` | `3` | Documents after reranking |
| `QUERY_EXPANSION_ENABLED` | `true` | Enable multi-query expansion |
| `MCP_TRANSPORT` | `sse` | `stdio` \| `sse` |
| `API_PORT` | `9000` | API server port |
| `MCP_PORT` | `9001` | MCP SSE server port |

---

## License

MIT
