# RAG + MCP Assistant

A production-ready Python project demonstrating **all key concepts** of:
- **RAG** (Retrieval-Augmented Generation)
- **MCP** (Model Context Protocol)

Built with FastAPI, LangChain, ChromaDB, and the official Anthropic MCP SDK.

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
│       ├── app.py                   # FastAPI application factory
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

```bash
git clone <your-repo-url>
cd rag-mcp-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY (or use Ollama for local)
```

### 3. Ingest sample documents

```bash
python scripts/ingest_documents.py data/documents/
```

### 4. Run a quick demo

```bash
python scripts/demo_rag.py --question "What is the Model Context Protocol?"
```

### 5. Start the API server

```bash
python main.py
# API docs: http://localhost:9000/docs
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

## API Endpoints

### Documents
| Method | Path | Description |
|--------|------|-------------|
| POST | `/documents/ingest/url` | Ingest a URL |
| POST | `/documents/ingest/file` | Upload and ingest a file |
| GET | `/documents/stats` | Vector store statistics |
| DELETE | `/documents/collection` | Clear a collection |

### RAG
| Method | Path | Description |
|--------|------|-------------|
| POST | `/rag/query` | Full RAG query (all strategies) |
| POST | `/rag/query/stream` | Streaming RAG (SSE) |
| POST | `/rag/query/conversational` | Multi-turn conversational RAG |
| GET | `/rag/search` | Raw vector search (no generation) |

### MCP Bridge
| Method | Path | Description |
|--------|------|-------------|
| GET | `/mcp/tools` | List all MCP tools |
| POST | `/mcp/tools/call` | Call any MCP tool |
| GET | `/mcp/resources` | List all MCP resources |
| GET | `/mcp/resources/read` | Read a resource by URI |
| GET | `/mcp/prompts` | List all MCP prompts |
| POST | `/mcp/prompts/get` | Get a filled prompt |

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
- **api** → http://localhost:9000
- **mcp_server** → http://localhost:8001
- **chromadb** → http://localhost:8002

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
| `MCP_PORT` | `8001` | MCP SSE server port |

---

## License

MIT
