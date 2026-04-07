# Technical Details — RAG + MCP Assistant

This document explains **every tool, library, architectural pattern, and design decision** used in this project — why each was chosen and what role it plays.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Web Framework — FastAPI + Uvicorn](#2-web-framework--fastapi--uvicorn)
3. [LLM Integration — OpenAI via LangChain](#3-llm-integration--openai-via-langchain)
4. [RAG Pipeline — LangChain LCEL](#4-rag-pipeline--langchain-lcel)
5. [Document Loading — Multi-format Ingestion](#5-document-loading--multi-format-ingestion)
6. [Web Crawler — Playwright Headless Chromium](#6-web-crawler--playwright-headless-chromium)
7. [Text Splitting — Chunking Strategies](#7-text-splitting--chunking-strategies)
8. [Embeddings — HuggingFace + OpenAI](#8-embeddings--huggingface--openai)
9. [Vector Store — ChromaDB](#9-vector-store--chromadb)
10. [Retrieval Strategies — Similarity, MMR, Hybrid](#10-retrieval-strategies--similarity-mmr-hybrid)
11. [Query Expansion — Multi-Query + HyDE](#11-query-expansion--multi-query--hyde)
12. [Re-ranking — FlashRank Cross-Encoder](#12-re-ranking--flashrank-cross-encoder)
13. [Contextual Compression](#13-contextual-compression)
14. [MCP Server — Model Context Protocol](#14-mcp-server--model-context-protocol)
15. [Configuration — Pydantic Settings](#15-configuration--pydantic-settings)
16. [Structured Logging — structlog](#16-structured-logging--structlog)
17. [Retry Logic — Tenacity](#17-retry-logic--tenacity)
18. [Containerisation — Docker + Docker Compose](#18-containerisation--docker--docker-compose)
19. [Deployment — Railway + PORT Handling](#19-deployment--railway--port-handling)
20. [Frontend — Vanilla JS Chat UI](#20-frontend--vanilla-js-chat-ui)
21. [Dependency Management — pyproject.toml](#21-dependency-management--pyprojecttoml)

---

## 1. High-Level Architecture

```
Browser / API Client
        │
        ▼
 FastAPI (port 9000)  ←─ root_path: /gotoassistant
        │
        ├── GET  /             → Serves chat UI (static/index.html)
        ├── POST /rag/query    → RAG Pipeline
        ├── POST /documents/*  → Ingestion (URL, file, crawl)
        └── GET  /mcp/*        → MCP bridge routes
                │
                ▼
        RAG Pipeline
        ┌─────────────────────────────────────────┐
        │ Query                                   │
        │   → [Query Expansion]   (Multi-Query)   │
        │   → Retriever           (MMR / Sim / Hybrid) │
        │   → [Reranker]          (FlashRank)     │
        │   → [Compressor]        (EmbeddingFilter) │
        │   → LLM Prompt + Chain  (GPT-4o-mini)   │
        │   → Structured Response + Sources        │
        └─────────────────────────────────────────┘
                │
        ChromaDB (port 9002)   ← persistent vector store
        MCP Server (port 9001) ← SSE transport
```

The system is designed as a set of **interchangeable components**. Each layer can be swapped independently — e.g. swap OpenAI for Anthropic, ChromaDB for Pinecone, FlashRank for Cohere — without touching the pipeline logic.

---

## 2. Web Framework — FastAPI + Uvicorn

**Libraries:** `fastapi>=0.110.0`, `uvicorn[standard]>=0.29.0`, `httpx>=0.27.0`

**File:** `src/api/app.py`

### Why FastAPI?
FastAPI was chosen over Flask or Django for three specific reasons:

1. **Native async support** — RAG pipelines involve I/O-bound operations (LLM API calls, vector DB queries, HTTP fetches). FastAPI's `async def` endpoints let multiple requests run concurrently without blocking, which is critical under real load.

2. **Automatic OpenAPI docs** — Every route automatically generates interactive Swagger UI at `/gotoassistant/docs` and ReDoc at `/gotoassistant/redoc`. This is essential for a demo/API-first project — users can test every endpoint without writing a single line of client code.

3. **Pydantic request/response validation** — Request bodies are validated automatically. If a required field is missing or the wrong type, FastAPI returns a structured 422 error before any code runs. This eliminates an entire class of runtime bugs.

### Why `root_path="/gotoassistant"`?
When deployed behind a reverse proxy (like Railway's routing layer or nginx), the application is mounted at a sub-path. Setting `root_path` makes FastAPI generate correct URLs in redirects and Swagger UI without requiring nginx configuration. The `servers=[{"url": "/gotoassistant"}]` setting ensures the Swagger "Try it out" button sends requests to the right path.

### Why Uvicorn?
Uvicorn is an ASGI server — it implements Python's Asynchronous Server Gateway Interface. FastAPI requires ASGI (not WSGI like gunicorn). The `[standard]` extra installs `uvloop` (faster event loop on Linux) and `httptools` (faster HTTP parsing), giving ~40% better throughput than the default asyncio event loop.

### SSE (Server-Sent Events)
The `/documents/ingest/crawl` endpoint uses `StreamingResponse` with SSE format to stream crawl progress live to the browser. SSE was chosen over WebSockets because:
- It's unidirectional (server → client) — perfect for progress updates
- Works over plain HTTP/1.1 — no protocol upgrade needed
- Simpler to implement and debug than WebSockets

---

## 3. LLM Integration — OpenAI via LangChain

**Libraries:** `openai>=2.0.0`, `langchain-openai>=1.0.0`, `langchain-community>=0.3.0`

**File:** `src/llm/base.py`

### Why LangChain wrappers instead of the raw OpenAI SDK?
The raw `openai` Python SDK is perfectly capable of calling GPT-4o-mini. However, LangChain's `ChatOpenAI` wrapper provides:

1. **`.invoke()` / `.ainvoke()` / `.stream()` / `.astream()`** — a unified interface. This means the same pipeline code runs with OpenAI, Anthropic, or a local Ollama model by changing one config line. The `build_llm()` factory in `src/llm/base.py` reads `LLM_PROVIDER` from env and returns the appropriate LangChain object.

2. **LCEL composability** — LangChain Expression Language (`|` operator) lets you compose chains: `prompt | llm | StrOutputParser()`. This is significantly more readable and maintainable than manually calling each step.

3. **Structured output** (future-ready) — LangChain's `.with_structured_output()` makes it trivial to add JSON-schema-validated LLM responses.

### Why GPT-4o-mini as default?
`gpt-4o-mini` provides the best price/performance ratio for RAG:
- It is instruction-tuned for question-answering and follows the RAG prompt reliably
- It supports 128K context window — important when retrieval returns many large chunks
- Cost is ~95% cheaper than `gpt-4o` for typical query volumes
- Latency is 2-4x faster

### Fallback answer
The pipeline contains a `_fallback_answer()` function. If `OPENAI_API_KEY` is not set (e.g. someone deploys without configuring it), the system still returns the retrieved document content directly instead of crashing. This makes the application degrade gracefully.

---

## 4. RAG Pipeline — LangChain LCEL

**Library:** `langchain>=1.0.0`, `langchain-core>=1.0.0`

**File:** `src/rag/pipeline.py`

### What is RAG?
Retrieval-Augmented Generation adds an external knowledge base to an LLM. Instead of relying solely on what the model "memorised" during training, the system:
1. Converts the user's question into an embedding vector
2. Searches a vector database for the most relevant document chunks
3. Injects those chunks into the LLM prompt as context
4. Asks the LLM to answer *only* using that context

This eliminates hallucination for domain-specific knowledge and keeps answers up-to-date without retraining the model.

### Why LCEL (`|` chaining)?
LangChain Expression Language composes steps as a pipeline:
```python
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"question": q, "context": ctx})
```
This is not just syntactic sugar — LCEL:
- Automatically supports `.stream()` and `.astream()` on the whole chain
- Enables LangSmith tracing for every step
- Allows individual step substitution without rewriting the chain

### `RAGPipeline` class design
The pipeline is constructed once per request with configurable options (strategy, k, rerank, compression, query expansion). This stateless-per-request design is intentional: it avoids shared mutable state between concurrent requests, which would cause race conditions in an async server.

### Sync vs Async
Both `query()` (sync) and `aquery()` (async) are provided. The FastAPI route uses `aquery()` to avoid blocking the event loop. The sync version is exposed for scripts and testing.

### Conversational RAG (`query_with_history`)
`ConversationalRetrievalChain` adds a "condense question" step: given the chat history and follow-up question, it first rewrites the follow-up into a standalone question (e.g. *"What about its API?"* → *"What is the API for FileNet P8?"*) before running retrieval. This is necessary because vector search needs a self-contained query.

---

## 5. Document Loading — Multi-format Ingestion

**Libraries:** `pypdf>=4.2.0`, `python-docx>=1.1.0`, `unstructured>=0.14.0`, `beautifulsoup4>=4.12.0`, `lxml>=5.2.0`, `langchain-community`

**File:** `src/rag/document_loader.py`

### Why multiple loaders?
Different document formats require different parsing strategies:

| Format | Library | Why |
|---|---|---|
| PDF | `pypdf` via `PyPDFLoader` | Pure Python, no system deps, handles multi-page PDFs and extracts page metadata |
| DOCX | `python-docx` | Reads `.docx` natively — Word XML format requires deep parsing that `unstructured` handles well |
| TXT / Markdown | `TextLoader` | Raw text, no parsing needed |
| HTML (local file) | `BSHTMLLoader` + `beautifulsoup4` | Strips HTML tags, keeps prose content; `lxml` is used as the fast HTML parser backend |
| Web URL | `WebBaseLoader` | Fetches via HTTP + BeautifulSoup in one step; suitable for static HTML pages |

### Why `unstructured`?
`unstructured` is a document understanding library that does layout analysis — it identifies headings, lists, tables, and body paragraphs separately. For PDFs with complex layouts (two-column articles, tables), it produces much cleaner chunks than raw text extraction.

### Why `lxml`?
`lxml` is a C-based XML/HTML parser. It is 5-10x faster than Python's built-in `html.parser` and handles malformed HTML more robustly. It is used as the parser backend for BeautifulSoup wherever HTML is processed.

---

## 6. Web Crawler — Playwright Headless Chromium

**Library:** `playwright` (installed separately, not in `pyproject.toml` to keep the core install lean)

**File:** `src/rag/crawler.py`

### Why was a custom crawler needed?
LangChain's `WebBaseLoader` (and all urllib-based loaders) fetch raw HTML and parse it with BeautifulSoup. This works for traditional server-rendered pages. However, modern documentation sites like **IBM Docs, React-based SPAs, Angular apps** render their content entirely in JavaScript. A plain HTTP fetch returns a hollow HTML shell — `<div id="root"></div>` — with zero readable text.

**Confirmed failure:** IBM Docs at `https://www.ibm.com/docs/en/filenet-p8-platform/5.7.0` returned `text_bits: 0, links: 0` from BeautifulSoup.

### Why Playwright and not Selenium or Puppeteer?
- **Playwright** has an official Python SDK with both sync and async APIs
- It ships its own Chromium binaries — no system Chrome installation needed
- It is significantly faster than Selenium (uses CDP protocol directly, no WebDriver overhead)
- `playwright.sync_api` matches the crawler's synchronous thread model perfectly
- Puppeteer is JavaScript-only

### Auto-detection logic
The crawler probes the first page with a plain HTTP fetch. If BeautifulSoup finds fewer than 5 meaningful text chunks (the `_JS_NEEDS_THRESHOLD`), it switches to headless Chromium automatically. Static sites (GitHub Docs, man pages, plain HTML) use the fast urllib path; SPAs use the Playwright path. This keeps crawling fast for the common case.

### Why `wait_until="domcontentloaded"` + content selector wait?
`domcontentloaded` fires as soon as the DOM is parsed. For SPAs, React/Angular then runs its hydration pass. An additional `page.wait_for_selector("main, article, .content, #content")` call waits until actual content elements appear, ensuring the dynamic render is complete before extracting HTML. `networkidle` was avoided because some sites make background API calls indefinitely.

### Resource blocking
Images, fonts, videos, and audio are blocked via `page.route("**/*.{png,jpg,gif,woff,...}", abort)`. These resources add zero extractable text but can consume 60-80% of page load time. Blocking them makes crawling 3-5x faster.

### BFS (Breadth-First Search)
The crawler uses BFS with a `deque` instead of DFS (recursion). BFS explores links level by level, which means the most directly linked pages (typically the most important) are indexed first. It also naturally respects the `max_pages` limit at a predictable depth. Recursion-based DFS risks hitting Python's default recursion limit on large sites.

---

## 7. Text Splitting — Chunking Strategies

**Library:** `langchain-core`, `tiktoken>=0.7.0`

**File:** `src/rag/text_splitter.py`

### Why split documents into chunks?
LLMs have a fixed context window. A 300-page PDF cannot be injected verbatim into a prompt. More importantly, embedding models produce a single vector per chunk — a very long document produces one averaged embedding that dilutes the signal of any specific concept. Splitting into 512-token chunks means each embedding directly represents one idea.

### Three splitting strategies

**1. Recursive Character Splitter** (default)
Splits on `\n\n` (paragraphs) first, then `\n` (lines), then ` ` (words), then characters. This respects natural document structure — it only breaks mid-sentence if there is no other option. `chunk_size=512` tokens with `chunk_overlap=64` tokens is the sweet spot: small enough for precise retrieval, with 64-token overlap ensuring sentences at chunk boundaries are not lost.

**2. Token-based splitter**
Uses `tiktoken` to count tokens in exactly the same way OpenAI's API counts them. Guarantees no chunk ever exceeds the embedding model's token limit. Used for PDFs where whitespace structure is unreliable.

**3. Semantic splitter** (embedding-aware)
Splits based on cosine similarity between consecutive sentences. When the meaning shifts (e.g. a new section starts), a split is inserted. This produces semantically coherent chunks at the cost of variable chunk sizes and higher preprocessing time. Best for long narrative documents.

### Why 512 tokens?
- OpenAI's `text-embedding-3-small` and HuggingFace's `BAAI/bge-small-en-v1.5` both have a 512-token max input. Larger chunks would be truncated silently.
- 512 tokens (~350 words) is enough to contain a complete concept with surrounding context.
- Smaller chunks (128 tokens) cause loss of context; larger chunks (1024 tokens) dilute retrieval precision.

---

## 8. Embeddings — HuggingFace + OpenAI

**Libraries:** `sentence-transformers>=4.0.0`, `langchain-huggingface>=1.0.0`, `langchain-openai>=1.0.0`

**File:** `src/rag/embeddings.py`

### What are embeddings?
An embedding model converts text into a dense numeric vector (e.g. 384 dimensions for `bge-small`). Texts with similar meaning produce vectors that are close in space (high cosine similarity). This enables semantic search: "What is the FileNet API?" finds chunks about "Content Engine programming interface" even without shared keywords.

### Why `BAAI/bge-small-en-v1.5` as default?
- **Runs locally** — no API key, no network latency, no cost per embedding
- **MTEB Leaderboard top-5** for its size class — excellent retrieval quality
- **Only 33M parameters** — embeds 512-token chunks in ~5ms on CPU, negligible memory
- **384-dimension vectors** — compact enough that ChromaDB index stays fast even with 100K chunks

### Why keep OpenAI embeddings as an option?
`text-embedding-3-small`'s 1536-dimension vectors achieve marginally better retrieval on English text. For production at scale with large corpora, the quality improvement may justify the cost (~$0.02 per 1M tokens). The provider is a single env var change: `EMBEDDING_PROVIDER=openai`.

### Singleton caching (`build_embedding_model`)
The embedding model is loaded once at FastAPI startup (in the `lifespan` hook) and cached as a module-level singleton. Loading a 33M-parameter model takes ~2 seconds — doing this on every request would be unacceptable.

---

## 9. Vector Store — ChromaDB

**Libraries:** `chromadb>=1.0.0`, `langchain-chroma>=1.0.0`

**File:** `src/rag/vector_store.py`

### What does ChromaDB do?
ChromaDB is an open-source embedding database. It stores:
- The raw text of each chunk
- The embedding vector for each chunk
- Metadata (source file, page number, crawl URL)

And provides:
- Approximate nearest-neighbour (ANN) search via HNSW index — finding the top-k most similar vectors in milliseconds even with millions of entries
- Exact metadata filtering (e.g. "only search documents from this URL")
- Persistence to disk

### Why ChromaDB over Pinecone, Weaviate, Qdrant?
- **Zero infrastructure** — runs in-process as a library. No server, no Docker image, no API key for local development. One import and it works.
- **Persistent by default** — data is written to `./data/chroma_db` (a SQLite + HNSW file) and survives process restarts.
- **LangChain native integration** — `langchain-chroma` wraps it seamlessly as a `VectorStore`.
- **Free and self-hosted** — suitable for the demo's Railway deployment without a separate managed vector DB service.

### Collections
Each ingestion can be targeted to a named collection (default: `rag_documents`). Collections partition the vector space — you can have separate collections per client, per topic, or per data source. Retrieval is always scoped to a single collection.

### `as_langchain_retriever`
This factory function wraps the ChromaDB collection as a LangChain `BaseRetriever`. The `search_type` parameter toggles between `"similarity"` and `"mmr"` — ChromaDB implements both natively in its HNSW index.

---

## 10. Retrieval Strategies — Similarity, MMR, Hybrid

**Libraries:** `rank-bm25>=0.2.2`, `langchain-community`, `langchain-core`

**File:** `src/rag/retriever.py`

### Why three strategies?

**Similarity (k-NN)**
Finds the k vectors closest to the query vector by cosine distance. Fast and accurate. Weakness: if the top-5 results are all from the same paragraph (near-duplicate chunks), the LLM sees highly redundant context and misses important but slightly less similar content.

**MMR (Maximum Marginal Relevance)**
Invented specifically to solve the redundancy problem. Algorithm:
1. Fetch `fetch_k = k × 4` candidates from the vector store
2. Select the most similar candidate
3. For each subsequent selection, choose the candidate that maximises: `λ × similarity(query) - (1-λ) × max_similarity(selected_docs)`
4. `lambda_mult=0.5` balances relevance vs. diversity

Used as the **default strategy** because real documentation sites have near-duplicate chunks (e.g. repeated warnings, boilerplate headers). MMR ensures the LLM gets diverse context.

**Hybrid (BM25 + Dense)**
BM25 is a classic TF-IDF-based ranking algorithm. It scores documents by exact term frequency. Combined with dense vector search via **Reciprocal Rank Fusion (RRF)**:
- BM25 excels when users use exact terminology ("FileNet CPE API")
- Dense embeddings excel for semantic queries ("how do I store files programmatically")
- RRF merges both rank lists without needing score normalisation
- Weights `[0.4, 0.6]` give slightly more weight to semantic understanding

### Why `EnsembleRetriever`?
LangChain's `EnsembleRetriever` implements RRF natively. It calls both retrievers, merges results, deduplicates, and returns a unified ranked list — all in a single `.invoke()` call.

---

## 11. Query Expansion — Multi-Query + HyDE

**Library:** `langchain-core`, `langchain-openai`

**File:** `src/rag/query_expansion.py`

### Problem being solved
A user's natural language question may not contain the exact words used in the indexed documents. "How does authentication work?" might not match chunks that say "login procedure" or "identity verification". This vocabulary mismatch is a known retrieval failure mode.

### Multi-Query Retriever
Sends the user's question to the LLM with a prompt asking it to generate 3 alternative phrasings. Each phrasing is independently sent to the retriever, and results are union-merged. This dramatically improves recall — at least one phrasing is likely to hit the right chunks.

**Example:**
- Original: *"How to upload files?"*
- Expansions: *"Document ingestion procedure"*, *"Add files to Content Engine"*, *"Store content in FileNet"*

### HyDE (Hypothetical Document Embeddings)
A more advanced technique: instead of generating query variations, HyDE asks the LLM to **write a hypothetical answer** to the question, then embeds that answer and searches for similar real documents. The embedding of a plausible answer is more similar to real document embeddings than the embedding of the question itself.

HyDE is especially powerful for question-answering over technical documentation where questions are terse but documents are verbose.

---

## 12. Re-ranking — FlashRank Cross-Encoder

**Library:** `flashrank>=0.2.0`

**File:** `src/rag/reranker.py`

### Why re-rank after retrieval?
Vector similarity search is an approximation — embedding models project meaning into a lower-dimensional space, which loses nuance. A retrieved chunk might be about the same general topic as the query but not actually answer it.

A **cross-encoder** re-ranker takes `(query, passage)` pairs and scores them together — it can attend to the full interaction between the two texts, unlike bi-encoders (embedding models) which encode them independently.

### Why FlashRank?
- **Runs entirely locally** — no API calls, no latency from network I/O
- **ONNX quantised models** — the `ms-marco-MiniLM-L-12-v2` model runs in ~10ms per batch even on CPU, because it uses INT8 quantisation via ONNX Runtime
- **MS-MARCO trained** — trained specifically on passage re-ranking for question-answering, matching this project's use case exactly
- **Zero additional cost** — compared to Cohere Rerank API which charges per re-rank call

### Effect on quality
In practice, re-ranking typically improves NDCG@3 (the quality of the top 3 results) by 15-25% over pure vector retrieval. The top 3 chunks sent to the LLM are more precise, leading to better-grounded answers.

### Cohere Rerank API (alternative)
The code supports Cohere's `rerank-english-v3.0` API as an alternative. It provides marginally better quality than FlashRank but requires an API key and adds ~200ms network latency. Available via `COHERE_API_KEY` env var.

---

## 13. Contextual Compression

**Library:** `langchain-core`, `langchain-community`

**File:** `src/rag/context_compression.py`

### Problem
Retrieved chunks are fixed-size (512 tokens). A chunk might be 90% relevant boilerplate with only one relevant sentence. Sending all 512 tokens to the LLM wastes context window and can confuse the model.

### EmbeddingsFilter
The simplest compression strategy: compute the cosine similarity between the query embedding and each sentence in the chunk. Sentences below a `similarity_threshold` (default 0.76) are dropped. This keeps only sentences that are semantically relevant to the query, often reducing chunk length by 50-70%.

### LLMChainExtractor
More powerful but slower: sends each `(query, chunk)` pair to the LLM with a prompt asking it to extract only the relevant sentences. Produces the cleanest results but adds one LLM call per chunk (latency cost).

### Why disabled by default?
Compression adds latency — EmbeddingsFilter adds ~20ms per chunk (extra embedding call), LLMChainExtractor adds ~500ms per chunk (LLM call). For most queries the gain does not justify the added latency. It is available as an opt-in pipeline step.

---

## 14. MCP Server — Model Context Protocol

**Library:** `mcp>=1.0.0` (Anthropic's official Python SDK)

**File:** `src/mcp/server.py`, `src/mcp/tools.py`, `src/mcp/resources.py`, `src/mcp/prompts.py`

### What is MCP?
The **Model Context Protocol** is an open standard (created by Anthropic) for connecting AI models to external tools and data sources. Think of it as USB-C for AI — a standardised way for any LLM client (Claude Desktop, Claude API, custom agents) to discover and call tools.

### Why MCP in addition to REST?
The REST API (`/rag/query`) is for browser/frontend use. MCP is for **AI-to-AI communication** — it allows Claude Desktop or any MCP-compatible agent to:
- Discover what tools are available (tool listing)
- Call `search_documents`, `ask_question`, `ingest_document` programmatically
- Read resources like `documents://list` (a live list of indexed documents)
- Request the LLM to fill in prompt templates

### Why `FastMCP` (SSE transport)?
`FastMCP` is the high-level API in the `mcp` SDK. For HTTP deployment, SSE (Server-Sent Events) transport is used so the MCP server can run on a separate port (9001) and be reached by remote clients. `stdio` transport is also supported for Claude Desktop integration where the MCP server runs as a subprocess.

### Tool annotations
MCP tools carry semantic annotations:
- `readOnly=True` — tool does not modify state (safe for agents to call freely)
- `destructive=True` — tool permanently modifies data (agent should confirm with user)
- `idempotent=True` — calling the same tool twice has the same effect as calling it once

These annotations let MCP clients apply appropriate safety policies automatically.

---

## 15. Configuration — Pydantic Settings

**Library:** `pydantic>=2.7.0`, `pydantic-settings>=2.3.0`, `python-dotenv>=1.0.0`

**File:** `src/config.py`

### Why Pydantic Settings over `os.getenv()`?
`os.getenv("API_PORT", "9000")` returns a string — you must manually cast to `int`. If the env var is malformed (e.g. `API_PORT=abc`), the error appears at the point of use, not at startup.

`pydantic-settings`:
- Parses and validates all env vars at startup — the app fails fast with a descriptive error if config is wrong
- Auto-coerces types: `API_PORT` is declared as `int`, it reads the string `"9000"` and returns the integer `9000`
- Reads from `.env` file in development, but env vars take precedence (cloud deployment)
- Generates documentation of all available config via `Settings.model_json_schema()`

### `AliasChoices` for PORT
Railway (and Render, Fly.io) injects a `PORT` env var. Local dev uses `API_PORT`. `AliasChoices("PORT", "API_PORT")` tries `PORT` first, then `API_PORT`, then falls back to `9000`. This handles both deployment environments with zero code change.

---

## 16. Structured Logging — structlog

**Library:** `structlog>=24.1.0`, `rich>=13.7.0`

**Files:** Throughout `src/`

### Why structlog over Python's built-in `logging`?
Standard `logging` produces flat text strings: `"2026-04-07 10:00:00 INFO Starting server"`. In production, logs are ingested by tools like Datadog, Grafana Loki, or Railway's log viewer, which work best with **structured JSON**:
```json
{"event": "api_starting", "host": "0.0.0.0", "port": 9000, "level": "info"}
```
This allows filtering by field (`level=error`), aggregation (`count by event`), and correlation (`trace_id`).

`structlog` emits structured key-value log events. In development (terminal), it renders them as colourised human-readable output via `rich`. In production (container), it outputs JSON. Same code, different renderers.

---

## 17. Retry Logic — Tenacity

**Library:** `tenacity>=8.3.0`

**File:** `src/config/DataSourceConfig.java` (khel_hisab) / used in crawler and LLM calls

### Why Tenacity?
Network calls (OpenAI API, ChromaDB, URL fetches) fail transiently — rate limits, momentary connectivity issues, cloud API blips. Without retries, a single transient failure returns a 500 error to the user.

`tenacity` provides:
- Exponential backoff with jitter: retries at 1s, 2s, 4s, 8s — jitter prevents thundering herd
- `retry_if_exception_type()` — only retry on network errors, not on validation errors
- `stop_after_attempt(3)` — gives up after 3 retries to prevent infinite loops
- `wait_random_exponential()` — adds random delay to spread retry load

---

## 18. Containerisation — Docker + Docker Compose

**Files:** `Dockerfile`, `docker-compose.yml`

### Dockerfile design decisions

```dockerfile
FROM python:3.11-slim
```
`python:3.11-slim` — uses Debian slim base (~50MB) rather than the full Debian image (~900MB) or Alpine (~5MB). Alpine's `musl libc` causes compatibility issues with `numpy`, `sentence-transformers`, and `chromadb` (which use C extensions compiled against glibc). `slim` is the right compromise.

**System packages installed:**
- `build-essential` — C compiler for packages that build native extensions from source
- `libmagic1` — MIME type detection used by `unstructured` to route documents to the right parser
- `poppler-utils` — PDF rendering tools used by `unstructured` for advanced PDF parsing
- `tesseract-ocr` — OCR engine; `unstructured` calls it for scanned PDFs
- `libatk1.0-0`, `libgbm1`, `libnss3`, etc. — X11/graphics system libraries required by Playwright's headless Chromium binary

**Playwright install inside Docker:**
```dockerfile
RUN pip install --no-cache-dir -e . && \
    pip install playwright && \
    python -m playwright install chromium
```
Playwright downloads its own Chromium binary during `install chromium` — it does not use the system Chrome. This ensures a consistent, pinned version of the browser regardless of what is installed on the host.

**Shell-form CMD:**
```dockerfile
CMD sh -c "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-9000}"
```
The exec-form `CMD ["uvicorn", ..., "--port", "${PORT:-9000}"]` does **not** perform shell variable substitution — `${PORT:-9000}` would be passed literally as a string. Shell form (`sh -c "..."`) expands env vars correctly at runtime. This is required for Railway's injected `PORT` variable to be respected.

### Docker Compose services
Three services are defined:
1. **api** — FastAPI app on port 9000
2. **mcp_server** — MCP SSE server on port 9001 (separate process for isolation)
3. **chromadb** — ChromaDB running in HTTP server mode on port 9002

In local development, the API talks to ChromaDB over HTTP (`localhost:9002`) — this mimics a cloud deployment where the vector store would be a separate service.

---

## 19. Deployment — Railway + PORT Handling

### Why Railway?
Railway was selected as the recommended deployment platform because:
- Auto-detects `Dockerfile` and builds without any config files (`railway.json`, `Procfile`, etc.)
- Provides free persistent **Volumes** (mounted at `/app/data`) for ChromaDB persistence
- Injects `PORT` env var automatically — the app always listens on the right port
- GitHub integration: every push to `main` triggers a redeploy automatically
- Free tier: 500 CPU-hours/month — sufficient for demo workloads

### The PORT problem explained
Railway does not expose your app on whatever port it listens on. It reverse-proxies through its internal routing layer and assigns a random high port to your container. It tells your app what port to use via the `PORT` env var. If your app ignores `PORT` and hardcodes `9000`, the routing fails and your app is unreachable.

The fix applied in this project:
1. Dockerfile CMD uses `${PORT:-9000}` (shell form for expansion)
2. `src/config.py` uses `AliasChoices("PORT", "API_PORT")` so both env var names work

---

## 20. Frontend — Vanilla JS Chat UI

**Libraries used (CDN):** `marked.js`, `highlight.js`

**File:** `src/api/static/index.html`

### Why no React/Vue/Angular?
A SPA framework would require a build pipeline (webpack/vite), TypeScript compilation, `node_modules`, and a separate dev server. For a single-page chat UI, this is engineering overload. The HTML file is served directly as a static file by FastAPI — no build step, no separate process, zero npm dependencies.

The file is served at `GET /gotoassistant/` via FastAPI's `StaticFiles` mount plus a dedicated route:
```python
app.mount("/static", StaticFiles(directory="src/api/static"))

@app.get("/")
async def root():
    return FileResponse("src/api/static/index.html")
```

### marked.js
Renders Markdown in the LLM's answer to formatted HTML. LLMs naturally produce Markdown (bold, code blocks, bullet lists) — without a renderer these appear as raw `**text**` and ` ``` ` in the browser.

### highlight.js
Applies syntax highlighting to code blocks within the rendered Markdown. The `github-dark` theme matches the UI's dark color scheme. Loaded from CDN — no local copy needed.

### SSE streaming for crawl progress
The "Add Knowledge" crawl feature uses the Fetch API's `ReadableStream` to read the SSE response incrementally:
```js
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  // parse SSE lines, update progress bar
}
```
This gives real-time feedback as pages are crawled, making long operations (100-page crawl = ~2 minutes) feel responsive rather than appearing frozen.

### Why the sidebar was simplified for public deployment
The original sidebar exposed Strategy, k-slider, Pipeline toggles, and Collection field. These are meaningful for developers exploring RAG concepts but confusing for end-users. Hardcoded sensible defaults (MMR, k=5, rerank on, query expansion on) give optimal quality without exposing complexity. The only user-facing control is "Add Knowledge" — the one action users actually need.

---

## 21. Dependency Management — pyproject.toml

**Standard:** PEP 517/518

### Why `pyproject.toml` over `requirements.txt`?
`requirements.txt` is a flat list of pinned packages — it cannot express:
- Version bounds (`>=1.0.0` means "at least version 1")
- Optional dependency groups (dev tools vs production deps)
- Build system metadata (how the package is built)
- Project metadata (name, version, author, Python requirement)

`pyproject.toml` is the modern Python packaging standard (PEP 517). `setuptools` reads it to build the package, `pip install -e .` reads it to install in editable mode, and tools like `ruff` and `mypy` read their config from the `[tool.*]` sections.

### Dev extras
```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff", "mypy", ...]
```
`pip install -e ".[dev]"` installs dev tools without polluting the production Docker image (which only runs `pip install -e .`).

### Key version constraints explained

| Package | Minimum | Why that floor |
|---|---|---|
| `langchain>=1.0.0` | 1.0 | v1.0 was a major breaking refactor — earlier versions have incompatible LCEL API |
| `chromadb>=1.0.0` | 1.0 | v1.0 changed the collection API; older versions have incompatible method signatures |
| `pydantic>=2.7.0` | 2.x | Pydantic v2 is 5-17x faster than v1 and has breaking API changes; langchain 1.x requires Pydantic v2 |
| `openai>=2.0.0` | 2.0 | v2 changed to `client.chat.completions.create()` pattern; v1 API is completely different |
| `sentence-transformers>=4.0.0` | 4.0 | v4 added the `encode()` batching improvements needed for efficient bulk embedding |

---

## Summary of Key Design Principles

1. **Swappable components** — every layer (LLM, embeddings, vector store, reranker) is behind a factory function and controlled by env vars. The system is a demo of alternatives, not a lock-in.

2. **Fail gracefully** — if the LLM key is missing, the app still returns retrieved content. If the crawler fails on one page, it logs and continues. No hard crashes from missing optional components.

3. **Async-first** — FastAPI + async pipeline + async ChromaDB calls means the server handles concurrent requests without blocking. One slow LLM call does not freeze all other users.

4. **Zero-cost defaults** — HuggingFace embeddings (local), FlashRank reranker (local), ChromaDB (local file). The only paid component is the LLM API call. This makes the project runnable for free beyond OpenAI usage.

5. **Stateless requests** — the `RAGPipeline` is constructed fresh per request. No shared mutable state between concurrent requests. Safe for horizontal scaling.
