# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Model (LLM)
responses by grounding them in external, verifiable knowledge. Instead of relying solely on the
parameters baked into the model during training, RAG dynamically retrieves relevant information
from a knowledge base at inference time and provides it as context to the LLM.

## Why RAG Matters

Large language models are trained on static datasets with a knowledge cutoff date. They cannot:
- Access proprietary or private documents
- Know about events after their training cutoff
- Cite specific sources reliably
- Avoid "hallucinations" on topics with sparse training data

RAG solves all of these problems by connecting LLMs to live, structured knowledge bases.

## The RAG Pipeline

A complete RAG system has two phases:

### Phase 1: Indexing (Offline)

1. **Document Loading** — Ingest source documents from files, databases, or APIs.
   Supported formats include PDF, Word documents, HTML, Markdown, and plain text.

2. **Text Splitting** — Break large documents into smaller, semantically coherent chunks.
   Typical chunk sizes range from 256 to 1024 tokens. Overlapping chunks prevent
   important context from being split across boundaries.

3. **Embedding** — Convert each chunk into a dense vector representation using an
   embedding model. Text with similar meaning will have vectors that are close together
   in the embedding space. Popular models include OpenAI text-embedding-3-small and
   BAAI/bge-small-en-v1.5.

4. **Vector Storage** — Store embeddings and their corresponding text in a vector database.
   ChromaDB, FAISS, Pinecone, and Weaviate are popular choices.

### Phase 2: Retrieval + Generation (Online)

5. **Query Processing** — Optionally expand, rephrase, or augment the user query.
   Techniques include multi-query expansion and HyDE (Hypothetical Document Embeddings).

6. **Retrieval** — Embed the query and perform similarity search against the vector store.
   Common strategies:
   - **Similarity Search**: Pure cosine/dot-product k-NN
   - **MMR (Maximum Marginal Relevance)**: Balance relevance + diversity
   - **Hybrid**: Combine dense (embedding) and sparse (BM25 keyword) retrieval

7. **Re-ranking** — Apply a more powerful cross-encoder model to re-order retrieved chunks.
   Cross-encoders jointly encode query + document pairs, achieving better precision
   at the cost of additional latency.

8. **Contextual Compression** — Filter retrieved chunks to include only the most
   relevant sentences, reducing prompt length and noise.

9. **Generation** — Construct a prompt combining the user query + retrieved context
   and pass it to the LLM. The LLM generates a grounded, cited response.

## Model Context Protocol (MCP)

MCP is an open standard created by Anthropic that provides a universal interface
between AI assistants and external data sources and tools. It is like a USB-C port
for AI — a single, standardised protocol that allows any AI model to connect to
any tool or data source.

### MCP Architecture

MCP uses a client-server model:

- **MCP Host**: The AI application (e.g., Claude Desktop, your custom app)
- **MCP Client**: Component within the host that communicates with MCP servers
- **MCP Server**: Lightweight process that exposes tools, resources, and prompts

### MCP Primitives

**Tools**: Executable functions the LLM can call to perform actions.
Each tool has a name, description, and JSON Schema input specification.
```
search_documents(query: str, k: int = 5) -> dict
ask_question(question: str) -> dict
ingest_document(source: str) -> dict
```

**Resources**: Read-only data sources accessible via URI.
Used for context, configuration, or current state information.
```
documents://list          — all indexed documents
documents://{source}      — content of a specific document
collection://stats        — vector store statistics
```

**Prompts**: Reusable, parameterised prompt templates.
Clients can request and fill prompts without knowing their implementation.
```
rag_answer(question, context)
document_analysis(document, depth)
qa_evaluation(question, answer, reference)
```

**Sampling**: The server can request the client LLM to perform inference.
This enables complex server-side reasoning without requiring an API key on the server.

### MCP Transport

MCP supports two transports:
- **stdio**: Server runs as a subprocess communicating via standard input/output.
  Used by Claude Desktop and local integrations.
- **SSE (Server-Sent Events)**: Server exposes an HTTP endpoint.
  Used for web integrations and remote deployments.

## Key Technical Concepts

### Embedding Space
An embedding maps text to a point in a high-dimensional vector space (typically 384–3072 dimensions).
The Euclidean distance or cosine similarity between points reflects semantic similarity.

### HNSW Index
Hierarchical Navigable Small World graphs are the algorithm behind efficient approximate
nearest-neighbour search in vector databases. ChromaDB uses HNSW by default, enabling
sub-second retrieval over millions of documents.

### Cross-Encoder vs Bi-Encoder
- **Bi-encoders** (used in retrieval): Encode query and document independently into vectors.
  Fast — query vector can be cached or computed once, then compared against pre-computed document vectors.
- **Cross-encoders** (used in re-ranking): Jointly encode query + document as a single input.
  Much more accurate but cannot pre-compute — requires a forward pass per (query, doc) pair.

### Reciprocal Rank Fusion (RRF)
Used in hybrid retrieval to combine rankings from BM25 and dense retrieval.
RRF score = Σ 1 / (k + rank_i) where k is a smoothing constant (typically 60).
This approach is robust to differences in score scales between the two systems.

### HyDE (Hypothetical Document Embeddings)
Instead of embedding the short user query (e.g., "What is RAG?"), HyDE asks the LLM to
first write a hypothetical answer ("RAG is a technique that..."), then embeds that.
This bridges the distribution gap between query style and document style.
