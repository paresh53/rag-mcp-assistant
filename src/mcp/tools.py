"""
MCP Concept — Tools
====================
MCP Tools are callable functions exposed to LLM clients.
The @mcp.tool() decorator registers them and generates a JSON Schema
from the function signature + docstring automatically.

Tool annotations control behaviour:
  • readOnly=True   — safe to call without user confirmation
  • destructive=True — should ask user before calling
  • idempotent=True  — calling multiple times has same effect

Tools registered here:
  1. search_documents       — semantic search over the vector store
  2. ingest_document        — add a new document to the vector store
  3. summarise_document     — summarise a document by its source path
  4. extract_keywords       — extract keywords from text
  5. generate_questions     — generate Q&A pairs for a passage
  6. get_collection_stats   — report collection size / metadata
  7. delete_document        — remove a document (destructive!)
  8. compare_documents      — compare two documents for similarity
"""
from __future__ import annotations

import logging
from typing import Annotated

logger = logging.getLogger(__name__)


def register_tools(mcp):
    """Register all tools on the given FastMCP instance."""

    # ── 1. Semantic document search ──────────────────────────────────────────

    @mcp.tool(
        description=(
            "Search indexed documents using semantic similarity. "
            "Returns the top-k most relevant passages along with source metadata."
        ),
        annotations={"readOnly": True},
    )
    def search_documents(
        query: Annotated[str, "The search query"],
        k: Annotated[int, "Number of results to return"] = 5,
        strategy: Annotated[str, "Retrieval strategy: similarity | mmr | hybrid"] = "mmr",
        collection: Annotated[str, "Vector store collection name"] = "",
    ) -> dict:
        """Search the vector store and return relevant passages."""
        from src.rag.retriever import RetrievalStrategy, build_retriever

        retriever = build_retriever(
            strategy=RetrievalStrategy(strategy),
            k=k,
            collection_name=collection or None,
        )
        docs = retriever.invoke(query)

        return {
            "query": query,
            "strategy": strategy,
            "results": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", ""),
                    "score": doc.metadata.get("score"),
                }
                for doc in docs
            ],
            "total": len(docs),
        }

    # ── 2. Ask a question (full RAG) ─────────────────────────────────────────

    @mcp.tool(
        description="Ask a question and get an answer grounded in indexed documents (full RAG).",
        annotations={"readOnly": True},
    )
    def ask_question(
        question: Annotated[str, "The question to answer"],
        strategy: Annotated[str, "Retrieval: similarity | mmr | hybrid"] = "mmr",
        enable_rerank: Annotated[bool, "Whether to rerank results"] = True,
    ) -> dict:
        """Run the full RAG pipeline and return answer + sources."""
        from src.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            retrieval_strategy=strategy,
            enable_rerank=enable_rerank,
        )
        response = pipeline.query(question)

        return {
            "answer": response.answer,
            "sources": response.sources,
            "strategy": response.retrieval_strategy,
            "reranked": response.reranked,
        }

    # ── 3. Ingest a document ──────────────────────────────────────────────────

    @mcp.tool(
        description=(
            "Ingest a file or URL into the vector store. "
            "Supports PDF, TXT, Markdown, DOCX, HTML, and web URLs."
        ),
        annotations={"readOnly": False, "idempotent": False},
    )
    def ingest_document(
        source: Annotated[str, "File path or HTTP/S URL to ingest"],
        chunk_size: Annotated[int, "Target chunk size in characters"] = 512,
        chunk_overlap: Annotated[int, "Overlap between chunks in characters"] = 64,
        collection: Annotated[str, "Target collection name"] = "",
    ) -> dict:
        """Load, chunk, embed and store a document."""
        from src.rag.document_loader import load_document
        from src.rag.text_splitter import SplitStrategy, split_documents
        from src.rag.vector_store import add_documents

        raw_docs = load_document(source)
        chunks = split_documents(
            raw_docs,
            strategy=SplitStrategy.RECURSIVE,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        ids = add_documents(chunks, collection_name=collection or None)

        return {
            "source": source,
            "raw_documents": len(raw_docs),
            "chunks_created": len(chunks),
            "ids": ids[:10],  # Return first 10 IDs
            "status": "success",
        }

    # ── 4. Summarise a document ───────────────────────────────────────────────

    @mcp.tool(
        description="Retrieve a document by source path and produce an LLM summary.",
        annotations={"readOnly": True},
    )
    def summarise_document(
        source: Annotated[str, "Source identifier (file path or URL)"],
        max_chunks: Annotated[int, "Max chunks to summarise"] = 10,
    ) -> dict:
        """Retrieve stored chunks for a source and summarise them."""
        from langchain_core.output_parsers import StrOutputParser

        from src.llm.base import build_llm
        from src.llm.prompt_templates import SUMMARISE_PROMPT
        from src.rag.vector_store import similarity_search_with_score

        # Fetch relevant chunks by using the source as the query
        results = similarity_search_with_score(source, k=max_chunks)
        doc_chunks = [doc for doc, _ in results]

        if not doc_chunks:
            return {"error": f"No document found for source: {source}"}

        combined = "\n\n".join(d.page_content for d in doc_chunks)

        chain = SUMMARISE_PROMPT | build_llm() | StrOutputParser()
        summary = chain.invoke({"document": combined})

        return {"source": source, "summary": summary, "chunks_used": len(doc_chunks)}

    # ── 5. Extract keywords ───────────────────────────────────────────────────

    @mcp.tool(
        description="Extract key terms and phrases from a text passage.",
        annotations={"readOnly": True},
    )
    def extract_keywords(
        text: Annotated[str, "Text to extract keywords from"],
    ) -> dict:
        """Use an LLM to extract the most important keywords from text."""
        from langchain_core.output_parsers import StrOutputParser

        from src.llm.base import build_llm
        from src.llm.prompt_templates import KEYWORD_PROMPT

        chain = KEYWORD_PROMPT | build_llm() | StrOutputParser()
        raw = chain.invoke({"text": text})
        keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]

        return {"keywords": keywords, "count": len(keywords)}

    # ── 6. Generate questions ─────────────────────────────────────────────────

    @mcp.tool(
        description="Generate evaluation questions from a text passage.",
        annotations={"readOnly": True},
    )
    def generate_questions(
        passage: Annotated[str, "Passage to generate questions for"],
        n: Annotated[int, "Number of questions to generate"] = 5,
    ) -> dict:
        """Generate n questions that can be answered from the given passage."""
        from langchain_core.output_parsers import StrOutputParser

        from src.llm.base import build_llm
        from src.llm.prompt_templates import QUESTION_GEN_PROMPT

        chain = QUESTION_GEN_PROMPT | build_llm() | StrOutputParser()
        raw = chain.invoke({"passage": passage, "n": n})
        questions = [q.strip() for q in raw.strip().split("\n") if q.strip()]

        return {"questions": questions, "count": len(questions)}

    # ── 7. Collection stats ───────────────────────────────────────────────────

    @mcp.tool(
        description="Get statistics about the vector store collection.",
        annotations={"readOnly": True},
    )
    def get_collection_stats(
        collection: Annotated[str, "Collection name (leave blank for default)"] = "",
    ) -> dict:
        """Return document count and metadata for the vector store collection."""
        from src.rag.vector_store import get_collection_stats

        return get_collection_stats(collection_name=collection or None)

    # ── 8. Delete document ────────────────────────────────────────────────────

    @mcp.tool(
        description="Delete documents from the vector store by their IDs.",
        annotations={"readOnly": False, "destructive": True},
    )
    def delete_documents(
        ids: Annotated[list[str], "List of vector store document IDs to delete"],
        collection: Annotated[str, "Collection name"] = "",
    ) -> dict:
        """Permanently delete documents from the vector store."""
        from src.rag.vector_store import delete_documents as _delete

        _delete(ids=ids, collection_name=collection or None)
        return {"deleted": len(ids), "ids": ids, "status": "success"}

    logger.info("Registered %d MCP tools", 8)
