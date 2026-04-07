"""
MCP Concept — Resources
========================
MCP Resources are read-only data sources accessible via URI.
The client (LLM host) can READ resources but cannot call them like tools.

URI schemes used here:
  • documents://list          — list all indexed document sources
  • documents://{source}      — fetch raw content for a specific source
  • collection://stats        — aggregate collection statistics
  • collection://{name}/docs  — list docs in a named collection

Resources support:
  • Static resources (fixed URI)
  • Dynamic resource templates (URI with {params})
  • Resource subscriptions (notify client when data changes)
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def register_resources(mcp):
    """Register all resources on the given FastMCP instance."""

    # ── Static resource: list all documents ──────────────────────────────────

    @mcp.resource(
        uri="documents://list",
        name="Document List",
        description="List all documents currently indexed in the vector store.",
        mime_type="application/json",
    )
    def list_documents() -> str:
        """Return a JSON array of all indexed document sources."""
        from src.config import settings
        from src.rag.vector_store import _get_chroma

        db = _get_chroma()
        # Query all documents (no filter) — only fetch metadata
        result = db.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        sources: dict[str, dict] = {}
        for meta in metadatas:
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "total_chunks": 0,
                    "page_count": set(),
                }
            sources[src]["total_chunks"] += 1
            if "page" in meta:
                sources[src]["page_count"].add(meta["page"])

        output = []
        for src_data in sources.values():
            output.append({
                "source": src_data["source"],
                "total_chunks": src_data["total_chunks"],
                "pages": len(src_data["page_count"]) or None,
            })

        return json.dumps(output, indent=2)

    # ── Dynamic resource template: fetch a document's chunks ─────────────────

    @mcp.resource(
        uri="documents://{source}",
        name="Document Content",
        description="Retrieve all stored chunks for a specific document source.",
        mime_type="application/json",
    )
    def get_document_content(source: str) -> str:
        """Return all chunks stored for the given source identifier."""
        from src.rag.vector_store import _get_chroma

        db = _get_chroma()
        result = db.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        chunks = []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            chunks.append({
                "chunk_index": i,
                "content": doc,
                "metadata": meta,
            })

        return json.dumps(
            {"source": source, "total_chunks": len(chunks), "chunks": chunks},
            indent=2,
        )

    # ── Static resource: collection statistics ────────────────────────────────

    @mcp.resource(
        uri="collection://stats",
        name="Collection Statistics",
        description="Aggregate statistics about the default vector store collection.",
        mime_type="application/json",
    )
    def collection_stats() -> str:
        """Return document count and storage info for the default collection."""
        from src.rag.vector_store import get_collection_stats

        stats = get_collection_stats()
        return json.dumps(stats, indent=2)

    # ── Dynamic resource: named collection stats ──────────────────────────────

    @mcp.resource(
        uri="collection://{name}/stats",
        name="Named Collection Stats",
        description="Statistics for a specific named collection.",
        mime_type="application/json",
    )
    def named_collection_stats(name: str) -> str:
        """Return stats for a specific named collection."""
        from src.rag.vector_store import get_collection_stats

        stats = get_collection_stats(collection_name=name)
        return json.dumps(stats, indent=2)

    # ── Static resource: server capabilities ─────────────────────────────────

    @mcp.resource(
        uri="server://capabilities",
        name="Server Capabilities",
        description="Describes the capabilities and configuration of this MCP server.",
        mime_type="application/json",
    )
    def server_capabilities() -> str:
        """Describe what this MCP server can do."""
        from src.config import settings

        info = {
            "name": settings.MCP_SERVER_NAME,
            "version": settings.MCP_SERVER_VERSION,
            "capabilities": {
                "tools": ["search_documents", "ask_question", "ingest_document",
                           "summarise_document", "extract_keywords", "generate_questions",
                           "get_collection_stats", "delete_documents"],
                "resources": ["documents://list", "documents://{source}",
                              "collection://stats", "collection://{name}/stats"],
                "prompts": ["rag_answer", "document_analysis", "qa_evaluation"],
            },
            "configuration": {
                "llm_provider": settings.LLM_PROVIDER,
                "llm_model": settings.LLM_MODEL,
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.EMBEDDING_MODEL,
                "retrieval_strategy": settings.RETRIEVAL_STRATEGY,
                "rerank_enabled": settings.RERANK_ENABLED,
            },
        }
        return json.dumps(info, indent=2)

    logger.info("Registered %d MCP resources", 5)
