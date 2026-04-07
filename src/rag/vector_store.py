"""
RAG Concept #4 — Vector Store
==============================
ChromaDB is used as the persistent vector store.
Demonstrates:
  • Adding documents
  • Persisting to disk
  • Listing / deleting collections
  • Direct similarity search with scores
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.rag.embeddings import build_embedding_model

logger = logging.getLogger(__name__)


def _get_chroma(collection_name: str | None = None) -> Chroma:
    """Return a Chroma instance bound to the configured persist directory."""
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
        embedding_function=build_embedding_model(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # Use cosine distance
    )


# ── Write operations ──────────────────────────────────────────────────────────

def add_documents(
    documents: list[Document],
    collection_name: str | None = None,
    batch_size: int = 100,
) -> list[str]:
    """
    Add chunked documents to the vector store.
    Returns the list of generated IDs.
    """
    db = _get_chroma(collection_name)
    all_ids: list[str] = []

    # Process in batches to avoid memory issues with large corpora
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        ids = db.add_documents(batch)
        all_ids.extend(ids)
        logger.info(
            "Indexed batch %d/%d (%d docs)",
            i // batch_size + 1,
            -(-len(documents) // batch_size),
            len(batch),
        )

    logger.info("Total indexed: %d documents → %d chunks", len(documents), len(all_ids))
    return all_ids


def delete_collection(collection_name: str | None = None) -> None:
    """Drop an entire collection from ChromaDB."""
    db = _get_chroma(collection_name)
    db.delete_collection()
    logger.warning("Deleted collection: %s", collection_name or settings.CHROMA_COLLECTION_NAME)


def delete_documents(ids: list[str], collection_name: str | None = None) -> None:
    """Delete specific documents by their vector store IDs."""
    db = _get_chroma(collection_name)
    db.delete(ids=ids)
    logger.info("Deleted %d documents", len(ids))


# ── Read operations ───────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    k: int = 5,
    collection_name: str | None = None,
    filter: dict[str, Any] | None = None,
) -> list[Document]:
    """
    Standard k-NN similarity search.

    Args:
        query:           The search query text.
        k:               Number of top results to return.
        collection_name: Target Chroma collection.
        filter:          Optional metadata filter, e.g. {"source": "report.pdf"}.

    Returns:
        Top-k Documents ranked by cosine similarity.
    """
    db = _get_chroma(collection_name)
    results = db.similarity_search(query, k=k, filter=filter)
    logger.debug("similarity_search(%r) → %d results", query, len(results))
    return results


def similarity_search_with_score(
    query: str,
    k: int = 5,
    collection_name: str | None = None,
) -> list[tuple[Document, float]]:
    """
    Same as similarity_search but also returns relevance scores (0‒1).
    Higher = more relevant.
    """
    db = _get_chroma(collection_name)
    return db.similarity_search_with_relevance_scores(query, k=k)


def get_collection_stats(collection_name: str | None = None) -> dict[str, Any]:
    """Return basic statistics about the vector store collection."""
    db = _get_chroma(collection_name)
    count = db._collection.count()
    return {
        "collection": collection_name or settings.CHROMA_COLLECTION_NAME,
        "total_documents": count,
        "persist_directory": settings.CHROMA_PERSIST_DIR,
    }


def as_langchain_retriever(
    collection_name: str | None = None,
    search_type: str = "similarity",
    search_kwargs: dict[str, Any] | None = None,
):
    """
    Expose the vector store as a LangChain BaseRetriever.
    Used by the pipeline — downstream chains call `.invoke(query)`.
    """
    db = _get_chroma(collection_name)
    return db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs or {"k": settings.RETRIEVAL_K},
    )
