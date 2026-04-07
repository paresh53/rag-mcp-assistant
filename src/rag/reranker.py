"""
RAG Concept #7 — Re-ranking
=============================
After initial retrieval, a cross-encoder re-ranker scores each (query, chunk)
pair more precisely — at the cost of higher latency.

Two options are shown:
  1. FlashRank (local, ultra-fast MS-MARCO cross-encoder)
  2. Cohere Rerank API (API-based, highest quality)

Re-ranking is a powerful post-retrieval step that dramatically improves
precision without changing chunk count reaching the LLM.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── FlashRank (local, no API key needed) ─────────────────────────────────────

def rerank_flashrank(
    query: str,
    documents: list[Document],
    top_n: int = 3,
    model: str = "ms-marco-MiniLM-L-12-v2",
) -> list[Document]:
    """
    Re-rank documents using FlashRank (local cross-encoder).

    FlashRank is extremely fast because it uses quantised ONNX models.
    The model scores each (query, passage) pair and returns top_n docs
    ordered by cross-encoder score descending.

    Args:
        query:     The user's original query.
        documents: Retrieved candidate documents.
        top_n:     How many documents to keep after re-ranking.
        model:     FlashRank model name.
    """
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        logger.warning("flashrank not installed — skipping re-rank")
        return documents[:top_n]

    ranker = Ranker(model_name=model, cache_dir="/tmp/flashrank")

    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(documents)
    ]
    request = RerankRequest(query=query, passages=passages)
    result = ranker.rerank(request)

    # Map back to Document objects in ranked order
    id_to_doc = {i: doc for i, doc in enumerate(documents)}
    reranked = []
    for item in result[:top_n]:
        doc = id_to_doc[item["id"]]
        doc.metadata["rerank_score"] = item["score"]
        reranked.append(doc)

    logger.debug("FlashRank: %d → %d docs (query=%r)", len(documents), len(reranked), query)
    return reranked


# ── Cohere Rerank API (cloud, highest quality) ────────────────────────────────

def rerank_cohere(
    query: str,
    documents: list[Document],
    top_n: int = 3,
    model: str = "rerank-english-v3.0",
    api_key: str | None = None,
) -> list[Document]:
    """
    Re-rank using the Cohere Rerank API.

    Cohere's reranker is state-of-the-art and works across languages.
    Requires COHERE_API_KEY environment variable or explicit api_key.

    Args:
        query:     User query.
        documents: Candidate documents from retrieval.
        top_n:     Documents to keep.
        model:     Cohere model name.
        api_key:   Optional explicit API key.
    """
    import os

    try:
        import cohere
    except ImportError:
        logger.warning("cohere not installed — falling back to FlashRank")
        return rerank_flashrank(query, documents, top_n)

    key = api_key or os.getenv("COHERE_API_KEY", "")
    if not key:
        logger.warning("COHERE_API_KEY not set — falling back to FlashRank")
        return rerank_flashrank(query, documents, top_n)

    co = cohere.Client(key)
    texts = [doc.page_content for doc in documents]
    response = co.rerank(query=query, documents=texts, top_n=top_n, model=model)

    reranked = []
    for hit in response.results:
        doc = documents[hit.index]
        doc.metadata["rerank_score"] = hit.relevance_score
        reranked.append(doc)

    logger.debug("Cohere rerank: %d → %d docs", len(documents), len(reranked))
    return reranked


# ── Unified entry point ───────────────────────────────────────────────────────

def rerank(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
    provider: str = "flashrank",
) -> list[Document]:
    """
    Re-rank retrieved documents.

    Args:
        query:     User query.
        documents: Candidate documents.
        top_n:     Number to keep. Defaults to settings.RERANK_TOP_N.
        provider:  'flashrank' or 'cohere'.
    """
    from src.config import settings

    top_n = top_n or settings.RERANK_TOP_N

    if not documents:
        return []

    if provider == "cohere":
        return rerank_cohere(query, documents, top_n=top_n)
    return rerank_flashrank(query, documents, top_n=top_n)
