"""
RAG Concept #5 — Retrieval Strategies
=======================================
Three strategies demonstrated:

  1. Similarity (k-NN)     — vanilla cosine/dot-product search
  2. MMR                   — Maximum Marginal Relevance (diversity-aware)
  3. Hybrid (BM25 + dense) — combines lexical and semantic signals via
                             Reciprocal Rank Fusion (RRF)

All strategies return a LangChain BaseRetriever so they can be plugged
directly into any LCEL chain.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import settings

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    SIMILARITY = "similarity"  # Standard k-NN
    MMR = "mmr"                # Max Marginal Relevance
    HYBRID = "hybrid"          # BM25 + dense (Ensemble)


# ── Public factory ────────────────────────────────────────────────────────────

def build_retriever(
    strategy: RetrievalStrategy | str | None = None,
    k: int | None = None,
    collection_name: str | None = None,
    documents: list[Document] | None = None,
) -> BaseRetriever:
    """
    Build and return the appropriate retriever.

    Args:
        strategy:        Which retrieval strategy to use.
        k:               Number of documents to return.
        collection_name: Chroma collection name.
        documents:       Required for HYBRID (used to build BM25 index).
    """
    strategy = RetrievalStrategy(strategy or settings.RETRIEVAL_STRATEGY)
    k = k or settings.RETRIEVAL_K

    if strategy == RetrievalStrategy.SIMILARITY:
        return _build_similarity_retriever(k, collection_name)

    if strategy == RetrievalStrategy.MMR:
        return _build_mmr_retriever(k, collection_name)

    if strategy == RetrievalStrategy.HYBRID:
        if documents is None:
            raise ValueError("HYBRID strategy requires `documents` to build BM25 index.")
        return _build_hybrid_retriever(k, collection_name, documents)

    raise ValueError(f"Unknown retrieval strategy: {strategy}")


# ── Strategy implementations ──────────────────────────────────────────────────

def _build_similarity_retriever(k: int, collection_name: str | None) -> BaseRetriever:
    """
    Standard cosine similarity k-NN.
    Fast and accurate for homogeneous corpora.
    """
    from src.rag.vector_store import as_langchain_retriever

    return as_langchain_retriever(
        collection_name=collection_name,
        search_type="similarity",
        search_kwargs={"k": k},
    )


def _build_mmr_retriever(k: int, collection_name: str | None) -> BaseRetriever:
    """
    Maximum Marginal Relevance (MMR).
    Balances relevance with diversity — avoids returning near-duplicate chunks.

    fetch_k:      How many candidates to fetch from the vector store first.
    lambda_mult:  0 = max diversity, 1 = max relevance (default 0.5).
    """
    from src.rag.vector_store import as_langchain_retriever

    return as_langchain_retriever(
        collection_name=collection_name,
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 4,   # Fetch more, then re-rank for diversity
            "lambda_mult": 0.5,
        },
    )


def _build_hybrid_retriever(
    k: int,
    collection_name: str | None,
    documents: list[Document],
) -> BaseRetriever:
    """
    Hybrid Retrieval = BM25 (lexical) + Dense (semantic) combined via
    Reciprocal Rank Fusion (RRF).

    BM25 excels at exact keyword matching.
    Dense embeddings excel at semantic / paraphrase matching.
    Together they cover both failure modes.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

    from src.rag.vector_store import as_langchain_retriever

    # BM25 needs the raw documents list (not the vector store)
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    dense_retriever = as_langchain_retriever(
        collection_name=collection_name,
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # weights=[bm25_weight, dense_weight] — must sum to 1
    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],  # Give slightly more weight to dense retrieval
    )


# ── Metadata filtering ────────────────────────────────────────────────────────

def retrieve_with_filter(
    query: str,
    metadata_filter: dict[str, Any],
    k: int | None = None,
    collection_name: str | None = None,
) -> list[Document]:
    """
    Retrieve documents filtered by metadata key-value pairs.
    Useful for multi-tenant or multi-source setups.
    Example: retrieve_with_filter("revenue", {"source": "annual_report.pdf"})
    """
    from src.rag.vector_store import similarity_search

    return similarity_search(
        query=query,
        k=k or settings.RETRIEVAL_K,
        collection_name=collection_name,
        filter=metadata_filter,
    )
