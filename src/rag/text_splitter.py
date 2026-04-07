"""
RAG Concept #2 — Text Splitting / Chunking
===========================================
Three strategies are shown:
  1. RecursiveCharacterTextSplitter — default, hierarchy-aware
  2. TokenTextSplitter              — split by token count (respects model context)
  3. SemanticChunker                — embedding-based breakpoints (LangChain experimental)

Choosing the right strategy is critical to retrieval quality.
"""
from __future__ import annotations

import logging
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SplitStrategy(str, Enum):
    RECURSIVE = "recursive"   # Best general-purpose choice
    TOKEN = "token"           # Respects token limits of the LLM
    SEMANTIC = "semantic"     # Embedding-aware — highest quality, slowest


def split_documents(
    documents: list[Document],
    strategy: SplitStrategy = SplitStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents:     Input documents (from document_loader).
        strategy:      Chunking strategy to use.
        chunk_size:    Target size of each chunk (chars or tokens).
        chunk_overlap: Number of chars/tokens to overlap between chunks.

    Returns:
        List of chunked Document objects, each preserving source metadata.
    """
    splitter = _build_splitter(strategy, chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Enrich each chunk with index metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata.setdefault("chunk_index", i)
        chunk.metadata.setdefault("chunk_strategy", strategy.value)

    logger.info(
        "Split %d docs → %d chunks (strategy=%s, size=%d, overlap=%d)",
        len(documents), len(chunks), strategy.value, chunk_size, chunk_overlap,
    )
    return chunks


def _build_splitter(strategy: SplitStrategy, chunk_size: int, chunk_overlap: int):
    if strategy == SplitStrategy.RECURSIVE:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Hierarchy: paragraph → sentence → word → char
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    if strategy == SplitStrategy.TOKEN:
        from langchain_text_splitters import TokenTextSplitter

        # Uses tiktoken (cl100k_base — compatible with gpt-4, gpt-3.5)
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )

    if strategy == SplitStrategy.SEMANTIC:
        # SemanticChunker uses embedding cosine similarity to find breakpoints.
        # Requires an embedding model — uses the one configured in settings.
        from langchain_experimental.text_splitter import SemanticChunker

        from src.rag.embeddings import build_embedding_model

        embeddings = build_embedding_model()
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # Options: percentile | standard_deviation | interquartile
            breakpoint_threshold_amount=95,
        )

    raise ValueError(f"Unknown split strategy: {strategy}")
