"""
RAG Concept #8 — Contextual Compression
=========================================
Contextual compression filters / summarises retrieved chunks so only the
most relevant *parts* of each document are passed to the LLM, reducing
prompt length and noise.

Two compressors demonstrated:
  1. LLMChainExtractor    — asks LLM to extract only the relevant sentences
  2. EmbeddingsFilter     — drops chunks whose embedding is far from the query
  3. LLMChainFilter       — asks LLM to approve/reject each chunk (binary)
  4. DocumentCompressorPipeline — chains multiple compressors sequentially
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def build_llm_extractor_retriever(
    base_retriever: BaseRetriever,
    llm: Any | None = None,
) -> BaseRetriever:
    """
    LLMChainExtractor compressor.
    For each retrieved chunk, asks the LLM to extract only the sentences
    that are directly relevant to the query. Returns a ContextualCompressionRetriever.
    """
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor

    from src.llm.base import build_llm

    llm = llm or build_llm()
    compressor = LLMChainExtractor.from_llm(llm)

    logger.info("LLMChainExtractor compression retriever configured")
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_embedding_filter_retriever(
    base_retriever: BaseRetriever,
    similarity_threshold: float = 0.76,
) -> BaseRetriever:
    """
    EmbeddingsFilter compressor.
    Drops any chunk whose embedding is below `similarity_threshold`
    similarity to the query — purely local, no extra LLM calls.
    Fast and cheap but less nuanced than LLM-based compression.
    """
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

    from src.rag.embeddings import build_embedding_model

    embeddings = build_embedding_model()
    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold,
    )

    logger.info("EmbeddingsFilter compression retriever (threshold=%.2f)", similarity_threshold)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_pipeline_compressor_retriever(
    base_retriever: BaseRetriever,
    llm: Any | None = None,
) -> BaseRetriever:
    """
    DocumentCompressorPipeline — chains multiple compressors:
      Step 1: EmbeddingsFilter (cheap, removes obviously irrelevant chunks)
      Step 2: LLMChainFilter   (LLM binary yes/no — more expensive but accurate)

    Running cheap filters first reduces the number of LLM calls needed.
    """
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import (
        DocumentCompressorPipeline,
        EmbeddingsFilter,
        LLMChainFilter,
    )

    from src.llm.base import build_llm
    from src.rag.embeddings import build_embedding_model

    llm = llm or build_llm()
    embeddings = build_embedding_model()

    pipeline = DocumentCompressorPipeline(
        transformers=[
            EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.65),
            LLMChainFilter.from_llm(llm),
        ]
    )

    logger.info("Pipeline compressor retriever configured (EmbeddingsFilter → LLMChainFilter)")
    return ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=base_retriever,
    )
