"""
RAG Concept #6 — Query Expansion
==================================
Two techniques that rewrite / augment the user query before retrieval,
improving recall when the query is ambiguous or short:

  1. Multi-Query   — generate N paraphrased queries, union the results
  2. HyDE          — Hypothetical Document Embeddings: generate a fake
                     "ideal answer" and embed that instead of the query
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


# ── Multi-Query Retriever ─────────────────────────────────────────────────────

def build_multi_query_retriever(
    base_retriever: BaseRetriever,
    llm: Any | None = None,
    num_queries: int = 3,
) -> BaseRetriever:
    """
    Multi-Query Expansion.

    Wraps `base_retriever` in a LangChain MultiQueryRetriever.
    For each user query the LLM generates `num_queries` paraphrases,
    retrieves from each, and de-duplicates the union.

    Benefits:
     - Handles ambiguous or colloquial queries
     - Increases recall without hurting precision much

    Args:
        base_retriever: The underlying retriever (similarity / mmr / hybrid).
        llm:            LLM used to generate sub-queries. Defaults to configured LLM.
        num_queries:    How many paraphrases to generate.
    """
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    from src.llm.base import build_llm

    llm = llm or build_llm()

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )
    # Expose how many queries will be generated (informational)
    retriever._num_queries = num_queries  # type: ignore[attr-defined]
    logger.info("MultiQueryRetriever configured with %d sub-queries", num_queries)
    return retriever


# ── HyDE (Hypothetical Document Embeddings) ───────────────────────────────────

def build_hyde_retriever(
    base_retriever: BaseRetriever | None = None,
    llm: Any | None = None,
    collection_name: str | None = None,
) -> BaseRetriever:
    """
    HyDE — Hypothetical Document Embeddings.

    Instead of embedding the user query directly, HyDE asks the LLM to write
    a hypothetical document that would answer the query, then embeds *that*.

    This bridges the gap between short user queries and longer document chunks.

    Steps:
      1. LLM writes a fake answer passage.
      2. That passage is embedded.
      3. The embedding is used for vector search.
    """
    from langchain.chains import HypotheticalDocumentEmbedder
    from langchain_community.vectorstores import Chroma

    from src.config import settings
    from src.llm.base import build_llm
    from src.rag.embeddings import build_embedding_model

    llm = llm or build_llm()
    base_embeddings = build_embedding_model()

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=llm,
        base_embeddings=base_embeddings,
        custom_prompt=_HYDE_PROMPT,
    )

    # Build (or reuse) a Chroma store that uses the HyDE embeddings
    from pathlib import Path
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    hyde_store = Chroma(
        collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
        embedding_function=hyde_embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    logger.info("HyDE retriever configured")
    return hyde_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})


# ── Prompts ───────────────────────────────────────────────────────────────────

from langchain_core.prompts import PromptTemplate

_HYDE_PROMPT = PromptTemplate.from_template(
    "Write a short, factual passage that directly answers the following question.\n"
    "Do not include the question. Just provide the answer passage.\n\n"
    "Question: {question}\n\n"
    "Passage:"
)
