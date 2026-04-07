"""
RAG Concept #9 — Full RAG Pipeline Orchestrator
=================================================
Ties together all RAG components using LangChain Expression Language (LCEL):

  Document → Chunk → Embed → Store
       ↓
  Query → [Expand?] → Retrieve → [Rerank?] → [Compress?] → Generate

Also demonstrates:
  • Streaming responses
  • Source attribution
  • Conversation memory (multi-turn RAG)
  • Asynchronous execution
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.rag.retriever import RetrievalStrategy, build_retriever

logger = logging.getLogger(__name__)


# ── Response model ────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    retrieval_strategy: str = ""
    reranked: bool = False
    compressed: bool = False


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline with configurable retrieval strategy,
    optional reranking, and optional contextual compression.
    """

    def __init__(
        self,
        retrieval_strategy: RetrievalStrategy | str | None = None,
        k: int | None = None,
        enable_rerank: bool | None = None,
        enable_compression: bool | None = None,
        enable_query_expansion: bool | None = None,
        collection_name: str | None = None,
        llm: Any | None = None,
    ) -> None:
        from src.llm.base import build_llm
        from src.llm.prompt_templates import RAG_PROMPT

        self.strategy = RetrievalStrategy(retrieval_strategy or settings.RETRIEVAL_STRATEGY)
        self.k = k or settings.RETRIEVAL_K
        self.enable_rerank = enable_rerank if enable_rerank is not None else settings.RERANK_ENABLED
        self.enable_compression = enable_compression if enable_compression is not None else False
        self.enable_query_expansion = (
            enable_query_expansion
            if enable_query_expansion is not None
            else settings.QUERY_EXPANSION_ENABLED
        )
        self.collection_name = collection_name
        self.llm = llm or build_llm()
        self.prompt = RAG_PROMPT

        # Build retriever
        self.retriever = build_retriever(
            strategy=self.strategy,
            k=self.k,
            collection_name=self.collection_name,
        )

        # Optionally wrap with multi-query
        if self.enable_query_expansion:
            from src.rag.query_expansion import build_multi_query_retriever
            self.retriever = build_multi_query_retriever(
                base_retriever=self.retriever,
                llm=self.llm,
            )

        # Optionally wrap with contextual compression
        if self.enable_compression:
            from src.rag.context_compression import build_embedding_filter_retriever
            self.retriever = build_embedding_filter_retriever(self.retriever)

    # ── Sync query ────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Run a full RAG query synchronously.
        Returns a RAGResponse with answer and source metadata.
        """
        docs = self.retriever.invoke(question)

        if self.enable_rerank and docs:
            from src.rag.reranker import rerank
            docs = rerank(question, docs, top_n=settings.RERANK_TOP_N)

        context = _format_context(docs)
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})

        return RAGResponse(
            answer=answer,
            sources=_extract_sources(docs),
            query=question,
            retrieval_strategy=self.strategy.value,
            reranked=self.enable_rerank,
            compressed=self.enable_compression,
        )

    # ── Async query ───────────────────────────────────────────────────────────

    async def aquery(self, question: str) -> RAGResponse:
        """Async version of query()."""
        docs = await self.retriever.ainvoke(question)

        if self.enable_rerank and docs:
            from src.rag.reranker import rerank
            docs = await asyncio.get_event_loop().run_in_executor(
                None, rerank, question, docs, settings.RERANK_TOP_N
            )

        context = _format_context(docs)
        chain = self.prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({"question": question, "context": context})

        return RAGResponse(
            answer=answer,
            sources=_extract_sources(docs),
            query=question,
            retrieval_strategy=self.strategy.value,
            reranked=self.enable_rerank,
            compressed=self.enable_compression,
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(self, question: str) -> Iterator[str]:
        """
        Stream the LLM answer token-by-token.
        Sources are appended as a final JSON block.
        """
        docs = self.retriever.invoke(question)

        if self.enable_rerank and docs:
            from src.rag.reranker import rerank
            docs = rerank(question, docs, top_n=settings.RERANK_TOP_N)

        context = _format_context(docs)
        chain = self.prompt | self.llm | StrOutputParser()

        for chunk in chain.stream({"question": question, "context": context}):
            yield chunk

    async def astream(self, question: str) -> AsyncIterator[str]:
        """Async streaming version."""
        docs = await self.retriever.ainvoke(question)

        if self.enable_rerank and docs:
            from src.rag.reranker import rerank
            docs = rerank(question, docs)

        context = _format_context(docs)
        chain = self.prompt | self.llm | StrOutputParser()

        async for chunk in chain.astream({"question": question, "context": context}):
            yield chunk

    # ── Conversational RAG ────────────────────────────────────────────────────

    def query_with_history(
        self,
        question: str,
        chat_history: list[tuple[str, str]],
    ) -> RAGResponse:
        """
        Multi-turn conversational RAG.
        chat_history: list of (human_message, ai_message) tuples.

        Uses a condense-question chain to rewrite the question given history,
        then runs the standard RAG pipeline.
        """
        from langchain.chains import ConversationalRetrievalChain

        from src.llm.prompt_templates import CONDENSE_QUESTION_PROMPT

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            return_source_documents=True,
        )
        result = chain.invoke({"question": question, "chat_history": chat_history})

        return RAGResponse(
            answer=result["answer"],
            sources=_extract_sources(result.get("source_documents", [])),
            query=question,
            retrieval_strategy=self.strategy.value,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a numbered context block for the prompt."""
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        ref = f"{source}" + (f" (p.{page})" if page != "" else "")
        parts.append(f"[{i}] (Source: {ref})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Extract deduplicated source metadata from retrieved documents."""
    seen: set[str] = set()
    sources = []
    for doc in docs:
        key = doc.metadata.get("source", "") + str(doc.metadata.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", ""),
                "score": doc.metadata.get("rerank_score", doc.metadata.get("score", None)),
                "snippet": doc.page_content[:200],
            })
    return sources
