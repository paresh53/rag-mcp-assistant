"""RAG query API routes — demonstrates all retrieval and generation strategies."""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to answer")
    strategy: str = Field("mmr", description="Retrieval: similarity | mmr | hybrid")
    k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    enable_rerank: bool = Field(True, description="Apply cross-encoder reranking")
    enable_compression: bool = Field(False, description="Apply contextual compression")
    enable_query_expansion: bool = Field(True, description="Use multi-query expansion")
    collection: str = Field("", description="Vector store collection name")


class SourceModel(BaseModel):
    source: str
    page: str | int | None
    score: float | None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    query: str
    retrieval_strategy: str
    reranked: bool
    compressed: bool


class ChatMessage(BaseModel):
    role: str  # "human" | "ai"
    content: str


class ConversationalQueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    strategy: str = "mmr"
    collection: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Ask a question (RAG)")
async def rag_query(body: QueryRequest):
    """
    Full RAG pipeline:
    query-expansion → retrieval → reranking → contextual-compression → generation

    Choose your retrieval strategy: similarity, mmr, or hybrid.
    """
    from src.rag.pipeline import RAGPipeline
    from src.rag.retriever import RetrievalStrategy

    try:
        pipeline = RAGPipeline(
            retrieval_strategy=RetrievalStrategy(body.strategy),
            k=body.k,
            enable_rerank=body.enable_rerank,
            enable_compression=body.enable_compression,
            enable_query_expansion=body.enable_query_expansion,
            collection_name=body.collection or None,
        )
        response = await pipeline.aquery(body.question)
        return QueryResponse(
            answer=response.answer,
            sources=[SourceModel(**s) for s in response.sources],
            query=response.query,
            retrieval_strategy=response.retrieval_strategy,
            reranked=response.reranked,
            compressed=response.compressed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query/stream", summary="Streaming RAG answer")
async def rag_query_stream(body: QueryRequest):
    """
    Stream the RAG answer token-by-token via Server-Sent Events.
    The client receives chunks as 'data: <token>\\n\\n'.
    """
    from src.rag.pipeline import RAGPipeline
    from src.rag.retriever import RetrievalStrategy

    pipeline = RAGPipeline(
        retrieval_strategy=RetrievalStrategy(body.strategy),
        k=body.k,
        enable_rerank=body.enable_rerank,
        collection_name=body.collection or None,
    )

    async def token_stream() -> AsyncIterator[str]:
        async for token in pipeline.astream(body.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@router.post("/query/conversational", response_model=QueryResponse, summary="Multi-turn RAG")
async def conversational_rag_query(body: ConversationalQueryRequest):
    """
    Conversational (multi-turn) RAG.
    Pass previous chat history to get context-aware answers.
    """
    from src.rag.pipeline import RAGPipeline
    from src.rag.retriever import RetrievalStrategy

    try:
        history = [(m.content, "") if m.role == "human" else ("", m.content) for m in body.history]
        # Build proper (human, ai) tuples from alternating messages
        chat_history = []
        for i in range(0, len(body.history) - 1, 2):
            if body.history[i].role == "human" and body.history[i + 1].role == "ai":
                chat_history.append((body.history[i].content, body.history[i + 1].content))

        pipeline = RAGPipeline(
            retrieval_strategy=RetrievalStrategy(body.strategy),
            collection_name=body.collection or None,
        )
        response = pipeline.query_with_history(body.question, chat_history)
        return QueryResponse(
            answer=response.answer,
            sources=[SourceModel(**s) for s in response.sources],
            query=response.query,
            retrieval_strategy=response.retrieval_strategy,
            reranked=response.reranked,
            compressed=response.compressed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search", summary="Raw vector search (no generation)")
async def vector_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20),
    strategy: str = Query("similarity"),
    collection: str = Query(""),
    with_scores: bool = Query(False, description="Include relevance scores"),
):
    """
    Raw retrieval without LLM generation.
    Returns the top-k most relevant document chunks.
    """
    from src.rag.vector_store import similarity_search, similarity_search_with_score

    if with_scores:
        results = similarity_search_with_score(q, k=k, collection_name=collection or None)
        return {
            "query": q,
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in results
            ],
        }
    else:
        docs = similarity_search(q, k=k, collection_name=collection or None)
        return {
            "query": q,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
        }
