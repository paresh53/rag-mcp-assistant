"""Shared pytest fixtures and configuration."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving "
                "external knowledge and injecting it into the prompt context."
            ),
            metadata={"source": "test_doc.txt", "page": 1},
        ),
        Document(
            page_content=(
                "The Model Context Protocol (MCP) is an open standard for connecting "
                "AI assistants to external data sources and tools."
            ),
            metadata={"source": "test_doc.txt", "page": 2},
        ),
        Document(
            page_content=(
                "ChromaDB is an open-source vector database that stores document "
                "embeddings and supports fast similarity search using HNSW."
            ),
            metadata={"source": "test_doc.txt", "page": 3},
        ),
        Document(
            page_content=(
                "Maximum Marginal Relevance (MMR) balances relevance and diversity "
                "by penalising retrieved documents that are too similar to each other."
            ),
            metadata={"source": "test_doc.txt", "page": 4},
        ),
        Document(
            page_content=(
                "Cross-encoder rerankers jointly encode query-document pairs and "
                "produce a relevance score more accurate than bi-encoder cosine similarity."
            ),
            metadata={"source": "test_doc.txt", "page": 5},
        ),
    ]


@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    """Override ChromaDB persist directory to a temporary path for test isolation."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
    # Clear lru_cache so settings reload from environment
    from src.rag import embeddings
    embeddings.build_embedding_model.cache_clear()
    yield tmp_path / "chroma"
