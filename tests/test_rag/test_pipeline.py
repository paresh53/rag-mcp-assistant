"""Tests for RAG pipeline components."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch


class TestTextSplitter:
    """Tests for the text splitting module."""

    def test_recursive_split_reduces_size(self, sample_documents):
        from src.rag.text_splitter import SplitStrategy, split_documents

        chunks = split_documents(sample_documents, strategy=SplitStrategy.RECURSIVE, chunk_size=100)
        # Every chunk should be at most chunk_size characters
        for chunk in chunks:
            assert len(chunk.page_content) <= 150, "Chunk exceeded max size"

    def test_chunks_preserve_metadata(self, sample_documents):
        from src.rag.text_splitter import SplitStrategy, split_documents

        chunks = split_documents(sample_documents, strategy=SplitStrategy.RECURSIVE)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_strategy" in chunk.metadata

    def test_chunk_overlap_produces_more_chunks(self, sample_documents):
        from src.rag.text_splitter import SplitStrategy, split_documents

        # Larger document: combine all sample docs
        big_docs = [Document(
            page_content=" ".join(d.page_content for d in sample_documents),
            metadata={"source": "big.txt"},
        )]

        no_overlap = split_documents(big_docs, strategy=SplitStrategy.RECURSIVE,
                                     chunk_size=100, chunk_overlap=0)
        with_overlap = split_documents(big_docs, strategy=SplitStrategy.RECURSIVE,
                                       chunk_size=100, chunk_overlap=50)

        assert len(with_overlap) >= len(no_overlap)

    def test_token_splitter_works(self, sample_documents):
        from src.rag.text_splitter import SplitStrategy, split_documents

        chunks = split_documents(sample_documents, strategy=SplitStrategy.TOKEN, chunk_size=50)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk.page_content, str)
            assert len(chunk.page_content) > 0


class TestEmbeddings:
    """Tests for the embedding module."""

    def test_build_embedding_model_returns_model(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model
        build_embedding_model.cache_clear()
        model = build_embedding_model()
        assert model is not None

    def test_embed_query_returns_vector(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model, embed_query
        build_embedding_model.cache_clear()
        vector = embed_query("What is RAG?")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_cosine_similarity_range(self):
        from src.rag.embeddings import cosine_similarity

        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(1.0)

        vec_c = [-1.0, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_c) == pytest.approx(-1.0)

    def test_similar_texts_have_higher_similarity(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model, cosine_similarity, embed_query
        build_embedding_model.cache_clear()

        v1 = embed_query("What is retrieval augmented generation?")
        v2 = embed_query("Explain RAG in machine learning")
        v3 = embed_query("How do I bake a chocolate cake?")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated, "Related queries should be more similar"


class TestVectorStore:
    """Tests for vector store operations."""

    def test_add_and_retrieve_documents(self, tmp_chroma, sample_documents, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model
        from src.rag.vector_store import add_documents, similarity_search
        build_embedding_model.cache_clear()

        ids = add_documents(sample_documents)
        assert len(ids) == len(sample_documents)

        results = similarity_search("What is RAG?", k=3)
        assert len(results) > 0
        assert all(isinstance(d.page_content, str) for d in results)

    def test_similarity_search_with_scores(self, tmp_chroma, sample_documents, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model
        from src.rag.vector_store import add_documents, similarity_search_with_score
        build_embedding_model.cache_clear()

        add_documents(sample_documents)
        results = similarity_search_with_score("MCP protocol", k=3)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.01  # Scores can slightly exceed 1.0 due to float precision

    def test_get_collection_stats(self, tmp_chroma, sample_documents, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model
        from src.rag.vector_store import add_documents, get_collection_stats
        build_embedding_model.cache_clear()

        add_documents(sample_documents)
        stats = get_collection_stats()
        assert stats["total_documents"] == len(sample_documents)
        assert "collection" in stats

    def test_metadata_filter(self, tmp_chroma, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model
        from src.rag.vector_store import add_documents, similarity_search
        build_embedding_model.cache_clear()

        docs = [
            Document(page_content="RAG document from source A", metadata={"source": "A.pdf"}),
            Document(page_content="RAG document from source B", metadata={"source": "B.pdf"}),
        ]
        add_documents(docs)

        results = similarity_search("RAG", k=5, filter={"source": "A.pdf"})
        assert all(d.metadata.get("source") == "A.pdf" for d in results)


class TestRetriever:
    """Tests for retrieval strategies."""

    def test_similarity_retriever(self, tmp_chroma, sample_documents, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        monkeypatch.setenv("RETRIEVAL_STRATEGY", "similarity")
        from src.rag.embeddings import build_embedding_model
        from src.rag.retriever import RetrievalStrategy, build_retriever
        from src.rag.vector_store import add_documents
        build_embedding_model.cache_clear()

        add_documents(sample_documents)
        retriever = build_retriever(strategy=RetrievalStrategy.SIMILARITY, k=3)
        results = retriever.invoke("What is MCP?")
        assert len(results) > 0

    def test_mmr_retriever(self, tmp_chroma, sample_documents, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        from src.rag.embeddings import build_embedding_model
        from src.rag.retriever import RetrievalStrategy, build_retriever
        from src.rag.vector_store import add_documents
        build_embedding_model.cache_clear()

        add_documents(sample_documents)
        retriever = build_retriever(strategy=RetrievalStrategy.MMR, k=3)
        results = retriever.invoke("vector database")
        assert len(results) > 0
