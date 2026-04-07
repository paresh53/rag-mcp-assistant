"""
RAG Concept #3 — Embeddings
============================
Embeddings convert text into dense vectors that capture semantic meaning.
Two providers are supported:
  • HuggingFace (local, free)  — default
  • OpenAI (API-based, higher quality)

The embedding model is chosen via the EMBEDDING_PROVIDER env variable.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


# ── Protocol (for type-checking without hard circular imports) ───────────────

class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# ── Builders ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)  # Singleton — loading a model is expensive
def build_embedding_model() -> Embeddings:
    """
    Return the configured embedding model instance.
    Result is cached so the model is loaded only once per process.
    """
    from src.config import settings

    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        return _build_openai_embeddings(settings)
    if provider == "huggingface":
        return _build_hf_embeddings(settings)

    raise ValueError(f"Unknown embedding provider: '{provider}'. Choose 'openai' or 'huggingface'.")


def _build_openai_embeddings(settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    logger.info("Using OpenAI embeddings: %s", settings.OPENAI_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        # Supports batching out of the box
    )


def _build_hf_embeddings(settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Using HuggingFace embeddings: %s", settings.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # Important for cosine similarity!
            "batch_size": 32,
        },
    )


# ── Utility helpers ───────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a, b = np.array(vec_a), np.array(vec_b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper: embed a batch of texts."""
    model = build_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Convenience wrapper: embed a single query string."""
    model = build_embedding_model()
    return model.embed_query(text)
