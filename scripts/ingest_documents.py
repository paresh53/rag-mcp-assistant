"""
Script: Ingest Documents
========================
Ingest one or more documents into the vector store.

Usage examples:
    # Ingest a single file
    python scripts/ingest_documents.py data/documents/rag_and_mcp_concepts.md

    # Ingest an entire directory
    python scripts/ingest_documents.py data/documents/

    # Ingest a web page
    python scripts/ingest_documents.py https://en.wikipedia.org/wiki/Retrieval-augmented_generation

    # With options
    python scripts/ingest_documents.py data/documents/ --chunk-size 256 --overlap 32 --strategy token
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def ingest(
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    strategy: str = "recursive",
    collection: str | None = None,
) -> None:
    from src.rag.document_loader import load_document
    from src.rag.text_splitter import SplitStrategy, split_documents
    from src.rag.vector_store import add_documents

    start = time.perf_counter()

    logger.info("Loading: %s", source)
    raw_docs = load_document(source)
    logger.info("Loaded %d raw document(s)", len(raw_docs))

    chunks = split_documents(
        raw_docs,
        strategy=SplitStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Generated %d chunks", len(chunks))

    ids = add_documents(chunks, collection_name=collection)
    elapsed = time.perf_counter() - start

    logger.info(
        "✓ Ingested %d chunks in %.2fs (collection=%s)",
        len(ids),
        elapsed,
        collection or "default",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("source", help="File path, directory, or URL to ingest")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=64, help="Chunk overlap in characters")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "token", "semantic"],
        default="recursive",
        help="Text splitting strategy",
    )
    parser.add_argument("--collection", default=None, help="Target collection name")

    args = parser.parse_args()

    ingest(
        source=args.source,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        strategy=args.strategy,
        collection=args.collection,
    )


if __name__ == "__main__":
    main()
