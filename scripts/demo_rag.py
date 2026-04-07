"""
Script: Quick RAG Demo
=======================
Demonstrates a full RAG query end-to-end without starting the API server.

Usage:
    python scripts/demo_rag.py
    python scripts/demo_rag.py --question "What is MCP?" --strategy mmr
    python scripts/demo_rag.py --ingest  # Also ingest sample docs first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def demo(question: str, strategy: str, also_ingest: bool) -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)  # Suppress noisy logs for demo

    if also_ingest:
        print("Ingesting sample documents...")
        from scripts.ingest_documents import ingest
        sample = str(Path(__file__).parent.parent / "data/documents")
        ingest(sample, chunk_size=512, chunk_overlap=64)
        print()

    print(f"Question: {question}")
    print(f"Strategy: {strategy}")
    print("─" * 60)

    from src.rag.pipeline import RAGPipeline
    from src.rag.retriever import RetrievalStrategy

    pipeline = RAGPipeline(
        retrieval_strategy=RetrievalStrategy(strategy),
        enable_rerank=True,
        enable_query_expansion=True,
    )

    print("Generating answer...\n")
    response = pipeline.query(question)

    print(f"Answer:\n{response.answer}")
    print("\nSources:")
    for i, src in enumerate(response.sources, 1):
        print(f"  [{i}] {src['source']} — {src['snippet'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="RAG end-to-end demo")
    parser.add_argument(
        "--question",
        default="What are the main components of a RAG pipeline?",
        help="Question to ask",
    )
    parser.add_argument("--strategy", default="mmr", choices=["similarity", "mmr", "hybrid"])
    parser.add_argument("--ingest", action="store_true", help="Ingest sample docs before querying")
    args = parser.parse_args()
    demo(args.question, args.strategy, args.ingest)


if __name__ == "__main__":
    main()
