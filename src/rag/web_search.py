"""
Web search fallback using DuckDuckGo.
Used when the vector store returns no relevant documents for a query.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo and return raw result dicts.
    Each dict contains: title, href, body.
    Returns an empty list on any failure.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        logger.info("Web search for %r returned %d results", query, len(results))
        return results
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return []


def web_results_to_docs(results: list[dict]) -> list[Document]:
    """Convert DuckDuckGo result dicts to LangChain Documents."""
    docs = []
    for r in results:
        body = r.get("body", "").strip()
        if not body:
            continue
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "source": r.get("href", ""),
                    "title": r.get("title", ""),
                    "web_search": True,
                },
            )
        )
    return docs
