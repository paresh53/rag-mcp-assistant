"""
Web Crawler — recursively discovers and loads all pages under a base URL.

Given a top-level URL (e.g. https://www.ibm.com/docs/en/filenet-p8-platform/5.7.0)
it will:
  1. Fetch the page and extract all same-domain links that share the base path prefix
  2. Recursively visit those links (BFS, up to max_pages)
  3. Return all LangChain Documents ready for chunking + embedding
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable
from urllib.parse import urljoin, urlparse, urldefrag

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Tags whose text we want to keep; everything else is stripped
_CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th",
                 "pre", "code", "blockquote", "article", "section", "main"}


def _fetch_page(url: str, timeout: int = 15) -> tuple[str, str]:
    """Return (html_text, resolved_url). Raises on HTTP error."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; RAGCrawler/1.0; "
                "+https://github.com/paresh53/rag-mcp-assistant)"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resolved = resp.url
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return "", resolved
        html = resp.read().decode("utf-8", errors="replace")
    return html, resolved


def _extract_text_and_links(html: str, base_url: str) -> tuple[str, list[str]]:
    """Parse HTML → (cleaned text, list of absolute href links)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 is required: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")

    # Remove script / style / nav / footer noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "svg", "img"]):
        tag.decompose()

    # Extract meaningful text
    chunks: list[str] = []
    for tag in soup.find_all(_CONTENT_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 30:          # skip trivially short snippets
            chunks.append(text)
    text = "\n\n".join(chunks)

    # Extract links
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Strip fragment
        href, _ = urldefrag(href)
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        links.append(abs_url)

    return text, links


def _same_scope(url: str, base_parsed) -> bool:
    """True if url is on the same host and starts with the same path prefix."""
    p = urlparse(url)
    return (
        p.scheme in ("http", "https")
        and p.netloc == base_parsed.netloc
        and p.path.startswith(base_parsed.path)
    )


def crawl(
    start_url: str,
    max_pages: int = 100,
    delay_seconds: float = 0.3,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    """
    BFS crawl starting from *start_url*.

    Args:
        start_url:      The root URL to begin crawling.
        max_pages:      Hard cap on number of pages to visit (default 100).
        delay_seconds:  Polite delay between requests (default 0.3 s).
        on_progress:    Optional callback(pages_done, total_queued, current_url).

    Returns:
        List of LangChain Documents (one per page, with source metadata).
    """
    base_parsed = urlparse(start_url)
    # Normalise: strip trailing slash for prefix matching
    base_parsed = base_parsed._replace(
        path=base_parsed.path.rstrip("/") or "/"
    )

    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    documents: list[Document] = []

    logger.info("Starting crawl from %s (max_pages=%d)", start_url, max_pages)

    while queue and len(visited) < max_pages:
        url = queue.popleft()

        # Normalise URL (strip fragment, trailing slash)
        url_clean, _ = urldefrag(url)
        url_clean = url_clean.rstrip("/") or url_clean

        if url_clean in visited:
            continue
        visited.add(url_clean)

        if on_progress:
            on_progress(len(visited), len(queue) + len(visited), url_clean)

        try:
            html, resolved = _fetch_page(url_clean)
            if not html:
                logger.debug("Skipped non-HTML: %s", url_clean)
                continue

            text, links = _extract_text_and_links(html, resolved)

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": url_clean, "crawled_from": start_url},
                ))
                logger.info("Crawled [%d/%d]: %s (%d chars)",
                            len(visited), max_pages, url_clean, len(text))

            # Enqueue in-scope links not yet visited
            for link in links:
                link_clean, _ = urldefrag(link)
                link_clean = link_clean.rstrip("/") or link_clean
                if link_clean not in visited and _same_scope(link_clean, base_parsed):
                    queue.append(link_clean)

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url_clean, exc)
            continue

    logger.info(
        "Crawl complete. Visited %d pages, collected %d documents.",
        len(visited), len(documents),
    )
    return documents
