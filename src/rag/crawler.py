"""
Web Crawler — recursively discovers and loads all pages under a base URL.

Supports two modes:
  - Static mode (fast): plain HTTP + BeautifulSoup for traditional HTML sites
  - JS mode (headless): Playwright/Chromium for JavaScript-rendered SPAs
    (IBM Docs, React/Angular/Vue sites, etc.)

Given a top-level URL it will:
  1. Auto-detect if the site requires JavaScript rendering
  2. Crawl all same-domain pages under the base path prefix (BFS)
  3. Return LangChain Documents ready for chunking + embedding
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable
from urllib.parse import urljoin, urlparse, urldefrag

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_CONTENT_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th",
    "pre", "code", "blockquote", "article", "section", "main", "div",
}
_JS_NEEDS_THRESHOLD = 5   # fewer than this many text chunks → assume JS-rendered


# ── HTML parsing helpers ───────────────────────────────────────────────────────

def _parse_html(html: str, base_url: str) -> tuple[str, list[str]]:
    """Return (cleaned text, absolute links) from an HTML string."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "noscript", "svg", "img"]):
        tag.decompose()

    chunks: list[str] = []
    for tag in soup.find_all(_CONTENT_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 40:
            chunks.append(text)
    text = "\n\n".join(chunks)

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href, _ = urldefrag(a["href"])
        if href:
            links.append(urljoin(base_url, href))

    return text, links


def _same_scope(url: str, base_parsed) -> bool:
    p = urlparse(url)
    return (
        p.scheme in ("http", "https")
        and p.netloc == base_parsed.netloc
        and p.path.startswith(base_parsed.path)
    )


def _normalise(url: str) -> str:
    clean, _ = urldefrag(url)
    return clean.rstrip("/") or clean


# ── Static (plain HTTP) fetch ──────────────────────────────────────────────────

def _fetch_static(url: str, timeout: int = 15) -> tuple[str, str]:
    """Return (html, resolved_url) via plain urllib."""
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return "", resp.url
        return resp.read().decode("utf-8", errors="replace"), resp.url


# ── JS-rendered (Playwright) crawl ────────────────────────────────────────────

def _crawl_js(
    start_url: str,
    base_parsed,
    max_pages: int,
    delay_seconds: float,
    on_progress: Callable | None,
) -> list[Document]:
    """Crawl a JS-rendered site using Playwright headless Chromium."""
    from playwright.sync_api import sync_playwright

    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    documents: list[Document] = []

    logger.info("Using headless Chromium (JS-rendered site detected)")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        # Block images/fonts/media to speed up crawling
        page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,mp4,mp3}", lambda r: r.abort())

        while queue and len(visited) < max_pages:
            url = _normalise(queue.popleft())
            if url in visited:
                continue
            visited.add(url)

            if on_progress:
                on_progress(len(visited), len(queue) + len(visited), url)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                # Wait for main content to load
                try:
                    page.wait_for_selector("main, article, .content, #content", timeout=5000)
                except Exception:
                    pass
                time.sleep(0.5)  # small extra wait for dynamic content

                html = page.content()
                text, links = _parse_html(html, url)

                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": url, "crawled_from": start_url},
                    ))
                    logger.info("Crawled (JS) [%d/%d]: %s (%d chars)",
                                len(visited), max_pages, url, len(text))

                for link in links:
                    lc = _normalise(link)
                    if lc not in visited and _same_scope(lc, base_parsed):
                        queue.append(lc)

                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            except Exception as exc:
                logger.warning("JS fetch failed %s: %s", url, exc)
                continue

        browser.close()

    return documents


# ── Static crawl ──────────────────────────────────────────────────────────────

def _crawl_static(
    start_url: str,
    base_parsed,
    max_pages: int,
    delay_seconds: float,
    on_progress: Callable | None,
    first_html: str,
    first_resolved: str,
) -> list[Document]:
    """BFS crawl using plain HTTP requests."""
    visited: set[str] = set()
    queue: deque[str] = deque()
    documents: list[Document] = []

    # Process the already-fetched first page
    def process(url: str, html: str, resolved: str):
        text, links = _parse_html(html, resolved)
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": url, "crawled_from": start_url},
            ))
        for link in links:
            lc = _normalise(link)
            if lc not in visited and _same_scope(lc, base_parsed):
                queue.append(lc)

    url0 = _normalise(start_url)
    visited.add(url0)
    if on_progress:
        on_progress(1, 1, url0)
    process(url0, first_html, first_resolved)

    while queue and len(visited) < max_pages:
        url = _normalise(queue.popleft())
        if url in visited:
            continue
        visited.add(url)

        if on_progress:
            on_progress(len(visited), len(queue) + len(visited), url)

        try:
            html, resolved = _fetch_static(url)
            if html:
                process(url, html, resolved)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        except Exception as exc:
            logger.warning("Static fetch failed %s: %s", url, exc)

    return documents


# ── Public API ────────────────────────────────────────────────────────────────

def crawl(
    start_url: str,
    max_pages: int = 100,
    delay_seconds: float = 0.3,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    """
    BFS crawl starting from *start_url*.

    Automatically detects if the site is JS-rendered and switches to
    Playwright headless Chromium if needed (e.g. IBM Docs, React SPAs).

    Args:
        start_url:   Root URL to crawl.
        max_pages:   Hard cap on pages (default 100).
        delay_seconds: Polite delay between requests (default 0.3s).
        on_progress: Optional callback(pages_done, queued, current_url).

    Returns:
        List of LangChain Documents.
    """
    base_parsed = urlparse(start_url)
    base_parsed = base_parsed._replace(path=base_parsed.path.rstrip("/") or "/")

    logger.info("Starting crawl from %s (max_pages=%d)", start_url, max_pages)

    # ── Probe first page to decide static vs JS mode ──
    try:
        first_html, first_resolved = _fetch_static(start_url)
    except Exception as exc:
        logger.warning("Initial fetch failed, trying JS mode: %s", exc)
        first_html, first_resolved = "", start_url

    if first_html:
        text_probe, _ = _parse_html(first_html, first_resolved)
        chunk_count = len([c for c in text_probe.split("\n\n") if len(c) > 40])
        js_mode = chunk_count < _JS_NEEDS_THRESHOLD
        logger.info(
            "Probe: %d text chunks found → %s mode",
            chunk_count, "JS/headless" if js_mode else "static",
        )
    else:
        js_mode = True
        logger.info("Empty response on probe → using JS/headless mode")

    if js_mode:
        docs = _crawl_js(start_url, base_parsed, max_pages, delay_seconds, on_progress)
    else:
        docs = _crawl_static(
            start_url, base_parsed, max_pages, delay_seconds,
            on_progress, first_html, first_resolved,
        )

    logger.info("Crawl complete. Collected %d documents.", len(docs))
    return docs
