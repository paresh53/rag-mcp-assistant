"""
RAG Concept #1 — Document Loading
==================================
Supports multiple file types: PDF, TXT, Markdown, DOCX, HTML, and web URLs.
Each loader is abstracted behind a common interface returning LangChain Documents.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Single-file loaders ──────────────────────────────────────────────────────

def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF using pypdf (page-level splitting)."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    logger.info("Loaded %d pages from PDF: %s", len(docs), file_path)
    return docs


def load_text(file_path: str | Path) -> list[Document]:
    """Load a plain-text or Markdown file."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    logger.info("Loaded text document: %s", file_path)
    return docs


def load_docx(file_path: str | Path) -> list[Document]:
    """Load a Microsoft Word (.docx) file."""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()
    logger.info("Loaded DOCX: %s", file_path)
    return docs


def load_html(file_path: str | Path) -> list[Document]:
    """Load an HTML file, stripping tags."""
    from langchain_community.document_loaders import BSHTMLLoader

    loader = BSHTMLLoader(str(file_path))
    docs = loader.load()
    logger.info("Loaded HTML: %s", file_path)
    return docs


def load_url(url: str) -> list[Document]:
    """
    Fetch and load a web page.
    Uses WebBaseLoader which relies on requests + BeautifulSoup.
    """
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(url)
    docs = loader.load()
    # Tag the source metadata cleanly
    for doc in docs:
        doc.metadata["source"] = url
    logger.info("Loaded web page: %s → %d docs", url, len(docs))
    return docs


def load_directory(directory: str | Path, glob: str = "**/*.*") -> list[Document]:
    """
    Recursively load all supported files in a directory.
    Supported extensions: .pdf, .txt, .md, .docx, .html
    """
    from langchain_community.document_loaders import DirectoryLoader

    LOADERS: dict[str, Any] = {
        ".pdf": ("langchain_community.document_loaders", "PyPDFLoader"),
        ".txt": ("langchain_community.document_loaders", "TextLoader"),
        ".md":  ("langchain_community.document_loaders", "TextLoader"),
        ".docx":("langchain_community.document_loaders", "Docx2txtLoader"),
        ".html":("langchain_community.document_loaders", "BSHTMLLoader"),
    }

    all_docs: list[Document] = []
    directory = Path(directory)

    for ext, (module, cls_name) in LOADERS.items():
        loader = DirectoryLoader(
            str(directory),
            glob=f"**/*{ext}",
            show_progress=True,
            use_multithreading=True,
        )
        try:
            docs = loader.load()
            all_docs.extend(docs)
            logger.info("  %s: %d documents", ext, len(docs))
        except Exception as exc:
            logger.warning("  %s: loader failed — %s", ext, exc)

    logger.info("Total documents loaded from %s: %d", directory, len(all_docs))
    return all_docs


# ── Unified dispatcher ───────────────────────────────────────────────────────

def load_document(source: str) -> list[Document]:
    """
    Auto-detect source type and dispatch to the correct loader.
    Accepts: file path (str/Path) or HTTP/S URL.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {source}")

    if path.is_dir():
        return load_directory(path)

    ext = path.suffix.lower()
    dispatch = {
        ".pdf":  load_pdf,
        ".txt":  load_text,
        ".md":   load_text,
        ".docx": load_docx,
        ".html": load_html,
        ".htm":  load_html,
    }
    loader_fn = dispatch.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file type '{ext}' for: {source}")

    return loader_fn(path)
