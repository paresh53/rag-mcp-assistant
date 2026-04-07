"""Document ingestion and management API routes."""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


# ── Request / Response models ────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
    chunk_size: int = 512
    chunk_overlap: int = 64
    collection: str = ""


class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 100
    chunk_size: int = 512
    chunk_overlap: int = 64
    collection: str = ""


class IngestResponse(BaseModel):
    source: str
    raw_documents: int
    chunks_created: int
    ids: list[str]
    status: str


class CollectionStatsResponse(BaseModel):
    collection: str
    total_documents: int
    persist_directory: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/ingest/url", response_model=IngestResponse, summary="Ingest a URL")
async def ingest_url(body: IngestURLRequest):
    """
    Load a web page or file URL, chunk it, embed it, and store in ChromaDB.
    """
    from src.rag.document_loader import load_document
    from src.rag.text_splitter import SplitStrategy, split_documents
    from src.rag.vector_store import add_documents

    try:
        raw_docs = load_document(body.url)
        chunks = split_documents(
            raw_docs,
            strategy=SplitStrategy.RECURSIVE,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
        ids = add_documents(chunks, collection_name=body.collection or None)
        return IngestResponse(
            source=body.url,
            raw_documents=len(raw_docs),
            chunks_created=len(chunks),
            ids=ids[:10],
            status="success",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest/file", response_model=IngestResponse, summary="Upload and ingest a file")
async def ingest_file(
    file: Annotated[UploadFile, File(description="File to ingest (PDF, TXT, DOCX, MD, HTML)")],
    chunk_size: Annotated[int, Form()] = 512,
    chunk_overlap: Annotated[int, Form()] = 64,
    collection: Annotated[str, Form()] = "",
):
    """
    Upload a file, chunk it, embed it, and store in ChromaDB.
    Supported formats: PDF, TXT, Markdown, DOCX, HTML.
    """
    from src.rag.document_loader import load_document
    from src.rag.text_splitter import SplitStrategy, split_documents
    from src.rag.vector_store import add_documents

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        raw_docs = load_document(tmp_path)
        # Override source metadata with original filename
        for doc in raw_docs:
            doc.metadata["source"] = file.filename or tmp_path

        chunks = split_documents(
            raw_docs,
            strategy=SplitStrategy.RECURSIVE,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        ids = add_documents(chunks, collection_name=collection or None)
        return IngestResponse(
            source=file.filename or tmp_path,
            raw_documents=len(raw_docs),
            chunks_created=len(chunks),
            ids=ids[:10],
            status="success",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post(
    "/ingest/crawl",
    summary="Crawl a website and ingest all pages (SSE stream)",
    response_class=StreamingResponse,
)
async def ingest_crawl(body: CrawlRequest):
    """
    Recursively crawl all pages under the given URL (same domain + path prefix),
    chunk and embed everything.

    Returns a Server-Sent Events stream with progress updates so the browser
    can show a live progress bar. Final event type is 'done' or 'error'.

    Example SSE events:
        event: progress
        data: {"page": 3, "total_queued": 12, "url": "https://..."}

        event: done
        data: {"pages_crawled": 45, "chunks_created": 820, "status": "success"}
    """
    from src.rag.crawler import crawl
    from src.rag.text_splitter import SplitStrategy, split_documents
    from src.rag.vector_store import add_documents

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        progress_events: asyncio.Queue[dict | None] = asyncio.Queue()

        def on_progress(done: int, queued: int, url: str) -> None:
            progress_events.put_nowait({"page": done, "total_queued": queued, "url": url})

        async def run_crawl():
            try:
                docs = await loop.run_in_executor(
                    None,
                    lambda: crawl(
                        start_url=body.url,
                        max_pages=body.max_pages,
                        delay_seconds=0.2,
                        on_progress=on_progress,
                    ),
                )
                return docs, None
            except Exception as exc:
                return [], str(exc)
            finally:
                await progress_events.put(None)  # sentinel

        crawl_task = asyncio.create_task(run_crawl())

        # Stream progress events while crawl runs
        while True:
            event = await progress_events.get()
            if event is None:
                break
            yield f"event: progress\ndata: {json.dumps(event)}\n\n"

        docs, error = await crawl_task

        if error:
            yield f"event: error\ndata: {json.dumps({'detail': error})}\n\n"
            return

        if not docs:
            yield f"event: error\ndata: {json.dumps({'detail': 'No pages could be crawled. The site may require login or block bots.'})}\n\n"
            return

        # Chunk + embed in executor so we don't block the event loop
        try:
            def index_docs():
                chunks = split_documents(
                    docs,
                    strategy=SplitStrategy.RECURSIVE,
                    chunk_size=body.chunk_size,
                    chunk_overlap=body.chunk_overlap,
                )
                ids = add_documents(chunks, collection_name=body.collection or None)
                return len(chunks), ids

            chunks_created, ids = await loop.run_in_executor(None, index_docs)

            yield (
                f"event: done\ndata: {json.dumps({'pages_crawled': len(docs), 'chunks_created': chunks_created, 'status': 'success'})}\n\n"
            )
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/stats", response_model=CollectionStatsResponse, summary="Vector store statistics")
async def collection_stats(collection: str = ""):
    """Return the number of documents in the vector store."""
    from src.rag.vector_store import get_collection_stats
    return get_collection_stats(collection_name=collection or None)


@router.delete("/collection", summary="Clear collection")
async def clear_collection(collection: str = ""):
    """Delete all documents in a collection. Irreversible!"""
    from src.rag.vector_store import delete_collection
    delete_collection(collection_name=collection or None)
    return {"status": "deleted", "collection": collection or "default"}
