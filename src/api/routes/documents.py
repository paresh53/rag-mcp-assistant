"""Document ingestion and management API routes."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()


# ── Request / Response models ────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
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
