"""
FastAPI application — REST API wrapping the RAG pipeline.
Demonstrates streaming, async endpoints, and proper error handling.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings

logger = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm models at startup."""
    logger.info("api_starting", host=settings.API_HOST, port=settings.API_PORT)
    from src.rag.embeddings import build_embedding_model
    from src.llm.base import build_llm
    build_embedding_model()
    build_llm()
    logger.info("models_loaded")
    yield
    logger.info("api_stopping")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG + MCP Assistant",
        description=(
            "A full-featured RAG (Retrieval-Augmented Generation) API with "
            "MCP (Model Context Protocol) server integration. "
            "Demonstrates all key RAG and MCP concepts."
        ),
        version="0.1.0",
        lifespan=lifespan,
        root_path="/gotoassistant",
        docs_url="/docs",
        redoc_url="/redoc",
        servers=[{"url": "/gotoassistant", "description": "GotoAssistant API"}],
    )

    # CORS — allow all origins in development, restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from src.api.routes.health import router as health_router
    from src.api.routes.documents import router as documents_router
    from src.api.routes.rag import router as rag_router
    from src.api.routes.mcp import router as mcp_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(documents_router, prefix="/documents", tags=["Documents"])
    app.include_router(rag_router, prefix="/rag", tags=["RAG"])
    app.include_router(mcp_router, prefix="/mcp", tags=["MCP"])

    # Serve the chat UI at root
    import pathlib
    static_dir = pathlib.Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def ui():
        return FileResponse(static_dir / "index.html")

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("unhandled_exception", error=str(exc), path=str(request.url))
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred.", "error": str(exc)},
        )

    return app


app = create_app()


# ── Run directly ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.API_LOG_LEVEL,
        reload=True,
    )


if __name__ == "__main__":
    main()
