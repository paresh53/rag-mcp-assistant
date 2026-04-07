from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    """Check that all components (embedding model, vector store) are ready."""
    checks = {}

    try:
        from src.rag.embeddings import build_embedding_model
        build_embedding_model()
        checks["embedding_model"] = "ok"
    except Exception as exc:
        checks["embedding_model"] = f"error: {exc}"

    try:
        from src.rag.vector_store import get_collection_stats
        stats = get_collection_stats()
        checks["vector_store"] = f"ok ({stats['total_documents']} docs)"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    overall = "ok" if all("ok" in v for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
