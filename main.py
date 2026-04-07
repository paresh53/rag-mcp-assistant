"""
RAG + MCP Assistant — entry point.

Run the FastAPI server:
    python main.py

Or directly:
    uvicorn src.api.app:app --reload
"""
import uvicorn

if __name__ == "__main__":
    from src.config import settings

    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.API_LOG_LEVEL,
        reload=True,
    )
