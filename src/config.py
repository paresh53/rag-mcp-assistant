"""
Central application configuration via pydantic-settings.
All values are read from environment variables or a .env file.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048

    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── Vector Store ──────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RETRIEVAL_K: int = 5
    RETRIEVAL_STRATEGY: str = "mmr"  # similarity | mmr | hybrid
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 3
    QUERY_EXPANSION_ENABLED: bool = True

    # ── MCP Server ────────────────────────────────────────────────────────────
    MCP_SERVER_NAME: str = "rag-mcp-assistant"
    MCP_SERVER_VERSION: str = "0.1.0"
    MCP_TRANSPORT: str = "sse"
    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 9001

    # ── API Server ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 9000
    API_LOG_LEVEL: str = "info"


settings = Settings()
