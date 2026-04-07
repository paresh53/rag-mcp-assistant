"""
MCP Concept — Server
=====================
The MCP server exposes Tools, Resources, and Prompts over two transports:

  • stdio  — used by Claude Desktop and local MCP clients
  • SSE    — Server-Sent Events over HTTP (for web integrations)

Additional MCP concepts demonstrated here:
  • Logging           — structured log messages sent to the MCP client
  • Progress tokens   — report progress of long-running operations
  • Roots             — declare filesystem boundaries the server operates in
  • Sampling          — the server requests the LLM to generate text via client
  • Lifespan hooks    — startup/shutdown resource management
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

from src.config import settings
from src.mcp.prompts import register_prompts
from src.mcp.resources import register_resources
from src.mcp.tools import register_tools

logger = structlog.get_logger(__name__)


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """
    MCP server lifespan — runs once at startup and once at shutdown.
    Use this to warm up models, open connections, etc.
    """
    logger.info("mcp_server_starting", name=settings.MCP_SERVER_NAME)

    # Pre-warm the embedding model so the first query isn't slow
    from src.rag.embeddings import build_embedding_model
    build_embedding_model()
    logger.info("embedding_model_loaded")

    yield  # Server is running

    logger.info("mcp_server_stopping", name=settings.MCP_SERVER_NAME)


# ── Build the FastMCP server ──────────────────────────────────────────────────

mcp = FastMCP(
    name=settings.MCP_SERVER_NAME,
    version=settings.MCP_SERVER_VERSION,
    # Roots: declare the filesystem boundaries this server is allowed to access
    roots=[str(Path("./data").resolve())],
    lifespan=lifespan,
)

# Register all tools, resources, and prompts
register_tools(mcp)
register_resources(mcp)
register_prompts(mcp)


# ── MCP Sampling (server-side LLM call via client) ───────────────────────────

async def request_sampling(
    messages: list[dict],
    system_prompt: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """
    MCP Concept — Sampling.

    The MCP server can ask the *client* (e.g. Claude Desktop) to perform
    an LLM inference call. This is the reverse of the normal tool-call flow.

    Use cases:
      - Server needs LLM reasoning during a multi-step tool
      - Server needs model capabilities it doesn't have locally
      - Keeps LLM secrets on the client side

    Note: Sampling requires the client to grant `sampling` capability.
    Here we fall back to the server's own LLM if sampling isn't available.
    """
    try:
        # Attempt client-side sampling via MCP protocol
        result = await mcp.get_context().session.create_message(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        return result.content.text if hasattr(result.content, "text") else str(result.content)
    except Exception as exc:
        logger.warning("sampling_fallback", error=str(exc))
        # Fall back: use the server's own LLM directly
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import StrOutputParser

        from src.llm.base import build_llm

        llm = build_llm()
        lc_messages = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        for msg in messages:
            lc_messages.append(HumanMessage(content=msg.get("content", "")))

        return await (llm | StrOutputParser()).ainvoke(lc_messages)


# ── MCP Logging ───────────────────────────────────────────────────────────────

def setup_mcp_logging():
    """
    Configure structured logging that forwards to the MCP client via the
    logging notification mechanism.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ── Entry points ──────────────────────────────────────────────────────────────

def run_stdio():
    """Run MCP server over stdio transport (for Claude Desktop, local clients)."""
    setup_mcp_logging()
    logger.info("mcp_transport_stdio_starting")
    mcp.run(transport="stdio")


def run_sse(host: str | None = None, port: int | None = None):
    """Run MCP server over SSE/HTTP transport (for web integrations)."""
    setup_mcp_logging()
    _host = host or settings.MCP_HOST
    _port = port or settings.MCP_PORT
    logger.info("mcp_transport_sse_starting", host=_host, port=_port)
    mcp.run(transport="sse", host=_host, port=_port)


if __name__ == "__main__":
    transport = settings.MCP_TRANSPORT.lower()
    if transport == "stdio":
        run_stdio()
    elif transport == "sse":
        run_sse()
    else:
        print(f"Unknown transport: {transport}. Use 'stdio' or 'sse'.", file=sys.stderr)
        sys.exit(1)
