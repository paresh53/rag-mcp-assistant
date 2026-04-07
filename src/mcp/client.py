"""
MCP Concept — Client
=====================
An MCP client connects to an MCP server and:
  1. Lists available tools / resources / prompts
  2. Calls tools and processes results
  3. Reads resources
  4. Gets prompts

This client supports both stdio (subprocess) and SSE (HTTP) transports.
Use it in tests, scripts, or integrations with other systems.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Async MCP client.

    Usage (stdio):
        async with MCPClient.connect_stdio("python", ["-m", "src.mcp.server"]) as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_documents", {"query": "AI"})

    Usage (SSE):
        async with MCPClient.connect_sse("http://localhost:8001") as client:
            result = await client.call_tool("ask_question", {"question": "What is RAG?"})
    """

    def __init__(self, session):
        self._session = session
        self._capabilities: dict = {}

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    async def connect_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "MCPClient":
        """Connect to an MCP server via stdio (launches subprocess)."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=command, args=args or [], env=env)
        read, write = await stdio_client(params).__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        result = await session.initialize()
        client = cls(session)
        client._capabilities = result.capabilities.__dict__ if result.capabilities else {}
        logger.info("MCP stdio client connected — capabilities: %s", client._capabilities)
        return client

    @classmethod
    async def connect_sse(cls, base_url: str) -> "MCPClient":
        """Connect to an MCP server running over SSE/HTTP."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        read, write = await sse_client(f"{base_url}/sse").__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        result = await session.initialize()
        client = cls(session)
        client._capabilities = result.capabilities.__dict__ if result.capabilities else {}
        logger.info("MCP SSE client connected to %s", base_url)
        return client

    # ── Tools ─────────────────────────────────────────────────────────────────

    async def list_tools(self) -> list[dict]:
        """List all tools available on the server."""
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by name with the given arguments."""
        from mcp.types import CallToolResult

        result: CallToolResult = await self._session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"MCP tool '{name}' returned error: {result.content}")

        # Parse JSON content if possible
        raw = result.content[0].text if result.content else ""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            return raw

    # ── Resources ─────────────────────────────────────────────────────────────

    async def list_resources(self) -> list[dict]:
        """List all resources exposed by the server."""
        result = await self._session.list_resources()
        return [
            {"uri": r.uri, "name": r.name, "description": r.description, "mime_type": r.mimeType}
            for r in result.resources
        ]

    async def read_resource(self, uri: str) -> Any:
        """Read the content of a resource by URI."""
        result = await self._session.read_resource(uri)
        raw = result.contents[0].text if result.contents else ""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            return raw

    # ── Prompts ───────────────────────────────────────────────────────────────

    async def list_prompts(self) -> list[dict]:
        """List all prompts available on the server."""
        result = await self._session.list_prompts()
        return [
            {"name": p.name, "description": p.description, "arguments": p.arguments}
            for p in result.prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> dict:
        """Fetch a prompt template with filled arguments."""
        result = await self._session.get_prompt(name, arguments or {})
        return {
            "description": result.description,
            "messages": [
                {"role": m.role, "content": m.content.text}
                for m in result.messages
                if hasattr(m.content, "text")
            ],
        }

    # ── Subscriptions & notifications ─────────────────────────────────────────

    async def subscribe_resource(self, uri: str) -> None:
        """
        MCP Concept — Resource Subscriptions.
        Ask the server to send notifications when resource content changes.
        """
        await self._session.subscribe_resource(uri)
        logger.info("Subscribed to resource: %s", uri)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._session.__aexit__(None, None, None)

    # ── Convenience async context manager ────────────────────────────────────

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# ── Quick demo ────────────────────────────────────────────────────────────────

async def demo():
    """Quick demonstration of the MCP client (requires server running on port 8001)."""
    from src.config import settings

    base_url = f"http://localhost:{settings.MCP_PORT}"
    print(f"Connecting to MCP server at {base_url} ...")

    async with MCPClient.connect_sse(base_url) as client:
        print("\n=== Tools ===")
        for tool in await client.list_tools():
            print(f"  {tool['name']}: {tool['description']}")

        print("\n=== Resources ===")
        for resource in await client.list_resources():
            print(f"  {resource['uri']}: {resource['name']}")

        print("\n=== Prompts ===")
        for prompt in await client.list_prompts():
            print(f"  {prompt['name']}: {prompt['description']}")

        print("\n=== Collection Stats ===")
        stats = await client.read_resource("collection://stats")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(demo())
