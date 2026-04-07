"""MCP bridge routes — call MCP tools/resources/prompts via HTTP REST API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict = {}


class PromptRequest(BaseModel):
    prompt: str
    arguments: dict[str, str] = {}


@router.get("/tools", summary="List available MCP tools")
async def list_tools():
    """List all tools registered on the MCP server."""
    from src.mcp.server import mcp
    tools = mcp._tool_manager.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in tools
        ]
    }


@router.post("/tools/call", summary="Call an MCP tool")
async def call_tool(body: ToolCallRequest):
    """Call any registered MCP tool with the given arguments."""
    from mcp.types import CallToolResult
    from src.mcp.server import mcp

    try:
        result = await mcp._tool_manager.call_tool(body.tool, body.arguments)
        return {"tool": body.tool, "result": result}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Tool '{body.tool}' not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/resources", summary="List available MCP resources")
async def list_resources():
    """List all resources exposed by the MCP server."""
    from src.mcp.server import mcp
    resources = mcp._resource_manager.list_resources()
    return {
        "resources": [
            {"uri": r.uri, "name": r.name, "description": r.description}
            for r in resources
        ]
    }


@router.get("/resources/read", summary="Read an MCP resource")
async def read_resource(uri: str):
    """Read the content of a resource by URI."""
    from src.mcp.server import mcp
    try:
        content = await mcp._resource_manager.read_resource(uri)
        return {"uri": uri, "content": content}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/prompts", summary="List available MCP prompts")
async def list_prompts():
    """List all prompt templates registered on the MCP server."""
    from src.mcp.server import mcp
    prompts = mcp._prompt_manager.list_prompts()
    return {
        "prompts": [
            {"name": p.name, "description": p.description}
            for p in prompts
        ]
    }


@router.post("/prompts/get", summary="Get a filled MCP prompt")
async def get_prompt(body: PromptRequest):
    """Retrieve and fill a named prompt template."""
    from src.mcp.server import mcp
    try:
        result = await mcp._prompt_manager.get_prompt(body.prompt, body.arguments)
        return {
            "prompt": body.prompt,
            "messages": [
                {"role": m.role, "content": m.content.text}
                for m in result.messages
                if hasattr(m.content, "text")
            ],
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt '{body.prompt}' not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
