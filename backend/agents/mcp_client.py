from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from fastmcp import Client

from backend.mcp.server import build_runtime_server
from backend.mcp.schemas import MCPServerDefinition


def get_resource_payload(uri: str, *, db_path: Optional[str] = None) -> dict[str, Any]:
    """Read one MCP resource and decode the first JSON text payload."""
    return asyncio.run(_read_resource(uri, db_path=db_path))


def get_multiple_resource_payloads(
    uris: list[str], *, db_path: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """Read a small set of MCP resources and return them keyed by URI."""
    return asyncio.run(_read_resources(uris, db_path=db_path))


def call_tool_payload(
    tool_name: str,
    arguments: Optional[dict[str, Any]] = None,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Call one MCP tool and normalize its JSON or structured payload."""
    return asyncio.run(_call_tool(tool_name, arguments=arguments, db_path=db_path))


async def _read_resource(uri: str, *, db_path: Optional[str] = None) -> dict[str, Any]:
    """Open a short-lived FastMCP client session and read one resource."""
    results = await _read_resources([uri], db_path=db_path)
    return results[uri]


async def _read_resources(uris: list[str], *, db_path: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Open one FastMCP client session and read a bounded set of resources."""
    runtime = build_runtime_server(db_path=db_path)
    if isinstance(runtime, MCPServerDefinition):
        raise RuntimeError("FastMCP runtime is unavailable; cannot read MCP resources.")

    client = Client(runtime)
    async with client:
        results: dict[str, dict[str, Any]] = {}
        for uri in uris:
            contents = await client.read_resource(uri)
            if not contents:
                results[uri] = {"status": "missing", "message": f"No content returned for resource {uri}."}
                continue

            first = contents[0]
            text = getattr(first, "text", None)
            if text is None:
                raise ValueError(f"Resource {uri} did not return text content.")
            results[uri] = json.loads(text)

    return results


async def _call_tool(
    tool_name: str,
    *,
    arguments: Optional[dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Open a short-lived FastMCP client session and call one tool."""
    runtime = build_runtime_server(db_path=db_path)
    if isinstance(runtime, MCPServerDefinition):
        raise RuntimeError("FastMCP runtime is unavailable; cannot call MCP tools.")

    client = Client(runtime)
    async with client:
        result = await client.call_tool(tool_name, arguments=arguments or {})

    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return dict(structured)

    content = getattr(result, "content", None) or []
    if not content:
        return {"status": "missing", "message": f"No content returned for tool {tool_name}."}

    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        if isinstance(first, dict):
            return dict(first)
        raise ValueError(f"Tool {tool_name} did not return text or structured content.")
    return json.loads(text)
