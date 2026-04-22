from __future__ import annotations

import argparse
import sys

from backend.mcp.server import build_runtime_server, has_mcp_runtime
from backend.mcp.schemas import MCPServerDefinition


def build_parser() -> argparse.ArgumentParser:
    """Create a minimal CLI for running the planner FastMCP server."""
    parser = argparse.ArgumentParser(description="Run the finance planner MCP server.")
    parser.add_argument("--db-path", default=None, help="Optional planner DB path override.")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="FastMCP transport to use.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind for network transports.")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind for network transports.")
    parser.add_argument("--path", default="/mcp", help="Path to use for HTTP transport.")
    return parser


def main() -> None:
    """Build and run the FastMCP planner server."""
    parser = build_parser()
    args = parser.parse_args()

    if not has_mcp_runtime():
        raise RuntimeError(
            "fastmcp is not installed in the active environment. "
            "Install backend requirements first to run the MCP server."
        )

    runtime = build_runtime_server(db_path=args.db_path)
    if isinstance(runtime, MCPServerDefinition):
        raise RuntimeError("FastMCP runtime is unavailable; received abstract server definition instead.")

    run_kwargs = {"transport": args.transport}
    if args.transport in {"http", "sse"}:
        run_kwargs["host"] = args.host
        run_kwargs["port"] = args.port
    if args.transport == "http":
        run_kwargs["path"] = args.path

    runtime.run(**run_kwargs)


if __name__ == "__main__":
    main()
