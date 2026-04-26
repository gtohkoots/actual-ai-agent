from __future__ import annotations

import argparse
import json

from backend.agents.planner_agent import run_planner_agent


def build_parser() -> argparse.ArgumentParser:
    """Create a tiny CLI for running the planner agent outside the app."""
    parser = argparse.ArgumentParser(description="Run the MCP-backed planner agent.")
    parser.add_argument("message", help="User message or review request for the planner agent.")
    parser.add_argument("--db-path", default=None, help="Optional planner DB path override.")
    return parser


def main() -> None:
    """Run the planner agent and print a JSON response."""
    args = build_parser().parse_args()
    payload = run_planner_agent(args.message, db_path=args.db_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
