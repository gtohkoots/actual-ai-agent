from __future__ import annotations

import os
from typing import Any, Dict

from backend.agents.analysis_context import build_analysis_context
from backend.agents.analysis_llm import fallback_analysis_response, generate_analysis_response


def run_analysis_agent_turn(request: Any) -> Dict[str, Any]:
    facts = build_analysis_context(request)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_analysis_response(request, facts)

    try:
        return generate_analysis_response(request, facts)
    except Exception:
        return fallback_analysis_response(
            request,
            facts,
            "The model request failed, so this is a structured fallback response.",
        )
