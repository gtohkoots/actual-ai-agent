from __future__ import annotations

import json
from typing import Any, Dict

from backend.agents.analysis.executor import AnalysisToolExecutionError, execute_analysis_tool_plan
from backend.agents.analysis.intent import interpret_analysis_request
from backend.agents.analysis.llm import fallback_analysis_response, generate_analysis_response
from backend.agents.analysis.tool_planner import plan_analysis_tool_calls


def _log_analysis_request_payload(request: Any, analysis_request: Dict[str, Any]) -> None:
    payload = {
        "message": getattr(request, "message", ""),
        "analysis_request": analysis_request,
    }
    print("[analysis.intent] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))


def run_analysis_agent_turn(request: Any) -> Dict[str, Any]:
    analysis_request = interpret_analysis_request(request)
    _log_analysis_request_payload(request, analysis_request)
    tool_plan = plan_analysis_tool_calls(request, analysis_request)

    try:
        execution_result = execute_analysis_tool_plan(tool_plan)
    except AnalysisToolExecutionError as exc:
        return fallback_analysis_response(
            request,
            tool_plan,
            {
                "tool_results": [],
                "used_tools": [],
                "failures": [{"tool": "plan_validation", "error": str(exc), "step": -1, "args": {}}],
            },
            analysis_request=analysis_request,
            reason="The tool plan could not be executed safely, so this is a structured fallback response.",
        )

    try:
        return generate_analysis_response(
            request,
            tool_plan,
            execution_result,
            analysis_request=analysis_request,
        )
    except Exception:
        return fallback_analysis_response(
            request,
            tool_plan,
            execution_result,
            analysis_request=analysis_request,
            reason="The model request failed, so this is a structured fallback response.",
        )
