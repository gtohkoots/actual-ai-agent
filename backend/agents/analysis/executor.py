from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.analysis.tools import get_analysis_tool_spec


class AnalysisToolExecutionError(ValueError):
    """Raised when a tool plan cannot be executed safely."""


def _validate_tool_call(tool_call: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    if not isinstance(tool_call, dict):
        raise AnalysisToolExecutionError("Tool call must be an object.")

    tool_name = str(tool_call.get("tool", "")).strip()
    if not tool_name:
        raise AnalysisToolExecutionError("Tool call is missing a tool name.")

    spec = get_analysis_tool_spec(tool_name)
    raw_args = tool_call.get("args")
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise AnalysisToolExecutionError(f"Tool '{tool_name}' args must be an object.")

    properties = spec.arg_schema.get("properties", {})
    cleaned_args = {key: value for key, value in raw_args.items() if key in properties and value is not None}
    required = spec.arg_schema.get("required", [])
    missing = [key for key in required if key not in cleaned_args]
    if missing:
        raise AnalysisToolExecutionError(
            f"Tool '{tool_name}' is missing required args: {', '.join(sorted(missing))}."
        )

    return tool_name, cleaned_args


def execute_analysis_tool_plan(tool_plan: Dict[str, Any]) -> Dict[str, Any]:
    raw_calls = tool_plan.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise AnalysisToolExecutionError("Tool plan must include a non-empty tool_calls list.")

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    used_tools: List[str] = []

    for index, raw_call in enumerate(raw_calls):
        tool_name, cleaned_args = _validate_tool_call(raw_call)
        spec = get_analysis_tool_spec(tool_name)
        try:
            result = spec.executor(**cleaned_args)
        except Exception as exc:
            failures.append(
                {
                    "tool": tool_name,
                    "args": cleaned_args,
                    "step": index,
                    "error": str(exc),
                }
            )
            continue

        results.append(
            {
                "tool": tool_name,
                "args": cleaned_args,
                "step": index,
                "result": result,
            }
        )
        used_tools.append(tool_name)

    return {
        "scope": tool_plan.get("scope"),
        "intent": tool_plan.get("intent"),
        "reasoning": tool_plan.get("reasoning"),
        "planning_mode": tool_plan.get("planning_mode"),
        "tool_results": results,
        "used_tools": used_tools,
        "failures": failures,
    }
