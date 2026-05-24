from __future__ import annotations

from typing import Any


def render_analysis_chat_content(turn_result: dict[str, Any]) -> str:
    summary = str(turn_result.get("summary", "")).strip()
    findings = _clean_lines(turn_result.get("findings", []))
    risks = _clean_lines(turn_result.get("risks", []))
    opportunities = _clean_lines(turn_result.get("opportunities", []))
    actions = _clean_lines(turn_result.get("actions", []))
    fallback_note = str(turn_result.get("fallback_note", "")).strip()

    sections: list[str] = []
    if summary:
        sections.append(summary)
    if findings:
        sections.append("**Findings**\n" + "\n".join(f"- {item}" for item in findings))
    if risks:
        sections.append("**Risks**\n" + "\n".join(f"- {item}" for item in risks))
    if opportunities:
        sections.append("**Opportunities**\n" + "\n".join(f"- {item}" for item in opportunities))
    if actions:
        sections.append("**Next**\n" + "\n".join(f"- {item}" for item in actions))
    if fallback_note:
        sections.append(fallback_note)

    return "\n\n".join(sections) if sections else "No analysis response was generated."


def _clean_lines(items: Any) -> list[str]:
    return [str(item).strip() for item in items or [] if str(item).strip()]
