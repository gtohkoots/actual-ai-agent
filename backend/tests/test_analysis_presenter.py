import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis.presenter import render_analysis_chat_content


def test_render_analysis_chat_content_renders_sections():
    content = render_analysis_chat_content(
        {
            "summary": "Spending is trending higher this month.",
            "findings": ["Dining spend increased by $120."],
            "risks": ["Dining is pacing above baseline."],
            "opportunities": ["Review dining transactions for outliers."],
            "actions": ["Compare to last week"],
        }
    )

    assert "Spending is trending higher this month." in content
    assert "**Findings**" in content
    assert "**Risks**" in content
    assert "**Opportunities**" in content
    assert "**Next**" in content
