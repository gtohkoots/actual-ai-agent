from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from backend.services import documents as document_services
from backend.services import ledger_analysis


ToolExecutor = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class AnalysisToolSpec:
    name: str
    description: str
    arg_schema: Mapping[str, Any]
    executor: ToolExecutor

    def descriptor(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arg_schema": dict(self.arg_schema),
        }


TOOL_REGISTRY: Dict[str, AnalysisToolSpec] = {
    "get_portfolio_summary": AnalysisToolSpec(
        name="get_portfolio_summary",
        description="Summarize income, expense, net cash flow, and transaction count for a scoped time window.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "account_pid": {"type": "string"},
                "account_name": {"type": "string"},
            },
        },
        executor=ledger_analysis.get_portfolio_summary,
    ),
    "get_category_spend": AnalysisToolSpec(
        name="get_category_spend",
        description="Return the top expense categories for a period with spend shares.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "limit": {"type": "integer", "minimum": 1, "default": 10},
                "account_pid": {"type": "string"},
                "account_name": {"type": "string"},
            },
        },
        executor=ledger_analysis.get_category_spend,
    ),
    "get_account_breakdown": AnalysisToolSpec(
        name="get_account_breakdown",
        description="Break down income, expense, and net cash flow by account for a period.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
            },
        },
        executor=ledger_analysis.get_account_breakdown,
    ),
    "get_transaction_slice": AnalysisToolSpec(
        name="get_transaction_slice",
        description="Return a bounded filtered transaction slice for drill-down analysis.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "category_name": {"type": "string"},
                "payee": {"type": "string"},
                "account_name": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "default": 50},
            },
        },
        executor=ledger_analysis.get_transaction_slice,
    ),
    "compare_periods": AnalysisToolSpec(
        name="compare_periods",
        description="Compare two periods and return delta totals and category changes.",
        arg_schema={
            "required": ["current_start", "current_end", "previous_start", "previous_end"],
            "properties": {
                "current_start": {"type": "string", "format": "date"},
                "current_end": {"type": "string", "format": "date"},
                "previous_start": {"type": "string", "format": "date"},
                "previous_end": {"type": "string", "format": "date"},
            },
        },
        executor=ledger_analysis.compare_periods,
    ),
    "get_spending_drift": AnalysisToolSpec(
        name="get_spending_drift",
        description="Explain spending drift against a baseline period, including the largest category deltas.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "baseline_start": {"type": "string", "format": "date"},
                "baseline_end": {"type": "string", "format": "date"},
            },
        },
        executor=ledger_analysis.get_spending_drift,
    ),
    "detect_spending_anomalies": AnalysisToolSpec(
        name="detect_spending_anomalies",
        description="Detect unusual expense transactions inside a requested period.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
            },
        },
        executor=ledger_analysis.detect_spending_anomalies,
    ),
    "find_recurring_charges": AnalysisToolSpec(
        name="find_recurring_charges",
        description="Find recurring charge candidates inside a requested period.",
        arg_schema={
            "required": ["period_start", "period_end"],
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
            },
        },
        executor=ledger_analysis.find_recurring_charges,
    ),
    "search_documents": AnalysisToolSpec(
        name="search_documents",
        description="Search indexed finance artifacts for matching text, categories, or payees.",
        arg_schema={
            "required": [],
            "properties": {
                "query": {"type": "string"},
                "doc_type": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "limit": {"type": "integer", "minimum": 1, "default": 5},
            },
        },
        executor=document_services.search_documents,
    ),
    "search_reports": AnalysisToolSpec(
        name="search_reports",
        description="Search historical weekly reports relevant to a free-form analysis question.",
        arg_schema={
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "limit": {"type": "integer", "minimum": 1, "default": 5},
            },
        },
        executor=document_services.search_reports,
    ),
    "search_past_weeks_by_category": AnalysisToolSpec(
        name="search_past_weeks_by_category",
        description="Find historical weekly snapshots that include a requested spending category.",
        arg_schema={
            "required": ["category"],
            "properties": {
                "category": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "limit": {"type": "integer", "minimum": 1, "default": 5},
            },
        },
        executor=document_services.search_past_weeks_by_category,
    ),
    "find_similar_spending_weeks": AnalysisToolSpec(
        name="find_similar_spending_weeks",
        description="Find prior weekly snapshots with similar category and cash-flow patterns.",
        arg_schema={
            "required": ["start_date", "end_date"],
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "limit": {"type": "integer", "minimum": 1, "default": 3},
            },
        },
        executor=document_services.find_similar_spending_weeks,
    ),
    "get_recent_anomalies": AnalysisToolSpec(
        name="get_recent_anomalies",
        description="Load recent anomaly records from historical weekly snapshots.",
        arg_schema={
            "required": [],
            "properties": {
                "payee": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "default": 5},
            },
        },
        executor=document_services.get_recent_anomalies,
    ),
}


def get_analysis_tool_registry() -> Dict[str, AnalysisToolSpec]:
    return dict(TOOL_REGISTRY)


def list_analysis_tool_specs() -> list[Dict[str, Any]]:
    return [spec.descriptor() for spec in TOOL_REGISTRY.values()]


def get_analysis_tool_spec(name: str) -> AnalysisToolSpec:
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown analysis tool: {name}") from exc


def execute_analysis_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
    return get_analysis_tool_spec(name).executor(**kwargs)
