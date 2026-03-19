# =============================
# Daily Snapshot Tool
# =============================
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# --- project imports ---
from services import insights
from utils.db import get_transactions_in_date_range


# =============================
# Lightweight explicit state
# =============================
STATE = {
    "start_date": None,   # "YYYY-MM-DD"
    "end_date": None,     # "YYYY-MM-DD"
    # Preferences (can move to a config file later)
    "big_expense_threshold": 200.0,
    "top_n_categories": 5,
    "top_n_payees": 5,
}


def _current_week_range() -> tuple[str, str]:
    today = datetime.today().date()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


# =============================
# Pydantic schemas for tools
# =============================
class TimeRange(BaseModel):
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")


class SaveReportInput(BaseModel):
    markdown: str = Field(..., description="The report content in Markdown")
    tag: Optional[str] = Field(None, description="Optional filename tag, e.g. '2025-08-31'")


class DailySnapshotInput(BaseModel):
    date: Optional[str] = Field(None, description="快照日期，YYYY-MM-DD，留空则为今天")


# =============================
# Tools
# =============================
@tool(args_schema=TimeRange, return_direct=False)
def update_time_window_tool(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """更新当前分析时间区间。必须使用 'YYYY-MM-DD' 格式。若未提供则设置为本周（周一到周日）。返回已生效的区间。"""
    if not start_date or not end_date:
        start_date, end_date = _current_week_range()
    # basic validation
    try:
        datetime.strptime(start_date, "%Y-%m-%d"); datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        return "日期格式错误，请使用 'YYYY-MM-DD'"
    STATE["start_date"], STATE["end_date"] = start_date, end_date
    return f"已设置时间区间: {start_date} ~ {end_date}"


@tool(args_schema=TimeRange)
def get_weekly_data_tool(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """获取某时间段的全部交易（JSON 数组）。列: date, amount(美元), payee, category, account。最多返回500行。"""
    s = start_date or STATE.get("start_date")
    e = end_date or STATE.get("end_date")
    if not s or not e:
        return "未设置时间区间，请先调用 update_time_window_tool 或传入参数"

    df = get_transactions_in_date_range(s, e, join_names=True, dollars=True)
    for source, alias in (("category_name", "category"), ("account_name", "account")):
        if alias not in df.columns and source in df.columns:
            df[alias] = df[source]
    df = df[[c for c in ["date", "amount", "payee", "category", "account"] if c in df.columns]].sort_values("date")
    if len(df) > 500:
        df = df.tail(500)
    # ensure floats with 2 decimals for amount
    df["amount"] = df["amount"].astype(float).round(2)
    return df.to_json(orient="records", date_format="iso", force_ascii=False)


@tool(args_schema=TimeRange)
def get_week_rollups_tool(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """返回该时间段的关键指标与榜单(JSON)：总收入/总支出/净现金流、Top 分类、Top payees、>$threshold 大额支出，并将结果保存到 weekly_snapshot 目录。"""
    s = start_date or STATE.get("start_date")
    e = end_date or STATE.get("end_date")
    if not s or not e:
        return "未设置时间区间，请先调用 update_time_window_tool 或传入参数"
    payload = insights.get_week_rollups(
        s,
        e,
        df=None,
        top_n_categories=STATE["top_n_categories"],
        top_n_payees=STATE["top_n_payees"],
        big_expense_threshold=STATE["big_expense_threshold"],
    )
    # 保存到 weekly_snapshot 目录
    snapshot_dir = Path("weekly_snapshots")
    snapshot_dir.mkdir(exist_ok=True)
    file_path = snapshot_dir / f"week_{s}_to_{e}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False)


@tool(args_schema=TimeRange)
def compare_to_last_week_tool(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """返回该时间段与上周对比(JSON): 收入/支出/净现金流差值与百分比、分类变化Top列表。"""
    s = start_date or STATE.get("start_date")
    e = end_date or STATE.get("end_date")
    if not s or not e:
        return "未设置时间区间，请先调用 update_time_window_tool 或传入参数"
    payload = insights.compare_week_over_week(s, e)
    return json.dumps(payload, ensure_ascii=False)


@tool(args_schema=SaveReportInput)
def save_weekly_report_tool(markdown: str, tag: Optional[str] = None) -> str:
    """将生成的 markdown 报告保存到 weekly_reports 目录。返回保存路径。"""
    ts = tag or datetime.now().strftime("%Y-%m-%d")
    p = Path("weekly_reports"); p.mkdir(exist_ok=True)
    path = p / f"weekly_report_{ts}.md"
    path.write_text(markdown)
    return str(path)


@tool(args_schema=DailySnapshotInput)
def save_daily_snapshot_tool(date: Optional[str] = None) -> str:
    """将指定日期（或今天）的财务分析快照保存为 JSON 文件，存储在 daily_snapshots 目录。返回保存路径。"""
    d = date or datetime.now().strftime("%Y-%m-%d")
    df = get_transactions_in_date_range(d, d, join_names=True, dollars=True)
    snapshot = insights.save_daily_snapshot(d, STATE["big_expense_threshold"], df)
    file_path = f"daily_snapshots/{d}.json"
    snapshot_dir = Path("daily_snapshots")
    snapshot_dir.mkdir(exist_ok=True)
    file_path = snapshot_dir / f"{d}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return f"已保存 {d} 快照: {file_path}"

# =============================
# LLM + Agent
# =============================
TOOLS = [
    update_time_window_tool,
    get_weekly_data_tool,
    get_week_rollups_tool,
    compare_to_last_week_tool,
    save_weekly_report_tool,
    save_daily_snapshot_tool,
]

SYSTEM_MESSAGE = (
    "你是一个中文个人财务分析助手。你可以更新分析时间区间、读取交易数据并生成深入的周报。\n\n"
    "输出规范：\n"
    "- 金额一律以美元显示，并在数字前添加 $，保留两位小数（如 $1,234.56）。\n"
    "- 生成周报时，请按以下流程：\n"
    "  1) 若时间不明确，调用 update_time_window_tool 设置为本周（周一至周日）；\n"
    "  2) 调用 get_week_rollups_tool 获取 Facts JSON；必要时调用 compare_to_last_week_tool 获取 WoW 对比；\n"
    "  3) 基于 Facts 先列出一个 JSON Facts 小节，然后再写 Markdown 报告（分章节：收入、支出、净现金流、Top 分类/Payees、WoW 对比、异常/大额、建议与预算）；\n"
    "  4) 如用户要求保存或归档，调用 save_weekly_report_tool(markdown)。\n"
    "  5) 请在周报中展示收入来源细分 (income_payee_distribution)\n"
    "我们默认今年是2025年, 所有分析、报告、默认时间都以此为准。\n"
)


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        agent_kwargs={"system_message": SYSTEM_MESSAGE},
    )

    print("\n💼 财务分析助手已启动。示例问题：")
    print("- 生成一份本周的详细周报并保存")
    print("- 将时间设为 2025-08-01 到 2025-08-07，然后给我详细报告")
    print("- 这周和上周相比，支出变化如何？")
    print("输入 exit/quit 退出\n")

    while True:
        query = input("你：").strip()
        if query.lower() in {"exit", "quit"}:
            print("再见"); break
        try:
            resp = agent.run(query)
            print("\n助手：", resp, "\n")
        except Exception as e:
            print(f"错误: {e}\n")


if __name__ == "__main__":
    main()
