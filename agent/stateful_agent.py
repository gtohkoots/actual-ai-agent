from agent.finance_agent import FinanceAgent
from langchain.agents import tool


class StatefulFinanceAgent:
    def __init__(self):
        self.start_date = "2025-08-01"
        self.end_date = "2025-08-31"
        self.agent = FinanceAgent(self.start_date, self.end_date)

    def update_time_window(self, time_range: str) -> str:
        """更新分析时间范围。输入格式应为 'YYYY-MM-DD to YYYY-MM-DD'。"""
        try:
            start_date, end_date = [s.strip() for s in time_range.split("to")]
            self.start_date = start_date
            self.end_date = end_date
            self.agent = FinanceAgent(start_date, end_date)
            return f"时间区间更新为 {start_date} 到 {end_date}"
        except Exception as e:
            return f"解析时间失败：请使用 'YYYY-MM-DD to YYYY-MM-DD' 格式。错误：{e}"
        
    def summarize_expenses(self) -> str:
        """输出当前时间区间内的支出分类统计"""
        summary = self.agent.summarize_expenses()
        return summary.to_string()

    def summarize_income(self) -> str:
        """输出当前时间区间内的收入来源汇总"""
        summary = self.agent.summarize_income()
        return summary.to_string()

    def detect_income(self) -> str:
        """识别当前时间区间内可能的收入交易记录"""
        df = self.agent.detect_income()
        return df.to_string(index=False)

    def preview_data(self) -> str:
        """预览当前时间区间内的交易数据"""
        df = self.agent.preview_data()
        return df.to_string(index=False)