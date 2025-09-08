from utils.db import load_flattened_transactions

class FinanceAgent:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self.df = load_flattened_transactions(start_date, end_date)

    def summarize_expenses(self):
        summary = self.df[self.df["amount"] < 0].groupby("category_name")["amount"].sum().sort_values()
        return summary

    def summarize_income(self):
        summary = self.df[self.df["amount"] > 0].groupby("payee")["amount"].sum().sort_values(ascending=False)
        return summary

    def detect_income(self):
        income_df = self.df[self.df["amount"] > 0].copy()
        return income_df

    def preview_data(self, n=5):
        return self.df.head(n)