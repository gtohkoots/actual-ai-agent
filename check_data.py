from utils.db import get_transactions_in_date_range
# 加载指定时间段的数据
df = get_transactions_in_date_range("2025-08-30", "2025-08-30", debug=True)

# 打印前几行
print("全部交易数据前几行：")
print(df.head())