

# 生成自定义横坐标标签，每月第一个日期显示YYYYMM，其余为空

import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据
file_path = 'data/user_balance_table.csv'
df = pd.read_csv(file_path)

# 确保report_date为字符串并按时间排序
if df['report_date'].dtype != 'O':
    df['report_date'] = df['report_date'].astype(str)
df = df.sort_values('report_date')

# 按日期汇总申购和赎回金额
trend = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 生成自定义横坐标标签，每月第一个日期显示YYYYMM，其余为空
dates = trend['report_date'].tolist()
xtick_labels = []
prev_month = None
for date in dates:
    yyyymm = date[:6]
    if prev_month != yyyymm:
        xtick_labels.append(yyyymm)
        prev_month = yyyymm
    else:
        xtick_labels.append('')  # 其余日期不显示

plt.figure(figsize=(60, 6))
# 绘制申购和赎回趋势曲线
plt.plot(dates, trend['total_purchase_amt'], label='Total Purchase Amount', marker='o')
plt.plot(dates, trend['total_redeem_amt'], label='Total Redeem Amount', marker='o')
plt.xlabel('Report Date')
plt.ylabel('Amount')
plt.title('Purchase and Redeem Trend by Report Date')
plt.xticks(range(len(dates)), xtick_labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.xlim(-0.5, len(dates) - 0.5)

# 保存图片到images文件夹
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'purchase_redeem_trend.png'))
plt.close() 