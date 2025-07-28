#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金流入流出趋势图功能模块

本模块用于绘制用户申购和赎回金额的时间趋势图。
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISUALIZATION_CONFIG, get_output_path

def plot_trend():
    """
    绘制资金流入流出趋势图
    """
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

    # 保存图片到output/images文件夹
    output_dir = VISUALIZATION_CONFIG['output_dir']
    output_path = get_output_path(os.path.join(output_dir, 'purchase_redeem_trend.png'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ 趋势图已保存: {output_path}")

if __name__ == "__main__":
    plot_trend() 