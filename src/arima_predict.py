#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA预测功能模块

本模块用于ARIMA模型预测，支持缓存参数。
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib
from utils import adf_test
from utils.cache_manager import cache_manager
from utils.menu_control import show_confirm_dialog, show_three_way_dialog
from config import get_data_file_path, VISUALIZATION_CONFIG, get_output_path
from src.arima_param_search import get_or_search_best_arima_params

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def arima_predict():
    """
    ARIMA预测主流程
    """
    print("=" * 60)
    print("📊 ARIMA预测工具 - 支持缓存参数")
    print("=" * 60)
    try:
        ts_train, predict_dates, steps = load_and_prepare_data()
        print(f"✅ 数据加载完成，训练集长度: {len(ts_train)}")
        data_file_path = get_data_file_path()
        # 统一通过get_or_search_best_arima_params获取最优参数
        arima_params = get_or_search_best_arima_params(ts_train, data_file_path, verbose=True)
        if arima_params is None:
            print("❌ 预测已取消")
            return False
        forecast_predict, model_fit = perform_prediction(ts_train, predict_dates, steps, arima_params)
        output_path = create_visualization(ts_train, forecast_predict, arima_params)
        print(f"\n{'='*50}")
        print("📈 预测结果统计")
        print(f"{'='*50}")
        print(f"训练集均值: {ts_train.mean():.2f}")
        print(f"训练集标准差: {ts_train.std():.2f}")
        print(f"预测均值: {forecast_predict.mean():.2f}")
        print(f"预测标准差: {forecast_predict.std():.2f}")
        print(f"模型AIC: {model_fit.aic:.2f}")
        print(f"模型BIC: {model_fit.bic:.2f}")
        cache_manager.save_image_cache(
            data_file_path,
            'prediction',
            output_path,
            f"ARIMA{arima_params}模型预测申购金额图 (output/images/)"
        )
        print(f"\n{'='*60}")
        print("🎉 ARIMA预测完成！")
        print(f"{'='*60}")
        return True
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        return False

def load_and_prepare_data():
    file_path = get_data_file_path()
    df = pd.read_csv(file_path)
    if df['report_date'].dtype != 'O':
        df['report_date'] = df['report_date'].astype(str)
    df = df[df['report_date'] >= '20140301']
    trend = df.groupby('report_date')['total_purchase_amt'].sum().reset_index()
    trend = trend.sort_values('report_date')
    dates = pd.to_datetime(trend['report_date'], format='%Y%m%d')
    ts = pd.Series(trend['total_purchase_amt'].values, index=dates)
    ts_train = ts[(ts.index >= '2014-03-01') & (ts.index <= '2014-08-31')]
    predict_dates = pd.date_range('2014-09-01', '2014-12-31')
    steps = (predict_dates[0] - ts_train.index[-1]).days + len(predict_dates)
    return ts_train, predict_dates, steps

def get_arima_params_with_cache(data_file_path, ts_train):
    cache_manager.refresh_cache()
    cached_info = cache_manager.get_cached_params(data_file_path)
    if cached_info and isinstance(cached_info, dict) and 'best_params' in cached_info:
        print(f"\n{'='*60}")
        print("📋 发现缓存的ARIMA参数:")
        print(f"最优参数: ARIMA{cached_info['best_params']}")
        print(f"AIC值: {cached_info['best_aic']:.2f}")
        print(f"参数个数: {cached_info.get('total_params', 'N/A')} ({cached_info.get('param_ratio', 'N/A')}%)")
        print(f"缓存时间: {cached_info.get('timestamp', 'Unknown')}")
        print(f"{'='*60}")
        choice = show_three_way_dialog(
            "是否使用缓存的ARIMA参数进行预测？",
            ["✅ 使用缓存参数", "🔄 生成新参数", "❌ 取消预测"]
        )
        if choice == 0:
            print("✅ 使用缓存的ARIMA参数进行预测")
            return cached_info['best_params']
        elif choice == 1:
            print("🔄 将生成新的ARIMA参数...")
            return generate_new_params(ts_train)
        elif choice == 2:
            print("❌ 预测已取消")
            return None
        else:
            print("❌ 无效选择，预测已取消")
            return None
    else:
        print(f"\n{'='*60}")
        print("📭 未发现缓存的ARIMA参数")
        print("💡 建议：先生成最佳ARIMA参数，再进行预测")
        print(f"{'='*60}")
        choice = show_confirm_dialog(
            "是否现在生成新的最佳ARIMA参数并进行预测？",
            default_yes=True
        )
        if choice:
            print("🔄 将生成新的ARIMA参数...")
            return generate_new_params(ts_train)
        else:
            print("❌ 预测已取消")
            return None

def generate_new_params(ts_train):
    try:
        from utils.arima_grid_search import arima_grid_search
        print("🔍 开始ARIMA参数网格搜索...")
        adf_test(ts_train, title='2014年3月至8月申购金额训练集')
        data_length = len(ts_train)
        max_params = min(10, int(data_length * 0.05))
        best_params, best_model = arima_grid_search(
            ts=ts_train,
            p_range=range(0, 4),
            d_range=range(0, 3),
            q_range=range(0, 4),
            max_params=max_params,
            verbose=True
        )
        if best_params is not None:
            print(f"✅ 成功生成新的ARIMA参数: ARIMA{best_params}")
            return best_params
        else:
            print("❌ 参数搜索失败，使用默认参数 (2,1,4)")
            return (2, 1, 4)
    except Exception as e:
        print(f"❌ 参数生成失败: {e}")
        print("💡 使用默认参数 (2,1,4)")
        return (2, 1, 4)

def perform_prediction(ts_train, predict_dates, steps, arima_params):
    print(f"\n{'='*50}")
    print(f"🚀 开始ARIMA预测 (参数: ARIMA{arima_params})")
    print(f"{'='*50}")
    model = ARIMA(ts_train, order=arima_params)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_predict = forecast[-len(predict_dates):]
    print(f"✅ 预测完成，预测步数: {steps}")
    print(f"📊 预测区间: {predict_dates[0].strftime('%Y-%m-%d')} 至 {predict_dates[-1].strftime('%Y-%m-%d')}")
    return forecast_predict, model_fit

def create_visualization(ts_train, forecast_predict, arima_params):
    print(f"\n{'='*50}")
    print("🎨 生成预测结果图表...")
    print(f"{'='*50}")
    plt.figure(figsize=(14, 6))
    ts_train.plot(label='训练集历史申购金额', color='tab:blue', linewidth=2)
    forecast_predict.plot(label='预测申购金额', color='tab:orange', linewidth=2)
    plt.title(f'2014年9月至2014年12月申购金额预测（ARIMA{arima_params}）', fontsize=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('申购金额', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    all_dates = list(ts_train.index) + list(forecast_predict.index)
    all_dates_series = pd.Series(1, index=pd.to_datetime(all_dates)).sort_index()
    month_starts = all_dates_series.resample('MS').first().dropna().index
    xticks = month_starts
    xtick_labels = [f"{dt.year}年{dt.month}月" for dt in xticks]
    plt.xticks(xticks, xtick_labels, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13)
    plt.tight_layout()
    output_dir = VISUALIZATION_CONFIG['output_dir']
    output_path = get_output_path(os.path.join(output_dir, 'arima_purchase_201409_201412_forecast.png'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存: {output_path}")
    return output_path

if __name__ == "__main__":
    arima_predict() 