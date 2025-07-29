#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA预测脚本 - 支持缓存参数使用

本脚本用于ARIMA模型预测，具有以下功能：
1. 自动检查并使用缓存的ARIMA参数
2. 如果没有缓存，提示用户是否生成新的最佳参数
3. 使用最佳参数进行预测
4. 生成预测结果图表

作者: AI Assistant
创建时间: 2024
版本: 2.0
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
# 添加上级目录到Python路径，以便导入utils模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib
from utils import adf_test
from utils.cache_manager import cache_manager
from utils.menu_control import show_confirm_dialog, show_three_way_dialog
from config import get_data_file_path

# 设置matplotlib中文字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """
    加载和准备数据
    
    返回：
        ts_train: pd.Series - 训练集时间序列
        predict_dates: pd.DatetimeIndex - 预测日期范围
        steps: int - 预测步数
    """
    # 读取数据
    file_path = get_data_file_path()
    df = pd.read_csv(file_path)
    
    # 确保report_date为字符串
    if df['report_date'].dtype != 'O':
        df['report_date'] = df['report_date'].astype(str)
    
    # 只保留2014年3月及以后的数据
    df = df[df['report_date'] >= '20140301']
    
    # 按日期汇总申购金额，并按时间排序
    trend = df.groupby('report_date')['total_purchase_amt'].sum().reset_index()
    trend = trend.sort_values('report_date')
    
    # 构造时间序列索引
    dates = pd.to_datetime(trend['report_date'], format='%Y%m%d')
    ts = pd.Series(trend['total_purchase_amt'].values, index=dates)
    
    # 训练集：2014年3月1日~2014年8月31日
    ts_train = ts[(ts.index >= '2014-03-01') & (ts.index <= '2014-08-31')]
    
    # 预测区间：2014年9月1日~2014年12月31日
    predict_dates = pd.date_range('2014-09-01', '2014-12-31')
    steps = (predict_dates[0] - ts_train.index[-1]).days + len(predict_dates)
    
    return ts_train, predict_dates, steps

def get_arima_params_with_cache(data_file_path, ts_train):
    """
    获取ARIMA参数，优先使用缓存
    
    参数：
        data_file_path: str - 数据文件路径
        ts_train: pd.Series - 训练集时间序列
    
    返回：
        tuple: (p, d, q) - ARIMA参数
    """
    # 刷新缓存，确保获取最新信息
    cache_manager.refresh_cache()
    
    # 检查是否有缓存的ARIMA参数
    cached_info = cache_manager.get_cached_params(data_file_path)
    
    if cached_info and isinstance(cached_info, dict) and 'best_params' in cached_info:
        # 有缓存参数，询问是否使用
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
        
        if choice == 0:  # 使用缓存参数
            print("✅ 使用缓存的ARIMA参数进行预测")
            return cached_info['best_params']
        elif choice == 1:  # 生成新参数
            print("🔄 将生成新的ARIMA参数...")
            return generate_new_params(ts_train)
        elif choice == 2:  # 取消
            print("❌ 预测已取消")
            return None
        else:
            print("❌ 无效选择，预测已取消")
            return None
    else:
        # 没有缓存参数，询问是否生成
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
    """
    生成新的ARIMA参数
    
    参数：
        ts_train: pd.Series - 训练集时间序列
    
    返回：
        tuple: (p, d, q) - ARIMA参数
    """
    try:
        # 导入网格搜索函数
        from utils.arima_grid_search import arima_grid_search
        
        print("🔍 开始ARIMA参数网格搜索...")
        print("💡 建议：在进行ARIMA建模前，建议先使用'数据平稳性检验'功能检查数据平稳性")
        
        # 设置参数搜索范围
        data_length = len(ts_train)
        max_params = min(10, int(data_length * 0.05))  # 最多10个参数或数据长度的5%
        
        # 进行网格搜索
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
    """
    执行ARIMA预测
    
    参数：
        ts_train: pd.Series - 训练集时间序列
        predict_dates: pd.DatetimeIndex - 预测日期范围
        steps: int - 预测步数
        arima_params: tuple - ARIMA参数 (p, d, q)
    
    返回：
        tuple: (forecast_predict, model_fit) - 预测结果和拟合模型
    """
    print(f"\n{'='*50}")
    print(f"🚀 开始ARIMA预测 (参数: ARIMA{arima_params})")
    print(f"{'='*50}")
    
    # ARIMA建模与预测
    model = ARIMA(ts_train, order=arima_params)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_predict = forecast[-len(predict_dates):]
    
    print(f"✅ 预测完成，预测步数: {steps}")
    print(f"📊 预测区间: {predict_dates[0].strftime('%Y-%m-%d')} 至 {predict_dates[-1].strftime('%Y-%m-%d')}")
    
    return forecast_predict, model_fit

def create_visualization(ts_train, forecast_predict, arima_params):
    """
    创建预测结果可视化
    
    参数：
        ts_train: pd.Series - 训练集时间序列
        forecast_predict: pd.Series - 预测结果
        arima_params: tuple - ARIMA参数
    """
    print(f"\n{'='*50}")
    print("🎨 生成预测结果图表...")
    print(f"{'='*50}")
    
    # 可视化：训练集+预测区间
    plt.figure(figsize=(14, 6))
    ts_train.plot(label='训练集历史申购金额', color='tab:blue', linewidth=2)
    forecast_predict.plot(label='预测申购金额', color='tab:orange', linewidth=2)
    
    plt.title(f'2014年9月至2014年12月申购金额预测（ARIMA{arima_params}）', fontsize=18)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('申购金额', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 只取每月第一个数据点做中文横坐标
    all_dates = list(ts_train.index) + list(forecast_predict.index)
    all_dates_series = pd.Series(1, index=pd.to_datetime(all_dates)).sort_index()
    month_starts = all_dates_series.resample('MS').first().dropna().index
    xticks = month_starts
    xtick_labels = [f"{dt.year}年{dt.month}月" for dt in xticks]
    plt.xticks(xticks, xtick_labels, rotation=45, fontsize=12)
    
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13)
    plt.tight_layout()
    
    # 保存图片
    output_dir = 'output/images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'arima_purchase_201409_201412_forecast.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图表已保存: {output_path}")
    
    return output_path

def main():
    """
    主函数
    """
    print("=" * 60)
    print("📊 ARIMA预测工具 - 支持缓存参数")
    print("=" * 60)
    
    try:
        # 1. 加载和准备数据
        print("📂 加载数据...")
        ts_train, predict_dates, steps = load_and_prepare_data()
        print(f"✅ 数据加载完成，训练集长度: {len(ts_train)}")
        
        # 2. 获取ARIMA参数（优先使用缓存）
        data_file_path = get_data_file_path()
        arima_params = get_arima_params_with_cache(data_file_path, ts_train)
        
        if arima_params is None:
            print("❌ 预测已取消")
            return False
        
        # 3. 执行预测
        forecast_predict, model_fit = perform_prediction(ts_train, predict_dates, steps, arima_params)
        
        # 4. 创建可视化
        output_path = create_visualization(ts_train, forecast_predict, arima_params)
        
        # 5. 输出预测统计信息
        print(f"\n{'='*50}")
        print("📈 预测结果统计")
        print(f"{'='*50}")
        print(f"训练集均值: {ts_train.mean():.2f}")
        print(f"训练集标准差: {ts_train.std():.2f}")
        print(f"预测均值: {forecast_predict.mean():.2f}")
        print(f"预测标准差: {forecast_predict.std():.2f}")
        print(f"模型AIC: {model_fit.aic:.2f}")
        print(f"模型BIC: {model_fit.bic:.2f}")
        
        # 6. 保存图片缓存
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

if __name__ == "__main__":
    main() 