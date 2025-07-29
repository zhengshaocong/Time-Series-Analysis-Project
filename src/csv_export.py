#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV导出功能模块

本模块用于根据ARIMA预测结果生成CSV文件，包含预测的申购和赎回金额。
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from utils.cache_manager import cache_manager
from utils.menu_control import show_confirm_dialog, show_three_way_dialog
from config import (
    get_data_file_path, CSV_CONFIG, get_output_path, 
    VISUALIZATION_CONFIG, ARIMA_CONFIG
)
from src.arima_param_search import get_or_search_best_arima_params
from src.arima_predict import load_and_prepare_data, perform_prediction

def csv_export():
    """
    CSV导出主流程
    """
    print("=" * 60)
    print("📊 CSV导出工具 - 根据ARIMA预测生成CSV文件")
    print("=" * 60)
    
    try:
        # 检查是否有预测缓存
        data_file_path = get_data_file_path()
        cache_manager.refresh_cache()
        
        # 检查是否有预测图片缓存
        image_cache = cache_manager.get_image_cache(data_file_path, 'prediction')
        has_prediction_cache = image_cache and image_cache.get('exists', False)
        
        if has_prediction_cache:
            print(f"\n{'='*60}")
            print("📋 发现预测结果缓存:")
            print(f"预测图路径: {image_cache['path']}")
            print(f"生成时间: {image_cache['timestamp']}")
            print(f"描述: {image_cache['description']}")
            print(f"{'='*60}")
            
            choice = show_three_way_dialog(
                "是否基于现有预测结果生成CSV？", 
                ["✅ 使用现有预测", "🔄 重新预测", "❌ 取消"]
            )
            
            if choice == 0:
                print("✅ 使用现有预测结果生成CSV...")
                return generate_csv_from_cache(data_file_path)
            elif choice == 1:
                print("🔄 将重新进行预测...")
                return generate_csv_with_new_prediction(data_file_path)
            elif choice == 2:
                print("❌ 操作已取消")
                return False
        else:
            print("📭 未发现预测结果缓存")
            choice = show_confirm_dialog(
                "是否现在进行ARIMA预测并生成CSV文件？",
                default_yes=True
            )
            if choice:
                return generate_csv_with_new_prediction(data_file_path)
            else:
                print("❌ 操作已取消")
                return False
                
    except Exception as e:
        print(f"❌ CSV导出过程中发生错误: {e}")
        return False

def generate_csv_from_cache(data_file_path):
    """
    基于缓存生成CSV文件
    """
    try:
        # 这里需要从缓存中恢复预测结果
        # 由于缓存只保存了图片信息，我们需要重新进行预测
        print("⚠️ 缓存中只有图片信息，需要重新进行预测...")
        return generate_csv_with_new_prediction(data_file_path)
    except Exception as e:
        print(f"❌ 从缓存生成CSV失败: {e}")
        return False

def generate_csv_with_new_prediction(data_file_path):
    """
    重新预测并生成CSV文件
    """
    try:
        # 加载和准备数据
        ts_train, predict_dates, steps = load_and_prepare_data()
        print(f"✅ 数据加载完成，训练集长度: {len(ts_train)}")
        
        # 获取ARIMA参数
        arima_params = get_or_search_best_arima_params(ts_train, data_file_path, verbose=True)
        if arima_params is None:
            print("❌ 预测已取消")
            return False
        
        # 执行预测
        forecast_predict, model_fit = perform_prediction(ts_train, predict_dates, steps, arima_params)
        
        # 生成CSV文件
        csv_path = generate_csv_file(forecast_predict, predict_dates, arima_params)
        
        print(f"\n{'='*50}")
        print("📈 CSV生成结果统计")
        print(f"{'='*50}")
        print(f"预测数据点数量: {len(forecast_predict)}")
        print(f"预测日期范围: {predict_dates[0].strftime('%Y-%m-%d')} 到 {predict_dates[-1].strftime('%Y-%m-%d')}")
        print(f"预测均值: {forecast_predict.mean():.2f}")
        print(f"预测标准差: {forecast_predict.std():.2f}")
        print(f"CSV文件路径: {csv_path}")
        print(f"模型AIC: {model_fit.aic:.2f}")
        
        print(f"\n{'='*60}")
        print("🎉 CSV文件生成完成！")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"❌ 预测并生成CSV失败: {e}")
        return False

def generate_csv_file(forecast_predict, predict_dates, arima_params):
    """
    生成CSV文件
    
    参数:
        forecast_predict: 申购金额预测结果
        predict_dates: 预测日期
        arima_params: ARIMA参数
    
    返回:
        str: CSV文件路径
    """
    # 确保输出目录存在
    output_dir = Path(CSV_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 预测赎回金额
    print(f"\n📊 开始预测赎回金额...")
    forecast_redeem = predict_redeem_amount(arima_params)
    
    # 生成CSV数据
    csv_data = []
    for i, (date, purchase_value, redeem_value) in enumerate(zip(predict_dates, forecast_predict, forecast_redeem)):
        # 格式化日期为YYYYMMDD格式
        date_str = date.strftime(CSV_CONFIG['format']['date_format'])
        
        # 格式化数值，保留指定小数位数
        purchase_value = round(float(purchase_value), CSV_CONFIG['format']['decimal_places'])
        redeem_value = round(float(redeem_value), CSV_CONFIG['format']['decimal_places'])
        
        csv_data.append({
            'report_date': date_str,
            'purchase': purchase_value,
            'redeem': redeem_value
        })
    
    # 创建DataFrame
    df = pd.DataFrame(csv_data)
    
    # 生成文件路径
    filename = CSV_CONFIG['files']['prediction']['filename']
    csv_path = output_dir / filename
    
    # 保存CSV文件
    df.to_csv(
        csv_path, 
        index=False, 
        encoding=CSV_CONFIG['format']['encoding']
    )
    
    print(f"✅ CSV文件已保存: {csv_path}")
    print(f"📊 数据行数: {len(df)}")
    print(f"📋 列名: {', '.join(df.columns.tolist())}")
    print(f"📈 申购金额统计: 均值={df['purchase'].mean():.2f}, 标准差={df['purchase'].std():.2f}")
    print(f"📉 赎回金额统计: 均值={df['redeem'].mean():.2f}, 标准差={df['redeem'].std():.2f}")
    
    return str(csv_path)

def predict_redeem_amount(arima_params):
    """
    预测赎回金额
    
    参数:
        arima_params: ARIMA参数 (p, d, q)
    
    返回:
        pd.Series: 赎回金额预测结果
    """
    try:
        # 加载赎回金额数据
        file_path = get_data_file_path()
        df = pd.read_csv(file_path)
        
        # 确保report_date为字符串
        if df['report_date'].dtype != 'O':
            df['report_date'] = df['report_date'].astype(str)
        
        # 只保留2014年3月及以后的数据
        df = df[df['report_date'] >= '20140301']
        
        # 按日期汇总赎回金额，并按时间排序
        trend = df.groupby('report_date')['total_redeem_amt'].sum().reset_index()
        trend = trend.sort_values('report_date')
        
        # 构造时间序列索引
        dates = pd.to_datetime(trend['report_date'], format='%Y%m%d')
        ts_redeem = pd.Series(trend['total_redeem_amt'].values, index=dates)
        
        # 训练集：2014年3月1日~2014年8月31日
        ts_train_redeem = ts_redeem[(ts_redeem.index >= '2014-03-01') & (ts_redeem.index <= '2014-08-31')]
        
        # 预测区间：2014年9月1日~2014年12月31日
        predict_dates = pd.date_range('2014-09-01', '2014-12-31')
        steps = (predict_dates[0] - ts_train_redeem.index[-1]).days + len(predict_dates)
        
        # 使用同样的ARIMA参数对赎回金额建模
        from statsmodels.tsa.arima.model import ARIMA
        model_redeem = ARIMA(ts_train_redeem, order=arima_params)
        model_fit_redeem = model_redeem.fit()
        forecast_redeem = model_fit_redeem.forecast(steps=steps)
        forecast_redeem_predict = forecast_redeem[-len(predict_dates):]
        
        print(f"✅ 赎回金额预测完成，预测步数: {steps}")
        print(f"📊 赎回金额预测区间: {predict_dates[0].strftime('%Y-%m-%d')} 至 {predict_dates[-1].strftime('%Y-%m-%d')}")
        
        return forecast_redeem_predict
        
    except Exception as e:
        print(f"❌ 赎回金额预测失败: {e}")
        print("💡 使用历史平均比例估算赎回金额...")
        
        # 如果预测失败，使用历史数据的赎回/申购比例来估算
        return estimate_redeem_by_ratio(arima_params)

def estimate_redeem_by_ratio(arima_params):
    """
    使用历史赎回/申购比例估算赎回金额
    
    参数:
        arima_params: ARIMA参数
    
    返回:
        pd.Series: 估算的赎回金额
    """
    try:
        # 加载历史数据计算比例
        file_path = get_data_file_path()
        df = pd.read_csv(file_path)
        
        if df['report_date'].dtype != 'O':
            df['report_date'] = df['report_date'].astype(str)
        
        # 只保留2014年3月及以后的数据
        df = df[df['report_date'] >= '20140301']
        
        # 按日期汇总
        trend = df.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()
        trend = trend.sort_values('report_date')
        
        # 计算历史赎回/申购比例
        purchase_total = trend['total_purchase_amt'].sum()
        redeem_total = trend['total_redeem_amt'].sum()
        redeem_ratio = redeem_total / purchase_total if purchase_total > 0 else 0.1
        
        print(f"📊 历史赎回/申购比例: {redeem_ratio:.2%}")
        
        # 获取申购金额预测结果
        ts_train, predict_dates, steps = load_and_prepare_data()
        forecast_predict, _ = perform_prediction(ts_train, predict_dates, steps, arima_params)
        
        # 根据比例估算赎回金额
        estimated_redeem = forecast_predict * redeem_ratio
        
        print(f"✅ 使用历史比例估算赎回金额完成")
        return estimated_redeem
        
    except Exception as e:
        print(f"❌ 比例估算失败: {e}")
        print("💡 使用默认比例0.1...")
        
        # 最后的备选方案：使用默认比例
        ts_train, predict_dates, steps = load_and_prepare_data()
        forecast_predict, _ = perform_prediction(ts_train, predict_dates, steps, arima_params)
        return forecast_predict * 0.1

def handle_csv_export_with_cache():
    """
    带缓存处理的CSV导出函数
    """
    data_file_path = get_data_file_path()
    cache_manager.refresh_cache()
    
    # 检查CSV文件缓存
    csv_cache = cache_manager.get_csv_cache(data_file_path)
    if csv_cache and csv_cache.get('exists'):
        print(f"\n{'='*60}")
        print("📋 发现CSV文件缓存:")
        print(f"文件路径: {csv_cache['path']}")
        print(f"生成时间: {csv_cache['timestamp']}")
        print(f"描述: {csv_cache['description']}")
        print(f"{'='*60}")
        
        choice = show_three_way_dialog(
            "是否使用缓存的CSV文件？", 
            ["✅ 查看文件", "🔄 重新生成", "❌ 取消"]
        )
        
        if choice == 0:
            print("✅ 查看缓存文件")
            _open_csv_file(csv_cache['path'])
            return True
        elif choice == 2:
            print("❌ 操作已取消")
            return False
        elif choice == 1:
            print("🔄 将重新生成CSV文件...")
    
    # 重新生成CSV
    result = csv_export()
    if result:
        # 保存CSV缓存
        output_path = get_output_path(os.path.join(CSV_CONFIG['output_dir'], CSV_CONFIG['files']['prediction']['filename']))
        cache_manager.save_csv_cache(
            data_file_path, 
            'prediction', 
            output_path, 
            f"ARIMA预测结果CSV文件 (output/data/)"
        )
    
    return result

def _open_csv_file(csv_path):
    """
    打开CSV文件
    """
    abs_csv_path = Path(csv_path)
    if not abs_csv_path.is_absolute():
        abs_csv_path = Path(os.getcwd()) / abs_csv_path
    
    try:
        if os.name == 'nt':  # Windows
            os.startfile(str(abs_csv_path))
        elif sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.run(['open', str(abs_csv_path)], check=True)
        else:  # Linux
            import subprocess
            subprocess.run(['xdg-open', str(abs_csv_path)], check=True)
        print(f"✅ 已在默认应用中打开CSV文件: {abs_csv_path}")
    except Exception as e:
        print(f"❌ 打开CSV文件失败: {e}")
        print(f"💡 请手动打开文件: {abs_csv_path}")

if __name__ == '__main__':
    csv_export() 