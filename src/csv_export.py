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
        forecast_predict: 预测结果
        predict_dates: 预测日期
        arima_params: ARIMA参数
    
    返回:
        str: CSV文件路径
    """
    # 确保输出目录存在
    output_dir = Path(CSV_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成CSV数据
    csv_data = []
    for i, (date, value) in enumerate(zip(predict_dates, forecast_predict)):
        # 格式化日期为YYYYMMDD格式
        date_str = date.strftime(CSV_CONFIG['format']['date_format'])
        
        # 格式化数值，保留指定小数位数
        purchase_value = round(float(value), CSV_CONFIG['format']['decimal_places'])
        
        # 这里假设赎回金额为申购金额的某个比例（可以根据实际需求调整）
        # 或者可以设置为0，表示只预测申购金额
        redeem_value = 0.0  # 可以根据实际需求调整
        
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
    
    return str(csv_path)

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