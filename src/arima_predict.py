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
        
        # 获取申购金额的最优参数
        purchase_params = get_or_search_best_arima_params(ts_train, data_file_path, verbose=True, series_type='purchase')
        if purchase_params is None:
            print("❌ 申购金额参数获取失败，预测已取消")
            return False
        
        # 获取赎回金额的最优参数
        print(f"\n{'='*50}")
        print("🔍 获取赎回金额最优参数...")
        print(f"{'='*50}")
        
        # 加载赎回金额数据
        df = pd.read_csv(data_file_path)
        if df['report_date'].dtype != 'O':
            df['report_date'] = df['report_date'].astype(str)
        df = df[df['report_date'] >= '20140301']
        redeem_trend = df.groupby('report_date')['total_redeem_amt'].sum().reset_index()
        redeem_trend = redeem_trend.sort_values('report_date')
        redeem_dates = pd.to_datetime(redeem_trend['report_date'], format='%Y%m%d')
        ts_redeem = pd.Series(redeem_trend['total_redeem_amt'].values, index=redeem_dates)
        ts_train_redeem = ts_redeem[(ts_redeem.index >= '2014-03-01') & (ts_redeem.index <= '2014-08-31')]
        
        redeem_params = get_or_search_best_arima_params(ts_train_redeem, data_file_path, verbose=True, series_type='redeem')
        if redeem_params is None:
            print("⚠️ 赎回金额参数获取失败，将使用申购金额的参数")
            redeem_params = purchase_params
        
        # 预测申购金额
        forecast_purchase, model_fit_purchase = perform_prediction(ts_train, predict_dates, steps, purchase_params)
        
        # 预测赎回金额（使用赎回金额的最优参数）
        print(f"\n{'='*50}")
        print("🔄 开始预测赎回金额...")
        print(f"{'='*50}")
        forecast_redeem, model_fit_redeem = perform_redeem_prediction_with_params(redeem_params, predict_dates, steps)
        
        # 创建包含申购和赎回的可视化
        output_path = create_visualization_with_redeem(ts_train, forecast_purchase, forecast_redeem, purchase_params, redeem_params)
        
        print(f"\n{'='*50}")
        print("📈 预测结果统计")
        print(f"{'='*50}")
        print(f"申购金额 - 训练集均值: {ts_train.mean():.2f}")
        print(f"申购金额 - 训练集标准差: {ts_train.std():.2f}")
        print(f"申购金额 - 预测均值: {forecast_purchase.mean():.2f}")
        print(f"申购金额 - 预测标准差: {forecast_purchase.std():.2f}")
        print(f"申购金额 - 模型AIC: {model_fit_purchase.aic:.2f}")
        print(f"申购金额 - 模型BIC: {model_fit_purchase.bic:.2f}")
        print(f"申购金额 - 使用参数: ARIMA{purchase_params}")
        
        if forecast_redeem is not None:
            print(f"赎回金额 - 预测均值: {forecast_redeem.mean():.2f}")
            print(f"赎回金额 - 预测标准差: {forecast_redeem.std():.2f}")
            if hasattr(model_fit_redeem, 'aic') and model_fit_redeem is not None:
                print(f"赎回金额 - 模型AIC: {model_fit_redeem.aic:.2f}")
                print(f"赎回金额 - 模型BIC: {model_fit_redeem.bic:.2f}")
            print(f"赎回金额 - 使用参数: ARIMA{redeem_params}")
        
        # 保存图片缓存信息
        cache_manager.save_image_cache(
            data_file_path,
            'prediction',
            output_path,
            f"ARIMA申购{purchase_params}_赎回{redeem_params}模型预测图 (output/images/)"
        )
        
        # 尝试打开生成的图片
        try:
            import subprocess
            import platform
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', output_path], check=True)
                print("💡 预测图生成完成，正在打开...")
            elif platform.system() == 'Windows':
                subprocess.run(['start', output_path], shell=True, check=True)
                print("💡 预测图生成完成，正在打开...")
            else:  # Linux
                subprocess.run(['xdg-open', output_path], check=True)
                print("💡 预测图生成完成，正在打开...")
        except subprocess.CalledProcessError as e:
            print(f"❌ 打开图片失败: {e}")
            print(f"💡 请手动打开文件: {output_path}")
        except Exception as e:
            print(f"⚠️ 打开图片时发生错误: {e}")
            print(f"💡 请手动打开文件: {output_path}")
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
        print("💡 建议：在进行ARIMA建模前，建议先使用'数据平稳性检验'功能检查数据平稳性")
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

def perform_redeem_prediction(arima_params, predict_dates, steps):
    """
    预测赎回金额
    
    参数:
        arima_params: ARIMA参数
        predict_dates: 预测日期
        steps: 预测步数
    
    返回:
        tuple: (forecast_redeem, model_fit_redeem)
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
        
        print(f"📊 赎回金额训练集长度: {len(ts_train_redeem)}")
        print(f"📊 赎回金额训练集均值: {ts_train_redeem.mean():.2f}")
        print(f"📊 赎回金额训练集标准差: {ts_train_redeem.std():.2f}")
        
        # 使用同样的ARIMA参数对赎回金额建模
        model_redeem = ARIMA(ts_train_redeem, order=arima_params)
        model_fit_redeem = model_redeem.fit()
        forecast_redeem = model_fit_redeem.forecast(steps=steps)
        forecast_redeem_predict = forecast_redeem[-len(predict_dates):]
        
        print(f"✅ 赎回金额预测完成，预测步数: {steps}")
        print(f"📊 赎回金额预测区间: {predict_dates[0].strftime('%Y-%m-%d')} 至 {predict_dates[-1].strftime('%Y-%m-%d')}")
        
        return forecast_redeem_predict, model_fit_redeem
        
    except Exception as e:
        print(f"❌ 赎回金额预测失败: {e}")
        print("💡 将使用比例估算方法...")
        
        # 如果预测失败，使用比例估算
        try:
            estimated_redeem = estimate_redeem_by_ratio(arima_params, predict_dates, steps)
            return estimated_redeem, None
        except Exception as e2:
            print(f"❌ 比例估算也失败: {e2}")
            return None, None

def perform_redeem_prediction_with_params(redeem_params, predict_dates, steps):
    """
    使用指定的ARIMA参数预测赎回金额
    
    参数:
        redeem_params: tuple - 赎回金额的ARIMA参数
        predict_dates: pd.DatetimeIndex - 预测日期
        steps: int - 预测步数
    
    返回:
        tuple: (forecast_redeem, model_fit_redeem)
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
        
        print(f"📊 赎回金额训练集长度: {len(ts_train_redeem)}")
        print(f"📊 赎回金额训练集均值: {ts_train_redeem.mean():.2f}")
        print(f"📊 赎回金额训练集标准差: {ts_train_redeem.std():.2f}")
        print(f"📊 使用ARIMA参数: {redeem_params}")
        
        # 使用指定的ARIMA参数对赎回金额建模
        model_redeem = ARIMA(ts_train_redeem, order=redeem_params)
        model_fit_redeem = model_redeem.fit()
        forecast_redeem = model_fit_redeem.forecast(steps=steps)
        forecast_redeem_predict = forecast_redeem[-len(predict_dates):]
        
        print(f"✅ 赎回金额预测完成，预测步数: {steps}")
        print(f"📊 赎回金额预测区间: {predict_dates[0].strftime('%Y-%m-%d')} 至 {predict_dates[-1].strftime('%Y-%m-%d')}")
        print(f"📊 赎回金额预测均值: {forecast_redeem_predict.mean():.2f}")
        print(f"📊 赎回金额预测标准差: {forecast_redeem_predict.std():.2f}")
        
        return forecast_redeem_predict, model_fit_redeem
        
    except Exception as e:
        print(f"❌ 赎回金额预测失败: {e}")
        print("💡 将使用比例估算方法...")
        
        # 如果预测失败，使用比例估算
        try:
            estimated_redeem = estimate_redeem_by_ratio(redeem_params, predict_dates, steps)
            return estimated_redeem, None
        except Exception as e2:
            print(f"❌ 比例估算也失败: {e2}")
            return None, None

def estimate_redeem_by_ratio(arima_params, predict_dates, steps):
    """
    使用历史赎回/申购比例估算赎回金额
    
    参数:
        arima_params: ARIMA参数
        predict_dates: 预测日期
        steps: 预测步数
    
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
        ts_train, _, _ = load_and_prepare_data()
        forecast_purchase, _ = perform_prediction(ts_train, predict_dates, steps, arima_params)
        
        # 根据比例估算赎回金额
        estimated_redeem = forecast_purchase * redeem_ratio
        
        print(f"✅ 使用历史比例估算赎回金额完成")
        return estimated_redeem
        
    except Exception as e:
        print(f"❌ 比例估算失败: {e}")
        print("💡 使用默认比例0.1...")
        
        # 最后的备选方案：使用默认比例
        ts_train, _, _ = load_and_prepare_data()
        forecast_purchase, _ = perform_prediction(ts_train, predict_dates, steps, arima_params)
        return forecast_purchase * 0.1

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

def create_visualization_with_redeem(ts_train, forecast_purchase, forecast_redeem, purchase_params, redeem_params):
    print(f"\n{'='*50}")
    print("🎨 生成预测结果图表...")
    print(f"{'='*50}")
    
    # 加载赎回金额历史数据
    try:
        file_path = get_data_file_path()
        df = pd.read_csv(file_path)
        
        if df['report_date'].dtype != 'O':
            df['report_date'] = df['report_date'].astype(str)
        
        df = df[df['report_date'] >= '20140301']
        trend = df.groupby('report_date')['total_redeem_amt'].sum().reset_index()
        trend = trend.sort_values('report_date')
        dates = pd.to_datetime(trend['report_date'], format='%Y%m%d')
        ts_redeem = pd.Series(trend['total_redeem_amt'].values, index=dates)
        ts_train_redeem = ts_redeem[(ts_redeem.index >= '2014-03-01') & (ts_redeem.index <= '2014-08-31')]
    except Exception as e:
        print(f"⚠️ 加载赎回金额历史数据失败: {e}")
        ts_train_redeem = None
    
    plt.figure(figsize=(14, 10))  # 增加图表高度
    
    # 绘制申购金额预测
    plt.subplot(2, 1, 1)
    ts_train.plot(label='训练集历史申购金额', color='tab:blue', linewidth=2)
    forecast_purchase.plot(label='预测申购金额', color='tab:orange', linewidth=2)
    plt.title(f'2014年9月至2014年12月申购金额预测（ARIMA{purchase_params}）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('申购金额', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    all_dates_purchase = list(ts_train.index) + list(forecast_purchase.index)
    all_dates_series_purchase = pd.Series(1, index=pd.to_datetime(all_dates_purchase)).sort_index()
    month_starts_purchase = all_dates_series_purchase.resample('MS').first().dropna().index
    xticks_purchase = month_starts_purchase
    xtick_labels_purchase = [f"{dt.year}年{dt.month}月" for dt in xticks_purchase]
    plt.xticks(xticks_purchase, xtick_labels_purchase, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=11)
    
    # 绘制赎回金额预测
    plt.subplot(2, 1, 2)
    if forecast_redeem is not None:
        if ts_train_redeem is not None:
            ts_train_redeem.plot(label='训练集历史赎回金额', color='tab:green', linewidth=2)
        forecast_redeem.plot(label='预测赎回金额', color='tab:red', linewidth=2)
        plt.title(f'2014年9月至2014年12月赎回金额预测（ARIMA{redeem_params}）', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('赎回金额', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        all_dates_redeem = list(forecast_redeem.index)
        if ts_train_redeem is not None:
            all_dates_redeem = list(ts_train_redeem.index) + all_dates_redeem
        all_dates_series_redeem = pd.Series(1, index=pd.to_datetime(all_dates_redeem)).sort_index()
        month_starts_redeem = all_dates_series_redeem.resample('MS').first().dropna().index
        xticks_redeem = month_starts_redeem
        xtick_labels_redeem = [f"{dt.year}年{dt.month}月" for dt in xticks_redeem]
        plt.xticks(xticks_redeem, xtick_labels_redeem, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=11)
    else:
        plt.text(0.5, 0.5, '赎回金额预测数据不可用', horizontalalignment='center', 
                verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('赎回金额预测', fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    output_dir = VISUALIZATION_CONFIG['output_dir']
    output_path = get_output_path(os.path.join(output_dir, 'arima_purchase_redeem_201409_201412_forecast.png'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存: {output_path}")
    return output_path

if __name__ == "__main__":
    arima_predict() 