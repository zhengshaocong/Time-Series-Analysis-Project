#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA参数搜索功能模块

本模块用于网格搜索最优ARIMA(p,d,q)参数。
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import arima_grid_search
from utils.cache_manager import cache_manager
from config import get_data_file_path, ARIMA_CONFIG, get_output_path

def get_or_search_best_arima_params(ts_train, data_file_path=None, verbose=True, series_type='purchase'):
    """
    获取最优ARIMA参数（优先缓存，无则搜索并缓存）
    
    参数:
        ts_train: pd.Series - 训练集时间序列
        data_file_path: str - 数据文件路径
        verbose: bool - 是否详细输出
        series_type: str - 序列类型 ('purchase' 或 'redeem')
    
    返回:
        tuple: (p, d, q) 或 None
    """
    if data_file_path is None:
        data_file_path = get_data_file_path()
    
    cache_manager.refresh_cache()
    cached_info = cache_manager.get_cached_params(data_file_path, series_type)
    
    if cached_info and isinstance(cached_info, dict) and 'best_params' in cached_info:
        if verbose:
            print(f"✅ 读取{series_type}缓存最优参数: ARIMA{cached_info['best_params']}")
        return cached_info['best_params']
    
    # 无缓存，执行参数搜索
    if verbose:
        print(f"⚠️ 未找到{series_type}的ARIMA参数缓存，自动执行参数搜索...")
    
    best_params = _search_and_cache_arima_params(ts_train, data_file_path, verbose=verbose, series_type=series_type)
    return best_params

def _search_and_cache_arima_params(ts_train, data_file_path, verbose=True, series_type='purchase'):
    """
    搜索并缓存ARIMA参数
    
    参数:
        ts_train: pd.Series - 训练集时间序列
        data_file_path: str - 数据文件路径
        verbose: bool - 是否详细输出
        series_type: str - 序列类型 ('purchase' 或 'redeem')
    
    返回:
        tuple: (p, d, q) 或 None
    """
    data_length = len(ts_train)
    max_params = min(ARIMA_CONFIG['param_limits']['max_params'], int(data_length * ARIMA_CONFIG['param_limits']['param_ratio']))
    p_range = range(*ARIMA_CONFIG['param_ranges']['p_range'])
    d_range = range(*ARIMA_CONFIG['param_ranges']['d_range'])
    q_range = range(*ARIMA_CONFIG['param_ranges']['q_range'])
    
    best_params, best_model = arima_grid_search(ts_train, p_range, d_range, q_range, max_params=max_params, verbose=verbose)
    
    if best_params:
        total_params = best_params[0] + best_params[2] + 1
        if best_model is not None:
            best_aic = best_model.aic
            cache_manager.save_params(
                data_file_path, best_params, best_aic, total_params, data_length, series_type
            )
            if verbose:
                print(f"✅ 已缓存{series_type}最优参数: ARIMA{best_params}")
        else:
            if verbose:
                print(f"⚠️  {series_type}模型拟合失败，无法保存到缓存")
    else:
        if verbose:
            print(f"❌ 未找到{series_type}有效的ARIMA参数组合")
    
    return best_params

def search_both_purchase_and_redeem_params():
    """
    同时搜索申购和赎回金额的最优ARIMA参数
    """
    print("=" * 60)
    print("🔍 同时搜索申购和赎回金额的最优ARIMA参数")
    print("=" * 60)
    
    file_path = get_data_file_path()
    df = pd.read_csv(file_path)
    
    if df['report_date'].dtype != 'O':
        df['report_date'] = df['report_date'].astype(str)
    
    df = df[df['report_date'] >= '20140301']
    
    # 准备申购金额数据
    print("\n📊 准备申购金额数据...")
    purchase_trend = df.groupby('report_date')['total_purchase_amt'].sum().reset_index()
    purchase_trend = purchase_trend.sort_values('report_date')
    purchase_dates = pd.to_datetime(purchase_trend['report_date'], format='%Y%m%d')
    ts_purchase = pd.Series(purchase_trend['total_purchase_amt'].values, index=purchase_dates)
    ts_purchase = ts_purchase.asfreq('D')
    ts_train_purchase = ts_purchase[(ts_purchase.index >= '2014-03-01') & (ts_purchase.index <= '2014-08-31')]
    
    print(f"✅ 申购金额训练集长度: {len(ts_train_purchase)}")
    print(f"📊 申购金额均值: {ts_train_purchase.mean():.2f}")
    print(f"📊 申购金额标准差: {ts_train_purchase.std():.2f}")
    
    # 准备赎回金额数据
    print("\n📊 准备赎回金额数据...")
    redeem_trend = df.groupby('report_date')['total_redeem_amt'].sum().reset_index()
    redeem_trend = redeem_trend.sort_values('report_date')
    redeem_dates = pd.to_datetime(redeem_trend['report_date'], format='%Y%m%d')
    ts_redeem = pd.Series(redeem_trend['total_redeem_amt'].values, index=redeem_dates)
    ts_redeem = ts_redeem.asfreq('D')
    ts_train_redeem = ts_redeem[(ts_redeem.index >= '2014-03-01') & (ts_redeem.index <= '2014-08-31')]
    
    print(f"✅ 赎回金额训练集长度: {len(ts_train_redeem)}")
    print(f"📊 赎回金额均值: {ts_train_redeem.mean():.2f}")
    print(f"📊 赎回金额标准差: {ts_train_redeem.std():.2f}")
    
    # 搜索申购金额最优参数
    print(f"\n{'='*50}")
    print("🔍 搜索申购金额最优ARIMA参数...")
    print(f"{'='*50}")
    purchase_params = get_or_search_best_arima_params(ts_train_purchase, file_path, verbose=True, series_type='purchase')
    
    # 搜索赎回金额最优参数
    print(f"\n{'='*50}")
    print("🔍 搜索赎回金额最优ARIMA参数...")
    print(f"{'='*50}")
    redeem_params = get_or_search_best_arima_params(ts_train_redeem, file_path, verbose=True, series_type='redeem')
    
    # 输出总结
    print(f"\n{'='*60}")
    print("📋 参数搜索总结")
    print(f"{'='*60}")
    print(f"申购金额最优参数: ARIMA{purchase_params}")
    print(f"赎回金额最优参数: ARIMA{redeem_params}")
    
    if purchase_params and redeem_params:
        print(f"\n💡 参数对比分析:")
        print(f"   申购金额 d={purchase_params[1]}, 赎回金额 d={redeem_params[1]}")
        if purchase_params[1] != redeem_params[1]:
            print(f"   ⚠️  差分次数不同，说明两个序列的平稳性特征不同")
        else:
            print(f"   ✅ 差分次数相同，两个序列的平稳性特征相似")
    
    return purchase_params, redeem_params

def arima_param_search():
    """
    网格搜索最优ARIMA参数（同时搜索申购和赎回）
    """
    return search_both_purchase_and_redeem_params()

if __name__ == "__main__":
    arima_param_search() 