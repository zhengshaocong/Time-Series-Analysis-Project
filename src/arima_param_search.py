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

def get_or_search_best_arima_params(ts_train, data_file_path=None, verbose=True):
    """
    获取最优ARIMA参数（优先缓存，无则搜索并缓存）
    返回 (p, d, q) 或 None
    """
    if data_file_path is None:
        data_file_path = get_data_file_path()
    cache_manager.refresh_cache()
    cached_info = cache_manager.get_cached_params(data_file_path)
    if cached_info and isinstance(cached_info, dict) and 'best_params' in cached_info:
        if verbose:
            print(f"✅ 读取缓存最优参数: ARIMA{cached_info['best_params']}")
        return cached_info['best_params']
    # 无缓存，执行参数搜索
    if verbose:
        print("⚠️ 未找到ARIMA参数缓存，自动执行参数搜索...")
    best_params = _search_and_cache_arima_params(ts_train, data_file_path, verbose=verbose)
    return best_params

def _search_and_cache_arima_params(ts_train, data_file_path, verbose=True):
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
                data_file_path, best_params, best_aic, total_params, data_length
            )
            if verbose:
                print(f"✅ 已缓存最优参数: ARIMA{best_params}")
        else:
            if verbose:
                print("⚠️  模型拟合失败，无法保存到缓存")
    else:
        if verbose:
            print("❌ 未找到有效的ARIMA参数组合")
    return best_params

def arima_param_search():
    """
    网格搜索最优ARIMA参数（只负责搜索和缓存，不返回参数）
    """
    file_path = get_data_file_path()
    df = pd.read_csv(file_path)
    if df['report_date'].dtype != 'O':
        df['report_date'] = df['report_date'].astype(str)
    df = df[df['report_date'] >= '20140301']
    trend = df.groupby('report_date')['total_purchase_amt'].sum().reset_index()
    trend = trend.sort_values('report_date')
    dates = pd.to_datetime(trend['report_date'], format='%Y%m%d')
    ts = pd.Series(trend['total_purchase_amt'].values, index=dates)
    ts = ts.asfreq('D')
    ts_train = ts[(ts.index >= '2014-03-01') & (ts.index <= '2014-08-31')]
    # 只负责搜索和缓存
    get_or_search_best_arima_params(ts_train, file_path, verbose=True)

if __name__ == "__main__":
    arima_param_search() 