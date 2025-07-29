#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
差分验证工具

本模块提供自动化的差分验证功能，用于：
1. 自动确定最优差分次数d
2. 逐步验证差分后的平稳性
3. 避免过度差分
4. 提供差分建议

作者: AI Assistant
创建时间: 2024
版本: 1.0
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from utils.adf import adf_test
from config import ARIMA_CONFIG

def validate_differencing(series, initial_d=0, max_d=3, test_methods=None):
    """
    自动验证差分次数
    
    参数：
        series: pd.Series - 原始时间序列
        initial_d: int - 初始差分次数，默认0
        max_d: int - 最大差分次数，默认3
        test_methods: list - 检验方法列表，默认['adf', 'kpss']
    
    返回：
        dict: 包含最优差分次数和验证结果
    """
    if test_methods is None:
        test_methods = ['adf', 'kpss']
    
    print("=" * 60)
    print("🔍 自动差分验证")
    print("=" * 60)
    
    results = {}
    current_series = series.copy()
    current_d = initial_d
    
    # 如果初始d>0，先进行差分
    if current_d > 0:
        print(f"📊 应用初始差分 d={current_d}")
        for i in range(current_d):
            current_series = current_series.diff().dropna()
        print(f"✅ 初始差分完成，序列长度: {len(current_series)}")
    
    # 逐步验证差分
    for d in range(current_d, max_d + 1):
        print(f"\n{'='*50}")
        print(f"🔬 验证差分次数 d={d}")
        print(f"{'='*50}")
        
        # 执行平稳性检验
        test_results = perform_stationarity_tests(current_series, test_methods, f"差分d={d}")
        
        # 判断是否平稳
        is_stationary = check_stationarity(test_results, test_methods)
        
        results[d] = {
            'series': current_series.copy(),
            'test_results': test_results,
            'is_stationary': is_stationary,
            'series_length': len(current_series)
        }
        
        print(f"📊 差分d={d} 平稳性: {'✅ 平稳' if is_stationary else '❌ 非平稳'}")
        
        # 如果已经平稳，停止验证
        if is_stationary:
            print(f"🎯 找到最优差分次数: d={d}")
            break
        
        # 如果还没到最大差分次数，继续差分
        if d < max_d:
            print(f"🔄 进行下一次差分...")
            current_series = current_series.diff().dropna()
            if len(current_series) < 10:  # 防止序列过短
                print(f"⚠️  差分后序列过短 ({len(current_series)} < 10)，停止差分")
                break
        else:
            print(f"⚠️  达到最大差分次数 d={max_d}")
    
    # 确定最优差分次数
    optimal_d = determine_optimal_d(results)
    
    return {
        'optimal_d': optimal_d,
        'validation_results': results,
        'summary': generate_summary(results, optimal_d)
    }

def perform_stationarity_tests(series, test_methods, title=''):
    """
    执行平稳性检验
    
    参数：
        series: pd.Series - 时间序列
        test_methods: list - 检验方法
        title: str - 标题
    
    返回：
        dict: 检验结果
    """
    results = {}
    
    for method in test_methods:
        if method == 'adf':
            print(f"\n📊 执行ADF检验...")
            try:
                adf_result = adf_test(series, title=title)
                results['adf'] = adf_result
            except Exception as e:
                print(f"❌ ADF检验失败: {e}")
                results['adf'] = None
        
        elif method == 'kpss':
            print(f"\n📊 执行KPSS检验...")
            try:
                kpss_result = kpss_test(series, title=title)
                results['kpss'] = kpss_result
            except Exception as e:
                print(f"❌ KPSS检验失败: {e}")
                results['kpss'] = None
    
    return results

def kpss_test(series, title=''):
    """
    KPSS平稳性检验
    
    参数：
        series: pd.Series - 时间序列数据
        title: str - 序列标题
    
    返回：
        dict: KPSS检验结果
    """
    print(f'\nKPSS检验结果: {title}')
    
    # 执行KPSS检验
    kpss_stat, p_value, lags, critical_values = kpss(series.dropna(), regression='c')
    
    # 提取结果
    result = {
        'KPSS统计量': kpss_stat,
        'p值': p_value,
        '滞后数': lags,
        '观测值数': len(series.dropna())
    }
    
    # 输出结果
    for key, value in result.items():
        print(f'{key}: {value}')
    
    # 输出临界值
    for key, value in critical_values.items():
        print(f'临界值 {key}: {value}')
    
    # 判断平稳性
    if p_value < 0.05:
        print('❌ 序列非平稳（拒绝原假设）')
    else:
        print('✅ 序列平稳（不能拒绝原假设）')
    
    return result

def check_stationarity(test_results, test_methods):
    """
    检查平稳性
    
    参数：
        test_results: dict - 检验结果
        test_methods: list - 检验方法
    
    返回：
        bool: 是否平稳
    """
    stationary_count = 0
    total_tests = 0
    
    for method in test_methods:
        if method in test_results and test_results[method] is not None:
            total_tests += 1
            result = test_results[method]
            
            if method == 'adf':
                if result['p值'] < 0.05:
                    stationary_count += 1
            elif method == 'kpss':
                if result['p值'] >= 0.05:
                    stationary_count += 1
    
    # 如果超过一半的检验认为平稳，则认为是平稳的
    return stationary_count / total_tests >= 0.5 if total_tests > 0 else False

def determine_optimal_d(results):
    """
    确定最优差分次数
    
    参数：
        results: dict - 验证结果
    
    返回：
        int: 最优差分次数
    """
    optimal_d = None
    
    for d in sorted(results.keys()):
        if results[d]['is_stationary']:
            optimal_d = d
            break
    
    # 如果没有找到平稳的序列，选择序列最长的
    if optimal_d is None:
        max_length = 0
        for d, result in results.items():
            if result['series_length'] > max_length:
                max_length = result['series_length']
                optimal_d = d
    
    return optimal_d

def generate_summary(results, optimal_d):
    """
    生成验证摘要
    
    参数：
        results: dict - 验证结果
        optimal_d: int - 最优差分次数
    
    返回：
        str: 摘要信息
    """
    summary = f"\n{'='*60}\n"
    summary += "📋 差分验证摘要\n"
    summary += f"{'='*60}\n"
    
    for d in sorted(results.keys()):
        result = results[d]
        status = "✅ 平稳" if result['is_stationary'] else "❌ 非平稳"
        summary += f"差分d={d}: {status} (长度: {result['series_length']})\n"
    
    summary += f"\n🎯 推荐差分次数: d={optimal_d}\n"
    
    if optimal_d == 0:
        summary += "💡 建议: 原序列已平稳，可直接建模\n"
    elif optimal_d == 1:
        summary += "💡 建议: 使用一阶差分，符合大多数金融数据特点\n"
    else:
        summary += f"💡 建议: 使用{optimal_d}阶差分，注意避免过度差分\n"
    
    return summary

def get_differenced_series(series, d):
    """
    获取差分后的序列
    
    参数：
        series: pd.Series - 原始序列
        d: int - 差分次数
    
    返回：
        pd.Series: 差分后的序列
    """
    if d == 0:
        return series
    
    diff_series = series.copy()
    for i in range(d):
        diff_series = diff_series.diff().dropna()
    
    return diff_series 