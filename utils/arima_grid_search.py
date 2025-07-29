#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA模型参数网格搜索工具

本模块提供了ARIMA(p,d,q)模型参数自动搜索的功能，主要用于：
1. 自动搜索最优的ARIMA参数组合
2. 基于AIC准则选择最佳模型
3. 防止过拟合（通过限制参数个数）
4. 提高ARIMA建模的自动化程度

作者: AI Assistant
创建时间: 2024
版本: 1.0
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

def arima_grid_search(ts, p_range, d_range, q_range, max_params=None, verbose=True):
    """
    网格搜索最优ARIMA参数
    
    参数:
        ts: pd.Series - 时间序列数据
        p_range: range - AR参数范围
        d_range: range - 差分次数范围
        q_range: range - MA参数范围
        max_params: int - 最大参数个数
        verbose: bool - 是否详细输出
    
    返回:
        tuple: (最优参数, 最优模型)
    """
    if max_params is None:
        max_params = min(10, int(len(ts) * 0.05))
    
    best_aic = float('inf')
    best_params = None
    best_model = None
    valid_combinations = 0
    total_combinations = len(p_range) * len(d_range) * len(q_range)
    
    if verbose:
        print(f"🔍 开始ARIMA参数网格搜索...")
        print(f"📊 参数范围: p={list(p_range)}, d={list(d_range)}, q={list(q_range)}")
        print(f"📊 最大参数个数: {max_params}")
        print(f"📊 总组合数: {total_combinations}")
        print(f"{'='*60}")
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                # 检查参数个数限制
                total_params = p + q + 1
                if total_params > max_params:
                    continue
                
                try:
                    # 创建ARIMA模型
                    model = ARIMA(ts, order=(p, d, q))
                    model_fit = model.fit()
                    
                    # 检查预测质量
                    forecast = model_fit.forecast(steps=10)
                    forecast_cv = forecast.std() / forecast.mean() if forecast.mean() != 0 else 0
                    forecast_range = forecast.max() - forecast.min()
                    
                    # 过滤掉产生直线预测的模型
                    if forecast_cv < 0.001 or forecast_range < 1000:  # 变异系数太小或预测范围太小
                        if verbose:
                            print(f"❌ ARIMA({p},{d},{q}): 预测过于平稳 (CV={forecast_cv:.4f}, 范围={forecast_range:.2f})")
                        continue
                    
                    # 检查AIC值
                    current_aic = model_fit.aic
                    
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_params = (p, d, q)
                        best_model = model_fit
                        
                        if verbose:
                            print(f"✅ 新最优: ARIMA{p,d,q} (AIC={current_aic:.2f}, CV={forecast_cv:.4f}, 范围={forecast_range:.2f})")
                    
                    valid_combinations += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"❌ ARIMA({p},{d},{q}): 拟合失败 - {str(e)[:50]}")
                    continue
    
    if verbose:
        print(f"{'='*60}")
        print(f"📊 搜索完成:")
        print(f"   有效组合数: {valid_combinations}/{total_combinations}")
        if best_params:
            print(f"   最优参数: ARIMA{best_params}")
            print(f"   最优AIC: {best_aic:.2f}")
        else:
            print(f"   ❌ 未找到有效参数组合")
    
    return best_params, best_model 