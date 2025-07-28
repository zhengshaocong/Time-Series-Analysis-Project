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
    对ARIMA模型进行(p,d,q)网格搜索，返回AIC最优的参数和模型
    
    ARIMA模型是时间序列分析中最常用的模型之一，包含三个参数：
    - p: AR（自回归）项数，表示当前值与前p个值的关系
    - d: 差分次数，用于使时间序列平稳
    - q: MA（移动平均）项数，表示当前值与过去q个预测误差的关系
    
    本函数通过网格搜索的方式，遍历所有可能的参数组合，选择AIC最小的模型。
    
    参数：
        ts: pd.Series
            待建模的时间序列数据
            必须是平稳的时间序列（可以通过差分实现）
        p_range: iterable
            AR参数p的取值范围，如range(0, 4)
            建议范围：0-10，根据数据长度调整
        d_range: iterable
            差分次数d的取值范围，如range(0, 3)
            建议范围：0-2，通常0或1就足够
        q_range: iterable
            MA参数q的取值范围，如range(0, 4)
            建议范围：0-10，根据数据长度调整
        max_params: int, 默认 None
            最大允许的参数个数（p+q+1），用于防止过拟合
            如果为None，则不限制参数个数
            建议设置为数据长度的5%或最多10个参数
        verbose: bool, 默认 True
            是否打印详细的搜索过程
            包括每次尝试的参数组合、AIC值和失败信息
    
    返回：
        best_params: tuple 或 None
            最优的ARIMA参数组合 (p, d, q)
            如果所有参数组合都失败，返回None
        best_model: ARIMA模型对象 或 None
            使用最优参数拟合的ARIMA模型
            包含完整的模型信息和预测方法
    
    示例：
        >>> import pandas as pd
        >>> from utils.arima_grid_search import arima_grid_search
        >>> 
        >>> # 创建示例时间序列
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> 
        >>> # 进行网格搜索
        >>> best_params, best_model = arima_grid_search(
        >>>     ts=data,
        >>>     p_range=range(0, 3),
        >>>     d_range=range(0, 2),
        >>>     q_range=range(0, 3),
        >>>     max_params=5,
        >>>     verbose=True
        >>> )
        >>> 
        >>> # 使用最优模型进行预测
        >>> if best_model is not None:
        >>>     forecast = best_model.forecast(steps=5)
        >>>     print(f"预测结果: {forecast}")
    
    注意事项：
        1. 输入时间序列必须是平稳的（可以通过ADF检验验证）
        2. 参数范围不宜过大，避免过拟合和计算时间过长
        3. max_params参数可以有效防止过拟合
        4. 如果所有参数组合都失败，检查数据质量和参数范围
        5. AIC越小表示模型越好，但也要考虑模型的简洁性
    """
    # 输入验证
    if ts is None or len(ts) == 0:
        raise ValueError("时间序列不能为空")
    
    if max_params is not None and max_params <= 0:
        raise ValueError("max_params必须大于0")
    
    # 过滤statsmodels的警告，避免输出过多警告信息
    warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')
    
    # 初始化最优结果
    best_aic = np.inf  # 初始化为无穷大，确保任何有效模型都能被选中
    best_params = None
    best_model = None
    
    # 计算总搜索次数，用于进度显示
    total_combinations = len(list(p_range)) * len(list(d_range)) * len(list(q_range))
    if verbose:
        print(f"开始网格搜索，共 {total_combinations} 种参数组合...")
    
    # 网格搜索：遍历所有可能的参数组合
    for p in p_range:
        for d in d_range:
            for q in q_range:
                # 检查参数个数是否超过限制
                total_params = p + q + 1  # AR参数 + MA参数 + 常数项
                if max_params is not None and total_params > max_params:
                    if verbose:
                        print(f'ARIMA({p},{d},{q}) - 参数个数({total_params})超过限制({max_params})，跳过')
                    continue
                
                try:
                    # 创建ARIMA模型
                    model = ARIMA(ts, order=(p, d, q))
                    
                    # 拟合模型，使用最基本的拟合方法
                    # 不添加任何额外参数，确保兼容性
                    model_fit = model.fit()
                    
                    # 获取AIC值（Akaike Information Criterion）
                    # AIC越小表示模型越好
                    aic = model_fit.aic
                    
                    if verbose:
                        print(f'ARIMA({p},{d},{q}) - 参数个数:{total_params} - AIC: {aic:.2f}')
                    
                    # 更新最优结果
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = model_fit
                        
                except Exception as e:
                    # 处理拟合失败的情况
                    if verbose:
                        error_msg = str(e)[:50]  # 只显示前50个字符
                        print(f'ARIMA({p},{d},{q}) - 失败: {error_msg}...')
                    continue
    
    # 输出最终结果
    if best_params is not None:
        total_params = best_params[0] + best_params[2] + 1
        print(f'✅ 最优参数: ARIMA{best_params}，参数个数:{total_params}，AIC={best_aic:.2f}')
        print(f'💡 建议：使用此参数组合进行ARIMA建模')
    else:
        print('❌ 未找到有效的ARIMA参数组合')
        print('💡 建议：检查数据质量或调整参数范围')
    
    return best_params, best_model 