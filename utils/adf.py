#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADF (Augmented Dickey-Fuller) 平稳性检验工具

本模块提供了时间序列平稳性检验的功能，主要用于：
1. 检验时间序列是否平稳
2. 为ARIMA模型确定差分次数d
3. 确保时间序列分析的有效性

作者: AI Assistant
创建时间: 2024
版本: 1.0
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''): 
    """
    对时间序列进行ADF检验，输出详细的检验结果
    
    ADF检验是时间序列分析中最常用的平稳性检验方法之一。
    原假设H0：时间序列是非平稳的（存在单位根）
    备择假设H1：时间序列是平稳的（不存在单位根）
    
    参数：
        series: pd.Series
            待检验的时间序列数据
            必须是一维的时间序列数据
        title: str, 默认 ''
            序列名称，用于输出结果的标识
            如果为空，则只显示"ADF检验结果"
    
    返回：
        result: dict
            包含ADF检验的主要结果，包括：
            - 'ADF统计量': ADF检验统计量
            - 'p值': 显著性水平
            - '滞后数': 自动选择的滞后阶数
            - '观测值数': 参与检验的观测值数量
    
    示例：
        >>> import pandas as pd
        >>> from utils.adf import adf_test
        >>> 
        >>> # 创建示例时间序列
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = adf_test(data, "示例序列")
        >>> 
        >>> # 判断结果
        >>> if result['p值'] < 0.05:
        >>>     print("序列平稳，可以进行ARIMA建模")
        >>> else:
        >>>     print("序列非平稳，需要进行差分处理")
    
    注意事项：
        1. 输入数据必须是pandas.Series类型
        2. 数据中不能包含无穷大或NaN值
        3. 建议在ARIMA建模前进行此检验
        4. p值小于0.05表示序列平稳（拒绝原假设）
    """
    # 输入验证
    if not isinstance(series, pd.Series):
        raise TypeError("输入数据必须是pandas.Series类型")
    
    if series.empty:
        raise ValueError("输入序列不能为空")
    
    # 移除NaN值并执行ADF检验
    # autolag='AIC'表示使用AIC准则自动选择最优滞后阶数
    print(f'\nADF检验结果: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    
    # 提取主要结果
    labels = ['ADF统计量', 'p值', '滞后数', '观测值数']
    out = dict(zip(labels, result[:4]))
    
    # 输出主要统计量
    for key, value in out.items():
        print(f'{key}: {value}')
    
    # 输出临界值（不同显著性水平下的临界值）
    for key, value in result[4].items():
        print(f'临界值 {key}: {value}')
    
    # 判断平稳性并输出结论
    if out['p值'] < 0.05:
        print('✅ 序列平稳（拒绝原假设）')
        print('💡 建议：可以直接进行ARIMA建模，差分次数d=0')
    else:
        print('❌ 序列非平稳（不能拒绝原假设）')
        print('💡 建议：需要进行差分处理，差分次数d>0')
    
    return out 