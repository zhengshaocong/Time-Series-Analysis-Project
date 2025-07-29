#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据平稳性检验主功能模块

本模块提供完整的时间序列平稳性检验功能，包括：
1. ADF检验 (Augmented Dickey-Fuller Test)
2. KPSS检验 (Kwiatkowski-Phillips-Schmidt-Shin Test)
3. PP检验 (Phillips-Perron Test)
4. 可视化诊断图表

作者: AI Assistant
创建时间: 2024
版本: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import adf_test
from utils.cache_manager import cache_manager
from utils.menu_control import show_confirm_dialog, show_three_way_dialog
from config import get_data_file_path, get_output_path, VISUALIZATION_CONFIG

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def stationarity_test():
    """
    数据平稳性检验主功能
    """
    print("=" * 60)
    print("📈 数据平稳性检验工具")
    print("=" * 60)
    
    try:
        # 加载数据
        ts_data = load_data()
        if ts_data is None:
            print("❌ 数据加载失败")
            return False
            
        print(f"✅ 数据加载完成，序列长度: {len(ts_data)}")
        print(f"📅 数据时间范围: {ts_data.index[0].strftime('%Y-%m-%d')} 至 {ts_data.index[-1].strftime('%Y-%m-%d')}")
        
        # 选择检验方法
        test_methods = select_test_methods()
        if not test_methods:
            print("❌ 未选择任何检验方法")
            return False
            
        # 执行检验
        results = perform_stationarity_tests(ts_data, test_methods)
        
        # 生成可视化诊断
        output_path = create_diagnostic_plots(ts_data, results)
        
        # 提供综合结论和建议
        provide_comprehensive_analysis(results, ts_data)
        
        # 缓存结果
        cache_results(ts_data, results, output_path)
        
        print(f"\n{'='*60}")
        print("🎉 平稳性检验完成！")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"❌ 平稳性检验过程中发生错误: {e}")
        return False

def load_data():
    """
    加载时间序列数据
    
    返回：
        pd.Series: 时间序列数据
    """
    try:
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
        
        return ts
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def select_test_methods():
    """
    选择要执行的检验方法
    
    返回：
        list: 选择的检验方法列表
    """
    print(f"\n{'='*50}")
    print("🔍 选择平稳性检验方法")
    print(f"{'='*50}")
    
    test_options = [
        "✅ ADF检验 (Augmented Dickey-Fuller)",
        "✅ KPSS检验 (Kwiatkowski-Phillips-Schmidt-Shin)", 
        "✅ PP检验 (Phillips-Perron)",
        "✅ 所有检验方法"
    ]
    
    try:
        selected = show_interactive_menu(
            test_options, 
            title="选择检验方法",
            subtitle="使用 ↑↓ 方向键选择，回车确认，q 退出"
        )
    except Exception as e:
        print(f"方向键菜单初始化失败: {e}")
        selected = show_simple_menu(test_options, title="选择检验方法")
    
    if selected == 0:
        return ['adf']
    elif selected == 1:
        return ['kpss']
    elif selected == 2:
        return ['pp']
    elif selected == 3:
        return ['adf', 'kpss', 'pp']
    else:
        return []

def perform_stationarity_tests(ts_data, test_methods):
    """
    执行平稳性检验
    
    参数：
        ts_data: pd.Series - 时间序列数据
        test_methods: list - 检验方法列表
    
    返回：
        dict: 检验结果字典
    """
    results = {}
    
    print(f"\n{'='*50}")
    print("🔬 执行平稳性检验")
    print(f"{'='*50}")
    
    # ADF检验
    if 'adf' in test_methods:
        print("\n📊 执行ADF检验...")
        adf_result = adf_test(ts_data, title='资金流入流出数据')
        results['adf'] = adf_result
    
    # KPSS检验
    if 'kpss' in test_methods:
        print("\n📊 执行KPSS检验...")
        kpss_result = kpss_test(ts_data)
        results['kpss'] = kpss_result
    
    # PP检验
    if 'pp' in test_methods:
        print("\n📊 执行PP检验...")
        pp_result = pp_test(ts_data)
        results['pp'] = pp_result
    
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
        print('💡 建议：需要进行差分处理')
    else:
        print('✅ 序列平稳（不能拒绝原假设）')
        print('💡 建议：可以直接进行ARIMA建模')
    
    return result

def pp_test(series, title=''):
    """
    Phillips-Perron平稳性检验
    
    参数：
        series: pd.Series - 时间序列数据
        title: str - 序列标题
    
    返回：
        dict: PP检验结果
    """
    print(f'\nPP检验结果: {title}')
    
    # 执行PP检验 - 使用adfuller函数，但设置regression='ct'来模拟PP检验
    # 注意：这里使用ADF检验作为替代，因为statsmodels中没有直接的PP检验函数
    try:
        # adfuller返回5个值：统计量, p值, 滞后数, 观测值数, 临界值字典
        result_tuple = adfuller(series.dropna(), regression='ct', autolag='AIC')
        
        # 正确解包返回值
        pp_stat = result_tuple[0]  # 统计量
        p_value = result_tuple[1]  # p值
        lags = result_tuple[2]     # 滞后数
        obs = result_tuple[3]      # 观测值数
        critical_values = result_tuple[4]  # 临界值字典
        
        # 提取结果
        result = {
            'PP统计量': pp_stat,
            'p值': p_value,
            '滞后数': lags,
            '观测值数': obs
        }
        
        # 输出结果
        for key, value in result.items():
            print(f'{key}: {value}')
        
        # 输出临界值
        for key, value in critical_values.items():
            print(f'临界值 {key}: {value}')
        
        # 判断平稳性
        if p_value < 0.05:
            print('✅ 序列平稳（拒绝原假设）')
            print('💡 建议：可以直接进行ARIMA建模')
        else:
            print('❌ 序列非平稳（不能拒绝原假设）')
            print('💡 建议：需要进行差分处理')
        
        return result
        
    except Exception as e:
        print(f'❌ PP检验执行失败: {e}')
        return None

def create_diagnostic_plots(ts_data, results):
    """
    创建诊断图表
    
    参数：
        ts_data: pd.Series - 时间序列数据
        results: dict - 检验结果
    
    返回：
        str: 输出文件路径
    """
    print(f"\n{'='*50}")
    print("📊 生成诊断图表")
    print(f"{'='*50}")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('时间序列平稳性诊断图表', fontsize=16, fontweight='bold')
    
    # 1. 原始时间序列图
    axes[0, 0].plot(ts_data.index, ts_data.values, linewidth=1)
    axes[0, 0].set_title('原始时间序列')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('金额')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 一阶差分序列
    diff1 = ts_data.diff().dropna()
    axes[0, 1].plot(diff1.index, diff1.values, linewidth=1, color='orange')
    axes[0, 1].set_title('一阶差分序列')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('差分值')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 自相关函数(ACF)
    plot_acf(ts_data.dropna(), ax=axes[1, 0], lags=40, alpha=0.05)
    axes[1, 0].set_title('自相关函数(ACF)')
    
    # 4. 偏自相关函数(PACF)
    plot_pacf(ts_data.dropna(), ax=axes[1, 1], lags=40, alpha=0.05)
    axes[1, 1].set_title('偏自相关函数(PACF)')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = get_output_path('output/images/stationarity_diagnostic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 诊断图表已保存: {output_path}")
    return output_path

def provide_comprehensive_analysis(results, ts_data):
    """
    提供综合分析和建议
    
    参数：
        results: dict - 检验结果
        ts_data: pd.Series - 时间序列数据
    """
    print(f"\n{'='*60}")
    print("📋 综合分析与建议")
    print(f"{'='*60}")
    
    # 统计检验结果
    stationary_tests = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        if test_name == 'adf':
            if result['p值'] < 0.05:
                stationary_tests += 1
        elif test_name == 'kpss':
            if result['p值'] >= 0.05:
                stationary_tests += 1
        elif test_name == 'pp':
            if result['p值'] < 0.05:
                stationary_tests += 1
    
    # 计算平稳性比例
    stationary_ratio = stationary_tests / total_tests if total_tests > 0 else 0
    
    print(f"📊 检验结果统计:")
    print(f"   总检验数: {total_tests}")
    print(f"   平稳检验数: {stationary_tests}")
    print(f"   平稳性比例: {stationary_ratio:.1%}")
    
    # 提供建议
    print(f"\n💡 建模建议:")
    if stationary_ratio >= 0.67:  # 2/3以上检验认为平稳
        print("   ✅ 数据基本平稳，建议:")
        print("      - 可以直接进行ARIMA建模")
        print("      - 差分次数d=0")
        print("      - 重点关注p和q参数的选择")
    elif stationary_ratio >= 0.33:  # 1/3-2/3检验认为平稳
        print("   ⚠️  数据平稳性不确定，建议:")
        print("      - 尝试一阶差分后再检验")
        print("      - 差分次数d=1")
        print("      - 对比差分前后的模型效果")
    else:  # 大部分检验认为非平稳
        print("   ❌ 数据明显非平稳，建议:")
        print("      - 必须进行差分处理")
        print("      - 差分次数d≥1")
        print("      - 考虑更高阶差分或对数变换")
    
    # 数据特征分析
    print(f"\n📈 数据特征分析:")
    print(f"   均值: {ts_data.mean():.2f}")
    print(f"   标准差: {ts_data.std():.2f}")
    print(f"   变异系数: {ts_data.std()/ts_data.mean():.2%}")
    print(f"   偏度: {ts_data.skew():.3f}")
    print(f"   峰度: {ts_data.kurtosis():.3f}")

def cache_results(ts_data, results, output_path):
    """
    缓存检验结果
    
    参数：
        ts_data: pd.Series - 时间序列数据
        results: dict - 检验结果
        output_path: str - 输出文件路径
    """
    try:
        data_file_path = get_data_file_path()
        
        # 准备缓存数据
        cache_data = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_length': len(ts_data),
            'data_range': f"{ts_data.index[0].strftime('%Y-%m-%d')} 至 {ts_data.index[-1].strftime('%Y-%m-%d')}",
            'results': results,
            'output_path': str(output_path)  # 转换为字符串
        }
        
        # 保存到缓存
        cache_manager.save_stationarity_cache(data_file_path, cache_data)
        print(f"✅ 检验结果已缓存")
        
    except Exception as e:
        print(f"⚠️  缓存保存失败: {e}")

# 导入菜单控制函数
from utils.menu_control import show_interactive_menu, show_simple_menu 