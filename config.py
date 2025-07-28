#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列分析项目配置文件
包含数据文件路径、程序列表等配置信息
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据配置
DATA_CONFIG = {
    # 原始数据文件路径
    'data_file': 'data/user_balance_table.csv',
    
    # 数据文件列名映射
    'columns': {
        'date': 'report_date',
        'purchase': 'total_purchase_amt',
        'redeem': 'total_redeem_amt'
    },
    
    # 数据过滤条件
    'filters': {
        'start_date': '20140301',  # 只保留2014年3月及以后的数据
        'date_format': '%Y%m%d'
    }
}

# 程序配置
PROGRAMS_CONFIG = {
    # 可运行的程序列表
    'programs': [
        {
            'id': 'plot',
            'name': '📈 绘制资金流入流出趋势图',
            'description': '分析用户申购和赎回金额的时间趋势',
            'script': 'script/user_balance_trend_plot.py',
            'output': 'output/images/purchase_redeem_trend.png',
            'enabled': True
        },
        {
            'id': 'param-search',
            'name': '🔍 ARIMA参数搜索',
            'description': '网格搜索最优ARIMA(p,d,q)参数',
            'script': 'script/arima_param_search.py',
            'output': '控制台输出',
            'enabled': True
        },
        {
            'id': 'predict',
            'name': '🔮 ARIMA预测',
            'description': '使用ARIMA模型预测申购金额',
            'script': 'script/arima_purchase_predict.py',
            'output': 'output/images/arima_purchase_201409_201412_forecast.png',
            'enabled': True
        }
    ]
}

# ARIMA模型配置
ARIMA_CONFIG = {
    # 参数搜索范围
    'param_ranges': {
        'p_range': (0, 10),  # AR参数范围
        'd_range': (0, 2),   # 差分次数范围
        'q_range': (0, 10)   # MA参数范围
    },
    
    # 参数限制
    'param_limits': {
        'max_params': 10,           # 最大参数个数
        'param_ratio': 0.05         # 参数个数/数据量比例
    },
    
    # 训练集配置
    'training': {
        'start_date': '2014-03-01',
        'end_date': '2014-08-31'
    },
    
    # 预测配置
    'prediction': {
        'start_date': '2014-09-01',
        'end_date': '2014-12-31'
    }
}

# 可视化配置
VISUALIZATION_CONFIG = {
    # 图片输出目录
    'output_dir': 'output/images',
    
    # 中文字体配置
    'fonts': {
        'sans_serif': ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'],
        'unicode_minus': False
    },
    
    # 图片尺寸
    'figure_size': (14, 6),
    
    # 颜色配置
    'colors': {
        'train': 'tab:blue',
        'predict': 'tab:orange',
        'purchase': 'tab:blue',
        'redeem': 'tab:red'
    }
}

# 工具函数配置
UTILS_CONFIG = {
    # utils目录路径
    'utils_path': 'utils',
    
    # 可用的工具函数
    'functions': {
        'adf_test': 'adf.py',
        'arima_grid_search': 'arima_grid_search.py',
        'menu_control': 'menu_control.py',
        'cache_manager': 'cache_manager.py'
    }
}

# 缓存配置
CACHE_CONFIG = {
    # 缓存文件路径
    'cache_file': 'cache/arima_cache.json',
    
    # 缓存设置
    'settings': {
        'enable_cache': True,           # 是否启用缓存
        'auto_save': True,              # 是否自动保存
        'show_cache_info': True,        # 是否在菜单中显示缓存信息
        'prompt_reset': True,           # 是否提示重置缓存
        'enable_image_cache': True,     # 是否启用图片缓存
        'auto_save_images': True        # 是否自动保存图片缓存
    },
    
    # 图片缓存配置
    'image_types': {
        'trend': {
            'name': '趋势图',
            'description': '用户申购和赎回金额的时间趋势图 (output/images/)'
        },
        'prediction': {
            'name': '预测图',
            'description': 'ARIMA模型预测申购金额图 (output/images/)'
        }
    }
}

def get_data_file_path():
    """获取数据文件完整路径"""
    return PROJECT_ROOT / DATA_CONFIG['data_file']

def get_script_path(script_name):
    """获取脚本文件完整路径"""
    return PROJECT_ROOT / script_name

def get_output_path(output_name):
    """获取输出文件完整路径"""
    return PROJECT_ROOT / output_name

def get_program_by_id(program_id):
    """根据程序ID获取程序配置"""
    for program in PROGRAMS_CONFIG['programs']:
        if program['id'] == program_id:
            return program
    return None

def get_enabled_programs():
    """获取所有启用的程序"""
    return [prog for prog in PROGRAMS_CONFIG['programs'] if prog['enabled']]

def validate_config():
    """验证配置文件的有效性"""
    errors = []
    
    # 检查数据文件是否存在
    data_file = get_data_file_path()
    if not data_file.exists():
        errors.append(f"数据文件不存在: {data_file}")
    
    # 检查脚本文件是否存在
    for program in PROGRAMS_CONFIG['programs']:
        if program['enabled']:
            script_path = get_script_path(program['script'])
            if not script_path.exists():
                errors.append(f"脚本文件不存在: {script_path}")
    
    # 检查utils目录是否存在
    utils_path = PROJECT_ROOT / UTILS_CONFIG['utils_path']
    if not utils_path.exists():
        errors.append(f"utils目录不存在: {utils_path}")
    
    return errors

def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("📋 配置摘要")
    print("=" * 60)
    
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据文件: {get_data_file_path()}")
    print(f"输出目录: {PROJECT_ROOT / VISUALIZATION_CONFIG['output_dir']}")
    
    print(f"\n可运行程序 ({len(get_enabled_programs())}个):")
    for i, program in enumerate(get_enabled_programs(), 1):
        print(f"  {i}. {program['name']}")
        print(f"     描述: {program['description']}")
        print(f"     脚本: {program['script']}")
        print(f"     输出: {program['output']}")
        print()
    
    print("ARIMA配置:")
    print(f"  参数范围: p={ARIMA_CONFIG['param_ranges']['p_range']}, "
          f"d={ARIMA_CONFIG['param_ranges']['d_range']}, "
          f"q={ARIMA_CONFIG['param_ranges']['q_range']}")
    print(f"  最大参数: {ARIMA_CONFIG['param_limits']['max_params']}")
    print(f"  训练期: {ARIMA_CONFIG['training']['start_date']} ~ {ARIMA_CONFIG['training']['end_date']}")
    print(f"  预测期: {ARIMA_CONFIG['prediction']['start_date']} ~ {ARIMA_CONFIG['prediction']['end_date']}")
    
    print("=" * 60)

if __name__ == "__main__":
    # 验证配置
    errors = validate_config()
    if errors:
        print("❌ 配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 配置验证通过")
        print_config_summary() 