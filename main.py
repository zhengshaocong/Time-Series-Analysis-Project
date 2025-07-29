#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口，统一调度src下的各功能模块，保留原run.py的交互体验。
"""
import sys
import argparse
from pathlib import Path
from src.plot_trend import plot_trend
from src.arima_param_search import arima_param_search
from src.arima_predict import arima_predict
from src.csv_export import handle_csv_export_with_cache
from src.stationarity_test import stationarity_test
from src.other_functions import run_all, show_help, exit_program, manage_cache, show_config, handle_plot_with_cache, handle_predict_with_cache
from utils.menu_control import (
    show_interactive_menu, show_simple_menu, clear_screen,
    show_confirm_dialog, show_three_way_dialog, show_continue_dialog,
    show_press_enter_dialog
)
from utils.cache_manager import cache_manager
from config import (
    get_enabled_programs, get_data_file_path, CACHE_CONFIG
)

# 主功能id到函数的映射
def get_program_func(program_id):
    if program_id == 'plot':
        return handle_plot_with_cache
    elif program_id == 'param-search':
        return arima_param_search
    elif program_id == 'predict':
        return handle_predict_with_cache
    elif program_id == 'csv-export':
        return handle_csv_export_with_cache
    elif program_id == 'stationarity-test':
        return stationarity_test
    else:
        return None

def get_image_cache_summary(data_file_path, image_type):
    image_cache = cache_manager.get_image_cache(data_file_path, image_type)
    if image_cache and image_cache.get('exists'):
        return f"🖼️ 已缓存"
    elif image_cache:
        return f"⚠️ 缓存丢失"
    else:
        return ""

def get_csv_cache_summary(data_file_path):
    csv_cache = cache_manager.get_csv_cache(data_file_path, 'prediction')
    if csv_cache and csv_cache.get('exists'):
        return f"📊 已缓存"
    elif csv_cache:
        return f"⚠️ 缓存丢失"
    else:
        return ""

def main():
    while True:
        enabled_programs = get_enabled_programs()
        data_file_path = get_data_file_path()
        # 动态主功能菜单项，右侧追加缓存摘要和图片缓存状态
        program_names = []
        program_ids = []
        for prog in enabled_programs:
            name = prog['name']
            # ARIMA参数搜索显示参数缓存摘要
            if prog['id'] == 'param-search' and CACHE_CONFIG['settings']['show_cache_info']:
                purchase_summary = cache_manager.get_cache_summary(data_file_path, 'purchase')
                redeem_summary = cache_manager.get_cache_summary(data_file_path, 'redeem')
                # 兼容旧格式
                old_summary = cache_manager.get_cache_summary(data_file_path)
                summary_parts = []
                if purchase_summary:
                    summary_parts.append(purchase_summary)
                if redeem_summary:
                    summary_parts.append(redeem_summary)
                if not summary_parts and old_summary:
                    summary_parts.append(old_summary)
                if summary_parts:
                    name += ' ' + ' '.join(summary_parts)
            # 趋势图显示图片缓存状态
            if prog['id'] == 'plot':
                img_status = get_image_cache_summary(data_file_path, 'trend')
                if img_status:
                    name += f" {img_status}"
            # 预测图显示图片缓存状态
            if prog['id'] == 'predict':
                img_status = get_image_cache_summary(data_file_path, 'prediction')
                if img_status:
                    name += f" {img_status}"
            # CSV导出显示缓存状态
            if prog['id'] == 'csv-export':
                csv_status = get_csv_cache_summary(data_file_path)
                if csv_status:
                    name += f" {csv_status}"
            program_names.append(name)
            program_ids.append(prog['id'])
        # 固定额外功能项
        extra_items = [
            "🚀 运行所有功能",
            "❓ 查看帮助",
            "⚙️  查看配置",
            "🗑️  管理缓存",
            "🚪 退出程序"
        ]
        menu_items = program_names + extra_items
        try:
            selected = show_interactive_menu(menu_items, title="📊 时间序列分析工具", subtitle="使用 ↑↓ 方向键选择，回车确认，q 退出")
        except Exception as e:
            print(f"方向键菜单初始化失败: {e}")
            selected = show_simple_menu(menu_items, title="📊 时间序列分析工具")
        if 0 <= selected < len(program_names):
            func = get_program_func(program_ids[selected])
            if func:
                func()
                # 功能执行完成后，让用户查看结果
                print(f"\n{'='*40}")
                print("💡 功能执行完成，请查看上方结果")
                input("按回车键继续...")
            else:
                print(f"❌ 未找到功能: {program_names[selected]}")
                input("按回车键继续...")
        elif selected == len(program_names):
            run_all()
        elif selected == len(program_names) + 1:
            show_help()
        elif selected == len(program_names) + 2:
            show_config()
        elif selected == len(program_names) + 3:
            manage_cache()
        elif selected == len(program_names) + 4 or selected == -1:
            exit_program()

if __name__ == '__main__':
    main() 