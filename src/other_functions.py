#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
其他功能模块

包含：运行所有功能、帮助、管理缓存、配置、退出等。
"""
from src.plot_trend import plot_trend
from src.arima_param_search import arima_param_search
from src.arima_predict import arima_predict
from utils.cache_manager import cache_manager
from config import print_config_summary, validate_config, get_data_file_path, get_output_path, VISUALIZATION_CONFIG
from utils.menu_control import show_press_enter_dialog, show_confirm_dialog, clear_screen, show_three_way_dialog
import os
import sys
import subprocess
from pathlib import Path

def run_all():
    print("\n🚀 开始运行所有主要功能...")
    results = []
    results.append(("趋势图", plot_trend()))
    results.append(("ARIMA参数搜索", arima_param_search()))
    results.append(("ARIMA预测", arima_predict()))
    print("\n📊 执行结果总结:")
    for name, result in results:
        print(f"{name:<20} : {'✅ 成功' if result else '❌ 失败'}")
    print("\n🎉 所有功能执行完成！")
    show_press_enter_dialog()

def handle_plot_with_cache():
    data_file_path = get_data_file_path()
    cache_manager.refresh_cache()
    image_cache = cache_manager.get_image_cache(data_file_path, 'trend')
    if image_cache and image_cache['exists']:
        print(f"\n{'='*60}")
        print("📋 发现趋势图缓存:")
        print(f"图片路径: {image_cache['path']}")
        print(f"生成时间: {image_cache['timestamp']}")
        print(f"描述: {image_cache['description']}")
        print(f"{'='*60}")
        choice = show_three_way_dialog("是否使用缓存的趋势图？", ["✅ 查看结果", "🔄 重新生成", "❌ 取消"])
        if choice == 0:
            print("✅ 查看缓存结果")
            _open_image(image_cache['path'])
            return
        elif choice == 2:
            print("❌ 操作已取消")
            return
        elif choice == 1:
            print("🔄 将重新生成趋势图...")
    # 重新生成
    result = plot_trend()
    # 保存图片缓存
    output_path = get_output_path(os.path.join(VISUALIZATION_CONFIG['output_dir'], 'purchase_redeem_trend.png'))
    cache_manager.save_image_cache(
        data_file_path, 'trend', output_path, "用户申购和赎回金额的时间趋势图 (output/images/)")
    print("💡 趋势图生成完成，正在打开...")
    _open_image(output_path)

def handle_predict_with_cache():
    data_file_path = get_data_file_path()
    cache_manager.refresh_cache()
    # 检查ARIMA参数缓存
    param_cache = cache_manager.get_cached_params(data_file_path)
    has_param_cache = param_cache and isinstance(param_cache, dict) and 'best_params' in param_cache
    # 检查图片缓存
    image_cache = cache_manager.get_image_cache(data_file_path, 'prediction')
    if image_cache and image_cache['exists']:
        print(f"\n{'='*60}")
        print("📋 发现预测图缓存:")
        print(f"图片路径: {image_cache['path']}")
        print(f"生成时间: {image_cache['timestamp']}")
        print(f"描述: {image_cache['description']}")
        print(f"{'='*60}")
        choice = show_three_way_dialog("是否使用缓存的预测图？", ["✅ 查看结果", "🔄 重新生成", "❌ 取消"])
        if choice == 0:
            print("✅ 查看缓存结果")
            _open_image(image_cache['path'])
            return
        elif choice == 2:
            print("❌ 操作已取消")
            return
        elif choice == 1:
            print("🔄 将重新生成预测图...")
    # 若无参数缓存，先自动执行ARIMA参数搜索
    if not has_param_cache:
        print("⚠️ 未找到ARIMA参数缓存，自动执行参数搜索...")
        arima_param_search()
        cache_manager.refresh_cache()
    # 重新生成预测图
    result = arima_predict()
    output_path = get_output_path(os.path.join(VISUALIZATION_CONFIG['output_dir'], 'arima_purchase_201409_201412_forecast.png'))
    cache_manager.save_image_cache(
        data_file_path, 'prediction', output_path, "ARIMA模型预测申购金额图 (output/images/)")
    print("💡 预测图生成完成，正在打开...")
    _open_image(output_path)

def _open_image(image_path):
    abs_image_path = Path(image_path)
    if not abs_image_path.is_absolute():
        abs_image_path = Path(os.getcwd()) / abs_image_path
    try:
        if os.name == 'nt':
            os.startfile(str(abs_image_path))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(abs_image_path)], check=True)
        else:
            subprocess.run(['xdg-open', str(abs_image_path)], check=True)
        print(f"✅ 已在默认应用中打开图片: {abs_image_path}")
    except Exception as e:
        print(f"❌ 打开图片失败: {e}")
        print(f"💡 请手动打开文件: {abs_image_path}")

def show_help():
    clear_screen()
    print("=" * 60)
    print("📖 帮助信息")
    print("=" * 60)
    print("1. 📈 绘制资金流入流出趋势图：分析用户申购和赎回金额的时间趋势，输出到 output/images/")
    print("2. 🔍 ARIMA参数搜索：自动网格搜索最优ARIMA(p,d,q)参数，支持缓存，输出AIC等信息")
    print("3. 🔮 ARIMA预测：使用ARIMA模型预测申购金额，支持缓存参数，输出预测图")
    print("4. 🚀 运行所有功能：依次执行所有主要分析功能")
    print("5. ⚙️  查看配置：显示当前所有配置参数和校验结果")
    print("6. 🗑️  管理缓存：查看/清除参数和图片缓存")
    print("7. 🚪 退出程序")
    print("=" * 60)
    print("使用方向键或数字选择功能，回车确认，q 退出")
    show_press_enter_dialog()

def show_config():
    clear_screen()
    print_config_summary()
    errors = validate_config()
    if errors:
        print("\n❌ 配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ 配置验证通过")
    show_press_enter_dialog()

def manage_cache():
    while True:
        clear_screen()
        print("=" * 60)
        print("🗑️  缓存管理")
        print("=" * 60)
        print("1. 📋 查看所有缓存")
        print("2. 🖼️  查看图片缓存")
        print("3. 🗑️  清除当前文件缓存")
        print("4. 🗑️  清除所有缓存")
        print("0. 🔙 返回主菜单")
        print("=" * 60)
        choice = input("请选择操作: ").strip()
        if choice == '0':
            break
        elif choice == '1':
            cache_manager.refresh_cache()
            cache_manager.list_cache()
            show_press_enter_dialog()
        elif choice == '2':
            show_image_cache_info()
        elif choice == '3':
            data_file_path = get_data_file_path()
            cache_manager.clear_cache(data_file_path)
            show_press_enter_dialog()
        elif choice == '4':
            confirm = show_confirm_dialog("确定要清除所有缓存吗？")
            if confirm:
                cache_manager.clear_cache()
            show_press_enter_dialog()
        else:
            print("❌ 无效选择")
            show_press_enter_dialog()

def show_image_cache_info():
    data_file_path = get_data_file_path()
    cache_manager.refresh_cache()
    all_images = cache_manager.get_all_image_cache(data_file_path)
    if not all_images:
        print("📭 暂无图片缓存记录")
        show_press_enter_dialog()
        return
    print("🖼️  图片缓存信息:")
    print("=" * 80)
    for image_type, image_info in all_images.items():
        print(f"类型: {image_type}")
        print(f"路径: {image_info['path']}")
        print(f"描述: {image_info['description']}")
        print(f"生成时间: {image_info['timestamp']}")
        print(f"文件存在: {'✅ 是' if image_info['exists'] else '❌ 否'}")
        print("-" * 40)
    show_press_enter_dialog()

def exit_program():
    print("👋 感谢使用，再见！")
    exit(0) 