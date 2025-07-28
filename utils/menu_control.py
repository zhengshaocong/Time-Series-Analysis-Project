#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式菜单控制工具

本模块提供了基于方向键的交互式菜单功能，主要用于：
1. 提供直观的方向键导航界面
2. 支持跨平台的键盘输入处理
3. 提供多种类型的交互对话框
4. 实现优雅的用户界面体验

支持的操作系统：
- Windows (使用msvcrt模块)
- macOS/Linux (使用tty/termios模块)

作者: AI Assistant
创建时间: 2024
版本: 1.0
"""

import os
import sys

# 根据操作系统导入不同的键盘输入模块
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Unix/Linux/Mac
    import tty
    import termios

def get_key():
    """
    获取键盘输入，支持方向键检测
    
    本函数实现了跨平台的键盘输入处理，能够识别：
    - 方向键（上下左右）
    - 回车键（确认）
    - q键（退出）
    - 其他普通字符
    
    返回：
        str: 按键标识
            - 'UP': 上方向键
            - 'DOWN': 下方向键
            - 'LEFT': 左方向键
            - 'RIGHT': 右方向键
            - 'ENTER': 回车键
            - 'QUIT': q键
            - 其他字符: 直接返回字符
    
    示例：
        >>> key = get_key()
        >>> if key == 'UP':
        >>>     print("用户按了上方向键")
        >>> elif key == 'ENTER':
        >>>     print("用户按了回车键")
    
    注意事项：
        1. 在Windows上使用msvcrt模块
        2. 在Unix系统上使用tty/termios模块
        3. 方向键会产生特殊的转义序列
        4. 函数会阻塞等待用户输入
    """
    if os.name == 'nt':  # Windows系统
        key = msvcrt.getch()
        if key == b'\xe0':  # 方向键前缀
            key = msvcrt.getch()
            if key == b'H':  # 上箭头
                return 'UP'
            elif key == b'P':  # 下箭头
                return 'DOWN'
            elif key == b'M':  # 右箭头
                return 'RIGHT'
            elif key == b'K':  # 左箭头
                return 'LEFT'
        elif key == b'\r':  # 回车
            return 'ENTER'
        elif key == b'q':  # q键退出
            return 'QUIT'
        else:
            return key.decode('utf-8', errors='ignore')
    else:  # Unix/Linux/Mac系统
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # ESC序列
                ch = sys.stdin.read(1)
                if ch == '[':
                    ch = sys.stdin.read(1)
                    if ch == 'A':  # 上箭头
                        return 'UP'
                    elif ch == 'B':  # 下箭头
                        return 'DOWN'
                    elif ch == 'C':  # 右箭头
                        return 'RIGHT'
                    elif ch == 'D':  # 左箭头
                        return 'LEFT'
            elif ch == '\r':  # 回车
                return 'ENTER'
            elif ch == 'q':  # q键退出
                return 'QUIT'
            else:
                return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def clear_screen():
    """
    清屏函数
    
    根据操作系统调用相应的清屏命令：
    - Windows: cls
    - Unix/Linux/Mac: clear
    
    示例：
        >>> clear_screen()  # 清空当前屏幕
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def _display_menu(title, subtitle, options, selected, show_instructions=True):
    """
    显示菜单的公共函数
    
    这是菜单显示的核心函数，负责：
    1. 清屏并显示菜单标题
    2. 显示操作说明
    3. 高亮显示当前选中项
    4. 格式化菜单布局
    
    参数：
        title: str
            菜单主标题
        subtitle: str
            菜单副标题，可以为空
        options: list
            选项列表
        selected: int
            当前选中的索引
        show_instructions: bool, 默认 True
            是否显示操作说明
    
    示例：
        >>> _display_menu("主菜单", "请选择功能", ["选项1", "选项2"], 0)
    """
    clear_screen()
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    if subtitle:
        print(subtitle)
        print("=" * 60)
    
    if show_instructions:
        print("使用 ↑↓ 方向键选择，回车确认，q 取消")
        print("=" * 60)
    
    for i, option in enumerate(options):
        if i == selected:
            print(f"▶ {option} ◀")  # 高亮显示选中项
        else:
            print(f"  {option}")
    
    print("=" * 60)

def _handle_menu_navigation(selected, total_options, key):
    """
    处理菜单导航的公共函数
    
    负责处理方向键导航逻辑，包括：
    1. 上下方向键移动选择
    2. 左右方向键移动选择（兼容性）
    3. 循环选择（到达边界时循环）
    
    参数：
        selected: int
            当前选中的索引
        total_options: int
            选项总数
        key: str
            按键标识
    
    返回：
        int: 新的选中索引
    
    示例：
        >>> new_selected = _handle_menu_navigation(0, 3, 'DOWN')
        >>> print(new_selected)  # 输出: 1
    """
    if key == 'UP':
        return (selected - 1) % total_options  # 向上循环
    elif key == 'DOWN':
        return (selected + 1) % total_options  # 向下循环
    elif key == 'LEFT':
        return (selected - 1) % total_options  # 向左循环（兼容性）
    elif key == 'RIGHT':
        return (selected + 1) % total_options  # 向右循环（兼容性）
    return selected

def show_interactive_menu(menu_items, title="菜单", subtitle="使用 ↑↓ 方向键选择，回车确认，q 退出"):
    """
    显示交互式菜单（方向键导航）
    
    这是主要的菜单显示函数，提供：
    1. 方向键导航选择
    2. 实时高亮显示
    3. 回车确认选择
    4. q键退出功能
    
    参数：
        menu_items: list
            菜单项列表
        title: str, 默认 "菜单"
            菜单标题
        subtitle: str, 默认 "使用 ↑↓ 方向键选择，回车确认，q 退出"
            菜单副标题
    
    返回：
        int: 选中的索引（从0开始）
        -1: 用户选择退出
    
    示例：
        >>> options = ["选项1", "选项2", "选项3"]
        >>> selected = show_interactive_menu(options, "我的菜单")
        >>> if selected >= 0:
        >>>     print(f"用户选择了: {options[selected]}")
        >>> else:
        >>>     print("用户选择退出")
    
    注意事项：
        1. 菜单项索引从0开始
        2. 返回-1表示用户选择退出
        3. 支持循环选择
        4. 实时更新显示
    """
    selected = 0
    
    while True:
        _display_menu(title, subtitle, menu_items, selected)
        
        # 获取用户输入
        key = get_key()
        
        if key in ['UP', 'DOWN']:
            selected = _handle_menu_navigation(selected, len(menu_items), key)
        elif key == 'ENTER':
            return selected
        elif key == 'QUIT':
            return -1
        # 忽略其他按键

def show_simple_menu(menu_items, title="菜单"):
    """
    显示简单数字菜单（备用方案）
    
    当方向键菜单不可用时，提供数字输入作为备用方案。
    功能包括：
    1. 数字选择菜单项
    2. 输入验证
    3. 错误处理
    4. 优雅退出
    
    参数：
        menu_items: list
            菜单项列表
        title: str, 默认 "菜单"
            菜单标题
    
    返回：
        int: 选中的索引（从0开始）
        -1: 用户选择退出
    
    示例：
        >>> options = ["功能1", "功能2", "功能3"]
        >>> selected = show_simple_menu(options, "备用菜单")
        >>> if selected >= 0:
        >>>     print(f"选择了: {options[selected]}")
    
    注意事项：
        1. 菜单项编号从1开始显示
        2. 输入0退出程序
        3. 包含输入验证和错误处理
        4. 支持Ctrl+C优雅退出
    """
    while True:
        clear_screen()
        print("=" * 60)
        print(title)
        print("=" * 60)
        
        for i, item in enumerate(menu_items, 1):
            print(f"{i}. {item}")
        
        print("0. 🚪 退出程序")
        print("=" * 60)
        
        try:
            choice = input("请选择功能 (输入数字): ").strip()
            
            if choice == '0':
                return -1
            elif choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(menu_items):
                    return choice_num - 1
                else:
                    print("❌ 无效选项，请输入正确的数字")
                    input("按回车继续...")
                    continue
            else:
                print("❌ 请输入数字")
                input("按回车继续...")
                continue
                
        except KeyboardInterrupt:
            return -1
        except EOFError:
            return -1

def show_confirm_dialog(message="确认操作？", default_yes=True):
    """
    显示确认对话框（使用方向键）
    
    提供"是/否"选择的交互对话框，支持：
    1. 上下方向键选择
    2. 回车确认
    3. q键取消
    4. 默认选择设置
    
    参数：
        message: str, 默认 "确认操作？"
            确认消息
        default_yes: bool, 默认 True
            默认是否选中"是"
    
    返回：
        bool: True表示选择"是"，False表示选择"否"
        None: 用户取消操作
    
    示例：
        >>> result = show_confirm_dialog("是否删除文件？")
        >>> if result is True:
        >>>     print("用户选择删除")
        >>> elif result is False:
        >>>     print("用户选择不删除")
        >>> else:
        >>>     print("用户取消操作")
    
    注意事项：
        1. 使用上下方向键选择
        2. 支持默认选择设置
        3. 返回None表示取消
        4. 界面友好，有明确的视觉提示
    """
    options = ["✅ 是", "❌ 否"]
    selected = 0 if default_yes else 1
    
    while True:
        _display_menu("❓ 确认操作", message, options, selected)
        
        # 获取用户输入
        key = get_key()
        
        if key in ['UP', 'DOWN']:
            selected = _handle_menu_navigation(selected, len(options), key)
        elif key == 'ENTER':
            return selected == 0  # 返回True表示"是"，False表示"否"
        elif key == 'QUIT':
            return None  # 取消操作
        # 忽略其他按键

def show_three_way_dialog(message="请选择操作", options=None):
    """
    显示三选项对话框（使用方向键）
    
    提供三个选项的交互对话框，常用于：
    1. 使用缓存/重新生成/取消
    2. 是/否/取消
    3. 其他三选项场景
    
    参数：
        message: str, 默认 "请选择操作"
            对话框消息
        options: list, 默认 None
            选项列表，默认为["使用缓存", "重新搜索", "取消"]
    
    返回：
        int: 0表示第一个选项，1表示第二个选项，2表示第三个选项
        -1: 用户取消操作
    
    示例：
        >>> result = show_three_way_dialog("如何处理缓存？", 
        >>>                                ["使用缓存", "重新生成", "取消"])
        >>> if result == 0:
        >>>     print("使用缓存")
        >>> elif result == 1:
        >>>     print("重新生成")
        >>> elif result == 2:
        >>>     print("取消操作")
    
    注意事项：
        1. 使用上下方向键选择
        2. 支持自定义选项
        3. 返回-1表示取消
        4. 选项编号从0开始
    """
    if options is None:
        options = ["✅ 使用缓存", "🔄 重新搜索", "❌ 取消"]
    
    selected = 0
    
    while True:
        _display_menu("❓ 选择操作", message, options, selected)
        
        # 获取用户输入
        key = get_key()
        
        if key in ['UP', 'DOWN']:
            selected = _handle_menu_navigation(selected, len(options), key)
        elif key == 'ENTER':
            return selected
        elif key == 'QUIT':
            return -1
        # 忽略其他按键

def show_continue_dialog():
    """
    显示继续操作对话框
    
    这是一个便捷函数，用于询问用户是否继续操作。
    实际上是show_confirm_dialog的包装，默认选择"是"。
    
    返回：
        bool: True表示继续，False表示退出
    
    示例：
        >>> if show_continue_dialog():
        >>>     print("继续执行")
        >>> else:
        >>>     print("退出程序")
    """
    return show_confirm_dialog("是否继续运行其他功能？", default_yes=True)

def show_press_enter_dialog(message="按回车继续..."):
    """
    显示"按回车继续"对话框
    
    用于暂停程序执行，等待用户按回车继续。
    常用于：
    1. 显示结果后暂停
    2. 错误信息显示后等待
    3. 操作完成后的确认
    
    参数：
        message: str, 默认 "按回车继续..."
            显示的消息
    
    示例：
        >>> print("操作完成！")
        >>> show_press_enter_dialog("按回车返回主菜单...")
    
    注意事项：
        1. 只等待回车键
        2. 支持q键退出
        3. 界面简洁清晰
        4. 常用于流程控制
    """
    _display_menu("ℹ️  提示", message, [], 0, show_instructions=False)
    print("按回车键继续...")
    print("=" * 60)
    
    # 等待回车键
    while True:
        key = get_key()
        if key == 'ENTER':
            break
        elif key == 'QUIT':
            break 