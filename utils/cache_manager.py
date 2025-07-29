#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理工具

本模块提供了ARIMA参数搜索结果的缓存管理功能，主要用于：
1. 缓存ARIMA参数搜索结果，避免重复计算
2. 缓存图片文件信息，提供快速访问
3. 基于文件哈希值确保缓存有效性
4. 提供完整的缓存管理功能

缓存机制：
- 使用文件哈希值作为缓存键的一部分
- 确保数据文件未修改时使用缓存
- 支持参数缓存和图片缓存
- 提供缓存查询、保存、清除功能

作者: AI Assistant
创建时间: 2024
版本: 1.0
"""

import json
import os
import hashlib
from pathlib import Path
from datetime import datetime

class CacheManager:
    """
    ARIMA参数缓存管理器
    
    负责管理ARIMA参数搜索结果的缓存，包括：
    1. 缓存文件的读写操作
    2. 缓存键的生成和验证
    3. 缓存数据的查询和保存
    4. 缓存的有效性检查
    
    缓存文件格式：JSON
    缓存键格式：文件名_文件哈希前8位
    """
    
    def __init__(self, cache_file="cache/arima_cache.json"):
        """
        初始化缓存管理器
        
        参数：
            cache_file: str, 默认 "cache/arima_cache.json"
                缓存文件的路径
                可以是相对路径或绝对路径
        
        示例：
            >>> cache_manager = CacheManager("my_cache.json")
            >>> cache_manager = CacheManager("/path/to/cache.json")
        """
        self.cache_file = Path(cache_file)
        self.cache_dir = self.cache_file.parent
        self.cache_data = self._load_cache()
    
    def _load_cache(self):
        """
        加载缓存数据
        
        从JSON文件中加载缓存数据，如果文件不存在或损坏则创建新的缓存。
        
        返回：
            dict: 缓存数据字典
                如果文件不存在，返回空字典
                如果文件损坏，返回空字典并打印警告
        
        异常处理：
            - JSONDecodeError: 文件格式错误
            - IOError: 文件读取错误
        
        示例：
            >>> cache_data = self._load_cache()
            >>> print(f"加载了 {len(cache_data)} 条缓存记录")
        """
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  缓存文件损坏，创建新缓存: {e}")
                return {}
        else:
            # 确保缓存目录存在
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            return {}
    
    def _save_cache(self):
        """
        保存缓存数据
        
        将缓存数据保存到JSON文件中，使用UTF-8编码确保中文正确显示。
        
        异常处理：
            - IOError: 文件写入错误时打印错误信息
        
        示例：
            >>> self._save_cache()
            >>> print("缓存已保存")
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"❌ 保存缓存失败: {e}")
    
    def _get_file_hash(self, file_path):
        """
        获取文件的MD5哈希值
        
        用于生成缓存键，确保数据文件未修改时使用缓存。
        
        参数：
            file_path: str 或 Path
                文件路径
        
        返回：
            str: MD5哈希值（32位十六进制字符串）
            None: 文件读取失败时返回None
        
        示例：
            >>> hash_value = self._get_file_hash("data.csv")
            >>> print(f"文件哈希: {hash_value}")
        
        注意事项：
            1. 使用MD5算法计算哈希值
            2. 以二进制模式读取文件
            3. 文件不存在或读取失败时返回None
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError:
            return None
    
    def get_cache_key(self, data_file_path):
        """
        根据数据文件路径生成缓存键
        
        缓存键格式：文件名_文件哈希前8位
        例如：user_balance_table.csv_a1b2c3d4
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
        
        返回：
            str: 缓存键
            None: 文件不存在或无法生成哈希时返回None
        
        示例：
            >>> key = self.get_cache_key("data/user_balance_table.csv")
            >>> print(f"缓存键: {key}")
        
        注意事项：
            1. 检查文件是否存在
            2. 计算文件MD5哈希值
            3. 使用文件名和哈希前8位组合
        """
        file_path = Path(data_file_path)
        if not file_path.exists():
            return None
        
        file_hash = self._get_file_hash(file_path)
        if file_hash is None:
            return None
        
        # 使用文件名和哈希值作为缓存键
        return f"{file_path.name}_{file_hash[:8]}"
    
    def get_cached_params(self, data_file_path):
        """
        获取缓存的ARIMA参数
        
        根据数据文件路径获取对应的ARIMA参数缓存。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
        
        返回：
            dict: 缓存的参数信息，包含：
                - best_params: 最优参数 (p, d, q)
                - best_aic: 最优AIC值
                - total_params: 参数个数
                - data_length: 数据长度
                - param_ratio: 参数比例
                - timestamp: 缓存时间
                - data_file: 数据文件路径
                - images: 图片缓存信息（如果有）
            None: 如果没有缓存则返回None
        
        示例：
            >>> cached_info = self.get_cached_params("data.csv")
            >>> if cached_info:
            >>>     print(f"最优参数: ARIMA{cached_info['best_params']}")
            >>>     print(f"AIC: {cached_info['best_aic']}")
        """
        cache_key = self.get_cache_key(data_file_path)
        if cache_key is None:
            return None
        
        return self.cache_data.get(cache_key)
    
    def save_params(self, data_file_path, best_params, best_aic, total_params, data_length):
        """
        保存ARIMA参数到缓存
        
        将ARIMA参数搜索结果保存到缓存中，包括：
        1. 最优参数组合
        2. AIC值
        3. 参数统计信息
        4. 时间戳
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
            best_params: tuple
                最优参数 (p, d, q)
            best_aic: float
                最优AIC值
            total_params: int
                参数个数 (p + q + 1)
            data_length: int
                数据长度
        
        示例：
            >>> self.save_params(
            >>>     "data.csv",
            >>>     (2, 1, 3),
            >>>     1234.56,
            >>>     6,
            >>>     184
            >>> )
            >>> print("参数已缓存")
        
        注意事项：
            1. 自动计算参数比例
            2. 记录当前时间戳
            3. 保存完整的数据文件路径
            4. 如果缓存键生成失败会跳过保存
        """
        cache_key = self.get_cache_key(data_file_path)
        if cache_key is None:
            print("❌ 无法生成缓存键，跳过缓存")
            return
        
        cache_info = {
            'best_params': best_params,
            'best_aic': best_aic,
            'total_params': total_params,
            'data_length': data_length,
            'param_ratio': round(total_params / data_length * 100, 2),
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data_file_path)
        }
        
        self.cache_data[cache_key] = cache_info
        self._save_cache()
        print(f"✅ 参数已缓存: {cache_key}")
    
    def save_image_cache(self, data_file_path, image_type, image_path, description=""):
        """
        保存图片缓存信息
        
        将图片文件信息保存到缓存中，包括：
        1. 图片文件路径
        2. 图片类型和描述
        3. 生成时间
        4. 文件存在状态
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
            image_type: str
                图片类型 ('trend', 'prediction', 'other')
            image_path: str 或 Path
                图片文件路径
            description: str, 默认 ""
                图片描述
        
        示例：
            >>> self.save_image_cache(
            >>>     "data.csv",
            >>>     "trend",
            >>>     "output/images/trend.png",
            >>>     "用户申购和赎回金额的时间趋势图"
            >>> )
            >>> print("图片缓存已保存")
        
        注意事项：
            1. 检查文件是否存在
            2. 记录生成时间
            3. 支持多种图片类型
            4. 如果缓存键生成失败会跳过保存
        """
        cache_key = self.get_cache_key(data_file_path)
        if cache_key is None:
            print("❌ 无法生成缓存键，跳过缓存")
            return
        
        # 确保缓存键存在
        if cache_key not in self.cache_data:
            self.cache_data[cache_key] = {}
        
        # 初始化图片缓存
        if 'images' not in self.cache_data[cache_key]:
            self.cache_data[cache_key]['images'] = {}
        
        # 保存图片信息
        image_info = {
            'path': str(image_path),
            'type': image_type,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'exists': Path(image_path).exists()
        }
        
        self.cache_data[cache_key]['images'][image_type] = image_info
        self._save_cache()
        print(f"✅ 图片缓存已保存: {image_type} -> {image_path}")
    
    def get_image_cache(self, data_file_path, image_type):
        """
        获取图片缓存信息
        
        根据数据文件路径和图片类型获取对应的图片缓存信息。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
            image_type: str
                图片类型
        
        返回：
            dict: 图片缓存信息，包含：
                - path: 图片文件路径
                - type: 图片类型
                - description: 图片描述
                - timestamp: 生成时间
                - exists: 文件是否存在
            None: 如果没有缓存则返回None
        
        示例：
            >>> image_cache = self.get_image_cache("data.csv", "trend")
            >>> if image_cache:
            >>>     print(f"图片路径: {image_cache['path']}")
            >>>     print(f"文件存在: {image_cache['exists']}")
        """
        cache_key = self.get_cache_key(data_file_path)
        if cache_key is None:
            return None
        
        cached_data = self.cache_data.get(cache_key, {})
        images = cached_data.get('images', {})
        return images.get(image_type)
    
    def get_all_image_cache(self, data_file_path):
        """
        获取所有图片缓存信息
        
        获取指定数据文件的所有图片缓存信息。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
        
        返回：
            dict: 所有图片缓存信息的字典
                键为图片类型，值为图片信息字典
                如果没有图片缓存，返回空字典
        
        示例：
            >>> all_images = self.get_all_image_cache("data.csv")
            >>> for img_type, img_info in all_images.items():
            >>>     print(f"{img_type}: {img_info['path']}")
        """
        cache_key = self.get_cache_key(data_file_path)
        if cache_key is None:
            return {}
        
        cached_data = self.cache_data.get(cache_key, {})
        return cached_data.get('images', {})
    
    def check_image_exists(self, data_file_path, image_type):
        """
        检查图片文件是否存在
        
        检查指定类型的图片文件是否存在，并更新缓存中的状态。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
            image_type: str
                图片类型，如 'trend', 'prediction'
        
        返回：
            tuple: (图片路径, 是否存在)
                图片路径: str 或 None
                是否存在: bool
        
        示例：
            >>> path, exists = self.check_image_exists("data.csv", "trend")
            >>> if exists:
            >>>     print(f"图片存在: {path}")
            >>> else:
            >>>     print("图片不存在")
        
        注意事项：
            1. 自动更新缓存中的存在状态
            2. 如果状态发生变化会保存缓存
            3. 返回实际的图片路径
        """
        image_cache = self.get_image_cache(data_file_path, image_type)
        if image_cache is None:
            return None, False
        
        image_path = Path(image_cache['path'])
        exists = image_path.exists()
        
        # 更新缓存中的存在状态
        if image_cache['exists'] != exists:
            image_cache['exists'] = exists
            self._save_cache()
        
        return str(image_path), exists
    
    def save_csv_cache(self, data_file_path, csv_type, csv_path, description=""):
        """
        保存CSV文件缓存
        
        保存CSV文件的路径和相关信息到缓存中。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
            csv_type: str
                CSV类型，如 'prediction'
            csv_path: str 或 Path
                CSV文件路径
            description: str, 默认 ""
                CSV文件描述
        
        示例：
            >>> self.save_csv_cache("data.csv", "prediction", "output/data/results.csv", "预测结果")
        
        注意事项：
            1. 自动创建缓存键
            2. 保存文件存在状态
            3. 记录时间戳
        """
        cache_key = self.get_cache_key(data_file_path)
        if not cache_key:
            print("❌ 无法生成缓存键")
            return
        
        # 确保缓存记录存在
        if cache_key not in self.cache_data:
            self.cache_data[cache_key] = {}
        
        # 确保CSV缓存结构存在
        if 'csv_files' not in self.cache_data[cache_key]:
            self.cache_data[cache_key]['csv_files'] = {}
        
        # 保存CSV缓存信息
        csv_path = Path(csv_path)
        self.cache_data[cache_key]['csv_files'][csv_type] = {
            'path': str(csv_path),
            'type': csv_type,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'exists': csv_path.exists()
        }
        
        self._save_cache()
        print(f"✅ CSV缓存已保存: {csv_type}")
    
    def get_csv_cache(self, data_file_path, csv_type='prediction'):
        """
        获取CSV文件缓存
        
        获取指定类型的CSV文件缓存信息。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
            csv_type: str, 默认 'prediction'
                CSV类型
        
        返回：
            dict 或 None: CSV缓存信息
                包含路径、类型、描述、时间戳、存在状态
                如果不存在则返回None
        
        示例：
            >>> cache = self.get_csv_cache("data.csv", "prediction")
            >>> if cache:
            >>>     print(f"CSV路径: {cache['path']}")
        
        注意事项：
            1. 自动检查文件存在状态
            2. 返回完整的缓存信息
            3. 如果缓存不存在返回None
        """
        cache_key = self.get_cache_key(data_file_path)
        if not cache_key or cache_key not in self.cache_data:
            return None
        
        cache_info = self.cache_data[cache_key]
        if 'csv_files' not in cache_info or csv_type not in cache_info['csv_files']:
            return None
        
        csv_cache = cache_info['csv_files'][csv_type]
        
        # 检查文件是否存在并更新状态
        csv_path = Path(csv_cache['path'])
        exists = csv_path.exists()
        if csv_cache['exists'] != exists:
            csv_cache['exists'] = exists
            self._save_cache()
        
        return csv_cache
    
    def get_all_csv_cache(self, data_file_path):
        """
        获取所有CSV文件缓存
        
        获取指定数据文件的所有CSV缓存信息。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
        
        返回：
            dict: 所有CSV缓存信息
                键为CSV类型，值为缓存信息
        
        示例：
            >>> all_csv = self.get_all_csv_cache("data.csv")
            >>> for csv_type, info in all_csv.items():
            >>>     print(f"{csv_type}: {info['path']}")
        
        注意事项：
            1. 返回所有CSV类型的缓存
            2. 自动更新文件存在状态
            3. 如果没有缓存返回空字典
        """
        cache_key = self.get_cache_key(data_file_path)
        if not cache_key or cache_key not in self.cache_data:
            return {}
        
        cache_info = self.cache_data[cache_key]
        if 'csv_files' not in cache_info:
            return {}
        
        # 更新所有CSV文件的存在状态
        for csv_type, csv_cache in cache_info['csv_files'].items():
            csv_path = Path(csv_cache['path'])
            exists = csv_path.exists()
            if csv_cache['exists'] != exists:
                csv_cache['exists'] = exists
        
        # 如果有状态变化，保存缓存
        if any(csv_cache['exists'] != Path(csv_cache['path']).exists() 
               for csv_cache in cache_info['csv_files'].values()):
            self._save_cache()
        
        return cache_info['csv_files']
    
    def clear_cache(self, data_file_path=None):
        """
        清除缓存
        
        清除指定文件或所有文件的缓存。
        
        参数：
            data_file_path: str 或 Path, 默认 None
                指定文件路径，如果为None则清除所有缓存
        
        示例：
            >>> # 清除指定文件的缓存
            >>> self.clear_cache("data.csv")
            >>> 
            >>> # 清除所有缓存
            >>> self.clear_cache()
        
        注意事项：
            1. 清除指定文件缓存时会验证缓存键
            2. 清除所有缓存会清空整个缓存字典
            3. 操作完成后会自动保存缓存文件
        """
        if data_file_path is None:
            # 清除所有缓存
            self.cache_data.clear()
            print("🗑️  已清除所有缓存")
        else:
            # 清除指定文件的缓存
            cache_key = self.get_cache_key(data_file_path)
            if cache_key and cache_key in self.cache_data:
                del self.cache_data[cache_key]
                print(f"🗑️  已清除缓存: {cache_key}")
            else:
                print("ℹ️  未找到对应的缓存记录")
        
        self._save_cache()
    
    def list_cache(self):
        """
        列出所有缓存记录
        
        显示所有缓存记录的详细信息，包括：
        1. ARIMA参数信息
        2. 图片缓存信息
        3. 缓存时间
        4. 文件路径
        
        示例：
            >>> self.list_cache()
            >>> # 输出所有缓存记录
        
        注意事项：
            1. 包含安全检查，避免KeyError
            2. 分别显示ARIMA参数和图片缓存
            3. 格式化输出，便于阅读
        """
        if not self.cache_data:
            print("📭 暂无缓存记录")
            return
        
        print("📋 缓存记录列表:")
        print("=" * 80)
        for cache_key, info in self.cache_data.items():
            # 安全检查：确保缓存信息是字典类型
            if not isinstance(info, dict):
                print(f"⚠️  缓存记录格式错误: {cache_key}")
                continue
            
            # 检查是否包含ARIMA参数信息
            if 'best_params' in info and 'best_aic' in info:
                print(f"文件: {info.get('data_file', 'Unknown')}")
                print(f"参数: ARIMA{info['best_params']}")
                print(f"AIC: {info['best_aic']:.2f}")
                print(f"参数个数: {info.get('total_params', 'N/A')} ({info.get('param_ratio', 'N/A')}%)")
                print(f"缓存时间: {info.get('timestamp', 'Unknown')}")
                
                # 检查是否有图片缓存
                if 'images' in info and info['images']:
                    print("图片缓存:")
                    for img_type, img_info in info['images'].items():
                        if isinstance(img_info, dict):
                            print(f"  - {img_type}: {img_info.get('path', 'Unknown')}")
                
                # 检查是否有CSV缓存
                if 'csv_files' in info and info['csv_files']:
                    print("CSV缓存:")
                    for csv_type, csv_info in info['csv_files'].items():
                        if isinstance(csv_info, dict):
                            print(f"  - {csv_type}: {csv_info.get('path', 'Unknown')}")
                
                print("-" * 40)
            else:
                # 只包含图片缓存的记录
                if 'images' in info and info['images']:
                    print(f"文件: {info.get('data_file', 'Unknown')}")
                    print("图片缓存:")
                    for img_type, img_info in info['images'].items():
                        if isinstance(img_info, dict):
                            print(f"  - {img_type}: {img_info.get('path', 'Unknown')}")
                    print("-" * 40)
                
                # 只包含CSV缓存的记录
                if 'csv_files' in info and info['csv_files']:
                    print(f"文件: {info.get('data_file', 'Unknown')}")
                    print("CSV缓存:")
                    for csv_type, csv_info in info['csv_files'].items():
                        if isinstance(csv_info, dict):
                            print(f"  - {csv_type}: {csv_info.get('path', 'Unknown')}")
                    print("-" * 40)
    
    def is_cache_valid(self, data_file_path):
        """
        检查缓存是否有效（文件是否被修改）
        
        通过比较文件哈希值来判断缓存是否仍然有效。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
        
        返回：
            bool: 缓存是否有效
        
        示例：
            >>> if self.is_cache_valid("data.csv"):
            >>>     print("缓存有效，可以使用")
            >>> else:
            >>>     print("缓存无效，需要重新计算")
        
        注意事项：
            1. 基于文件哈希值判断有效性
            2. 文件不存在时返回False
            3. 缓存键不存在时返回False
        """
        cache_key = self.get_cache_key(data_file_path)
        if cache_key is None:
            return False
        
        return cache_key in self.cache_data
    
    def refresh_cache(self):
        """
        刷新缓存数据（重新从文件加载）
        
        重新从缓存文件中加载数据，确保获取最新的缓存信息。
        
        返回：
            dict: 刷新后的缓存数据
        
        示例：
            >>> cache_data = self.refresh_cache()
            >>> print(f"刷新后缓存记录数: {len(cache_data)}")
        
        注意事项：
            1. 重新读取缓存文件
            2. 处理文件读取错误
            3. 返回最新的缓存数据
        """
        self.cache_data = self._load_cache()
        return self.cache_data
    
    def get_cache_summary(self, data_file_path):
        """
        获取缓存摘要信息（用于显示在菜单中）
        
        生成简洁的缓存摘要，用于在菜单中显示。
        
        参数：
            data_file_path: str 或 Path
                数据文件路径
        
        返回：
            str: 摘要字符串，格式如：
                "📋 ARIMA(2,1,3) (AIC:1234.5, 参数:6, 2.1%)"
            None: 如果没有缓存则返回None
        
        示例：
            >>> summary = self.get_cache_summary("data.csv")
            >>> if summary:
            >>>     print(f"缓存摘要: {summary}")
        
        注意事项：
            1. 先刷新缓存确保最新信息
            2. 包含安全检查避免KeyError
            3. 格式化输出便于显示
            4. 包含参数比例信息
        """
        # 先刷新缓存，确保获取最新信息
        self.refresh_cache()
        
        cached_info = self.get_cached_params(data_file_path)
        if cached_info is None:
            return None
        
        # 安全检查：确保缓存信息包含必要的字段
        if not isinstance(cached_info, dict):
            return None
        
        # 检查是否包含ARIMA参数信息
        if 'best_params' not in cached_info or 'best_aic' not in cached_info:
            return None
        
        params = cached_info['best_params']
        aic = cached_info['best_aic']
        total_params = cached_info.get('total_params', 0)
        param_ratio = cached_info.get('param_ratio', 0)
        
        return f"📋 ARIMA{params} (AIC:{aic:.1f}, 参数:{total_params}, {param_ratio}%)"

# 全局缓存管理器实例
cache_manager = CacheManager() 