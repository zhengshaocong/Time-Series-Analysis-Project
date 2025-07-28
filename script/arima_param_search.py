

# 网格搜索最优ARIMA参数

import warnings
warnings.filterwarnings('ignore')

import sys
import os
# 添加上级目录到Python路径，以便导入utils模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import arima_grid_search
from utils.cache_manager import cache_manager
from config import get_data_file_path

def main():
    # 读取数据
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

    # 构造时间序列索引，明确设置频率以减少警告
    dates = pd.to_datetime(trend['report_date'], format='%Y%m%d')
    ts = pd.Series(trend['total_purchase_amt'].values, index=dates)
    ts = ts.asfreq('D')  # 明确设置为日频率

    # 训练集：2014年3月1日~2014年8月31日
    ts_train = ts[(ts.index >= '2014-03-01') & (ts.index <= '2014-08-31')]

    # 计算数据量并设置最大参数量限制
    data_length = len(ts_train)
    max_params = min(10, int(data_length * 0.05))  # 最多10个参数，或数据量的5%

    print(f"数据量: {data_length} 天")
    print(f"最大参数量限制: {max_params} 个参数")
    print(f"参数比例: {max_params/data_length*100:.1f}%")

    # 设置ARIMA参数搜索范围
    p_range = range(0, 10)  # p的范围
    d_range = range(0, 2)   # d的范围
    q_range = range(0, 10)  # q的范围

    # 网格搜索最优ARIMA参数，限制最大参数量
    best_params, best_model = arima_grid_search(ts_train, p_range, d_range, q_range, max_params=max_params)

    print(f'\n最终结果:')
    print(f'最优ARIMA参数: {best_params}')
    if best_params:
        total_params = best_params[0] + best_params[2] + 1
        print(f'实际参数个数: {total_params}')
        print(f'参数比例: {total_params/data_length*100:.1f}%')
        
        # 保存到缓存
        if best_model is not None:
            best_aic = best_model.aic
            cache_manager.save_params(
                file_path, 
                best_params, 
                best_aic, 
                total_params, 
                data_length
            )
        else:
            print("⚠️  模型拟合失败，无法保存到缓存")
    else:
        print("❌ 未找到有效的ARIMA参数组合")

if __name__ == "__main__":
    main() 