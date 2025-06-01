"""
Author: <WU Xinyan>
数据预处理模块
"""
import pandas as pd
import warnings
from datetime import timedelta
from utils import get_column_names
from config import SCADA_DATA_PATH, FAULT_DATA_PATH, WINDOW_BEFORE, WINDOW_AFTER

warnings.filterwarnings('ignore')

def load_data():
    print("正在加载SCADA数据和故障数据...")

    # 加载SCADA数据
    scada_data = pd.read_csv(SCADA_DATA_PATH)

    # 加载故障数据
    fault_data = pd.read_csv(FAULT_DATA_PATH)

    print(f"SCADA数据行数: {scada_data.shape[0]}, 列数: {scada_data.shape[1]}")
    print(f"故障数据行数: {fault_data.shape[0]}, 列数: {fault_data.shape[1]}")

    return scada_data, fault_data


def preprocess_data(scada_data, fault_data):
    print("\n正在进行数据预处理...")

    scada_data = scada_data.copy()  # 创建副本以避免修改原始数据
    fault_data = fault_data.copy()

    # 检查是否已有DateTime列
    if 'DateTime' not in scada_data.columns:
        print("SCADA数据中未找到DateTime列，尝试创建...")
        try:
            # 尝试将第一列解析为日期时间
            scada_data['DateTime'] = pd.to_datetime(scada_data.iloc[:, 0], errors='coerce')
            print("成功从第一列创建DateTime列")
        except Exception as e:
            print(f"从第一列创建DateTime列失败: {str(e)}")
            # 如果无法从第一列创建，创建一个假的时间序列
            print("创建替代时间戳...")
            scada_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(scada_data), freq='10min')
    else:
        # 确保DateTime列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(scada_data['DateTime']):
            print("将SCADA数据中的DateTime列转换为datetime类型...")
            scada_data['DateTime'] = pd.to_datetime(scada_data['DateTime'], errors='coerce')

    # 处理故障数据中的时间
    if 'DateTime' not in fault_data.columns:
        if 'datetime' in fault_data.columns:  # 检查小写版本
            fault_data = fault_data.rename(columns={'datetime': 'DateTime'})
            print("故障数据中的'datetime'列已重命名为'DateTime'")
        else:
            # 尝试从第一列创建
            try:
                fault_data['DateTime'] = pd.to_datetime(fault_data.iloc[:, 0], errors='coerce')
                print("成功从第一列为故障数据创建DateTime列")
            except Exception as e:
                print(f"为故障数据创建DateTime列失败: {str(e)}")
                # 如果无法从第一列创建，创建一个假的时间序列
                print("为故障数据创建替代时间戳...")
                fault_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(fault_data), freq='1D')

    # 确保故障数据的DateTime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(fault_data['DateTime']):
        print("将故障数据中的DateTime列转换为datetime类型...")
        fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'], errors='coerce')

    # 打印转换后的数据类型
    print(f"SCADA数据DateTime列类型: {scada_data['DateTime'].dtype}")
    print(f"故障数据DateTime列类型: {fault_data['DateTime'].dtype}")

    # 2. 创建合适的列名（保留DateTime列）
    column_names = get_column_names()

    # 检查列数与列名数是否匹配
    datetime_col_exists = 'DateTime' in scada_data.columns

    if scada_data.shape[1] == len(column_names) - (1 if datetime_col_exists else 0):
        # 列数匹配
        new_columns = column_names.copy()
        if datetime_col_exists:
            # 如果已经有DateTime列，从新列名列表中移除
            new_columns.remove('DateTime')
            # 保留当前列名中的DateTime
            current_cols = list(scada_data.columns)
            datetime_idx = current_cols.index('DateTime')
            remaining_cols = current_cols[:datetime_idx] + current_cols[datetime_idx + 1:]
            # 合并列名
            final_columns = {}
            for old_col, new_col in zip(remaining_cols, new_columns):
                final_columns[old_col] = new_col
            # 重命名
            scada_data.rename(columns=final_columns, inplace=True)
        else:
            # 直接使用列名
            scada_data.columns = new_columns
    else:
        # 列数不匹配，生成通用列名但保留DateTime
        print(f"列名数量不匹配。生成通用列名...")
        if datetime_col_exists:
            # 保留DateTime列
            current_cols = list(scada_data.columns)
            datetime_idx = current_cols.index('DateTime')
            # 生成其他列的通用名称
            generic_columns = ['col_' + str(i) for i in range(scada_data.shape[1])]
            # 将DateTime放回原位置
            generic_columns[datetime_idx] = 'DateTime'
            scada_data.columns = generic_columns
        else:
            # 完全生成通用列名
            generic_columns = ['col_' + str(i) for i in range(scada_data.shape[1])]
            scada_data.columns = generic_columns
            # 将第一列作为DateTime (假设第一列是日期时间)
            scada_data.rename(columns={'col_0': 'DateTime_str'}, inplace=True)
            # 确保存在DateTime列
            if 'DateTime' not in scada_data.columns:
                scada_data['DateTime'] = pd.to_datetime(scada_data['DateTime_str'], errors='coerce')

    # 尝试保留或识别常见的列名
    common_columns = ['Time', 'WindSpeed', 'Power', 'Temperature', 'Vibration']
    for common_col in common_columns:
        # 查找可能匹配的列
        matching_cols = [col for col in scada_data.columns if common_col.lower() in col.lower()]
        if matching_cols and common_col not in scada_data.columns:
            # 重命名第一个匹配的列
            scada_data.rename(columns={matching_cols[0]: common_col}, inplace=True)

    # 3. 数据清洗
    # 检查缺失值
    scada_missing = scada_data.isnull().sum().sum()
    fault_missing = fault_data.isnull().sum().sum()

    print(f"SCADA数据中的缺失值数量: {scada_missing}")
    print(f"故障数据中的缺失值数量: {fault_missing}")

    # 处理缺失值（这里使用简单的删除策略，可根据需要调整）
    # 仅删除DateTime列为空的行
    if 'DateTime' in scada_data.columns:
        scada_data = scada_data.dropna(subset=['DateTime'])
    if 'DateTime' in fault_data.columns:
        fault_data = fault_data.dropna(subset=['DateTime'])

    # 检查并处理重复值
    scada_duplicates = scada_data.duplicated().sum()
    fault_duplicates = fault_data.duplicated().sum()

    print(f"SCADA数据中的重复行数量: {scada_duplicates}")
    print(f"故障数据中的重复行数量: {fault_duplicates}")

    # 删除重复行
    if scada_duplicates > 0:
        scada_data.drop_duplicates(inplace=True)
    if fault_duplicates > 0:
        fault_data.drop_duplicates(inplace=True)

    # 确保DateTime列存在并且可排序
    if 'DateTime' not in scada_data.columns:
        print("错误: SCADA数据中缺少DateTime列。请检查数据格式或预处理步骤。")
        # 创建一个临时的DateTime列以允许代码继续运行
        scada_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(scada_data), freq='10min')

    if 'DateTime' not in fault_data.columns:
        print("错误: 故障数据中缺少DateTime列。请检查数据格式或预处理步骤。")
        # 创建一个临时的DateTime列以允许代码继续运行
        fault_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(fault_data), freq='1D')

    # 4. 匹配故障时间与SCADA记录
    # 确保两个数据集都是有序的
    try:
        scada_data.sort_values('DateTime', inplace=True)
        fault_data.sort_values('DateTime', inplace=True)

        # 查看哪些故障时间在SCADA数据的时间范围内
        scada_min_time = scada_data['DateTime'].min()
        scada_max_time = scada_data['DateTime'].max()

        print(f"SCADA数据时间范围: {scada_min_time} 至 {scada_max_time}")

        # 进行比较前再次确认类型
        if not pd.api.types.is_datetime64_any_dtype(fault_data['DateTime']):
            fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'], errors='coerce')

        valid_faults = fault_data[(fault_data['DateTime'] >= scada_min_time) &
                                  (fault_data['DateTime'] <= scada_max_time)]

        print(f"在SCADA数据范围内的故障数量: {len(valid_faults)} (总数: {len(fault_data)})")

    except KeyError as e:
        print(f"排序或筛选错误: {e}。检查DataFrame的列...")
        print("SCADA数据列:", scada_data.columns.tolist())
        print("故障数据列:", fault_data.columns.tolist())
        # 为避免程序崩溃，使用所有故障数据继续
        valid_faults = fault_data.copy()

    # 5. 创建带故障标签的数据
    labeled_data = scada_data.copy()
    labeled_data['Fault'] = 'NoFault'  # 默认无故障

    # 转换窗口时间为timedelta对象
    window_before = timedelta(hours=WINDOW_BEFORE)
    window_after = timedelta(hours=WINDOW_AFTER)

    # 初始化故障映射字典
    fault_mapping = {}

    # 为每个故障时间点找到对应的SCADA数据时间窗口
    for _, fault_row in fault_data.iterrows():
        fault_time = fault_row['DateTime']
        fault_type = fault_row['Fault'] if 'Fault' in fault_row else 'Unknown'

        # 找出在故障窗口期内的所有SCADA记录
        mask = ((scada_data['DateTime'] >= fault_time - window_before) &
                (scada_data['DateTime'] <= fault_time + window_after))

        # 将这些记录的索引添加到故障映射中
        for idx in scada_data[mask].index:
            if idx in fault_mapping:
                fault_mapping[idx].append(fault_type)
            else:
                fault_mapping[idx] = [fault_type]

    # 将故障映射应用到标记数据中
    for idx, fault_types in fault_mapping.items():
        labeled_data.at[idx, 'Fault'] = fault_types[0]

    # 检查数据标记情况
    fault_distribution = labeled_data['Fault'].value_counts()
    print("\n标记后的故障分布:")
    print(fault_distribution)

    return labeled_data, fault_distribution