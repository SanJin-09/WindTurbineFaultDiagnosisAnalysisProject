"""
Author: <WU Xinyan>
数据预处理模块
"""
import pandas as pd
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

from utils import get_column_names
from config import SCADA_DATA_PATH, FAULT_DATA_PATH, WINDOW_BEFORE, WINDOW_AFTER


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
    try:
        scada_data['DateTime'] = pd.to_datetime(scada_data.iloc[:, 0], format='%m/%d/%Y %H:%M')
    except:
        try:
            scada_data['DateTime'] = pd.to_datetime(scada_data.iloc[:, 0])
        except:
            print("无法解析SCADA数据的时间格式。使用通用解析方法...")
            scada_data['DateTime'] = pd.to_datetime(scada_data.iloc[:, 0], errors='coerce')

    try:
        fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'])
    except:
        print("无法解析故障数据的时间格式。使用通用解析方法...")
        fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'], errors='coerce')

    # 2. 创建合适的列名
    column_names = get_column_names()
    if scada_data.shape[1] + 1 == len(column_names):
        new_columns = column_names.copy()
        new_columns.remove('DateTime')
        scada_data.columns = [scada_data.columns[0]] + new_columns
    else:
        print(f"列名数量不匹配。生成通用列名...")
        generic_columns = ['col_' + str(i) for i in range(scada_data.shape[1])]
        generic_columns[0] = 'DateTime_str'
        scada_data.columns = generic_columns
        scada_data.rename(columns={'col_1': 'Time'}, inplace=True)

    # 3. 数据清洗
    scada_missing = scada_data.isnull().sum().sum()
    fault_missing = fault_data.isnull().sum().sum()

    print(f"SCADA数据中的缺失值数量: {scada_missing}")
    print(f"故障数据中的缺失值数量: {fault_missing}")

    scada_data.dropna(inplace=True)
    fault_data.dropna(inplace=True)

    scada_duplicates = scada_data.duplicated().sum()
    fault_duplicates = fault_data.duplicated().sum()

    print(f"SCADA数据中的重复行数量: {scada_duplicates}")
    print(f"故障数据中的重复行数量: {fault_duplicates}")

    # 删除重复行
    if scada_duplicates > 0:
        scada_data.drop_duplicates(inplace=True)
    if fault_duplicates > 0:
        fault_data.drop_duplicates(inplace=True)

    # 4. 匹配故障时间与SCADA记录
    scada_data.sort_values('DateTime', inplace=True)
    fault_data.sort_values('DateTime', inplace=True)

    scada_min_time = scada_data['DateTime'].min()
    scada_max_time = scada_data['DateTime'].max()

    valid_faults = fault_data[(fault_data['DateTime'] >= scada_min_time) &
                              (fault_data['DateTime'] <= scada_max_time)]

    print(f"SCADA数据时间范围: {scada_min_time} 至 {scada_max_time}")
    print(f"在SCADA数据范围内的故障数量: {len(valid_faults)} (总数: {len(fault_data)})")

    # 5. 创建带故障标签的数据
    labeled_data = scada_data.copy()
    labeled_data['Fault'] = 'NoFault' # 默认值

    window_before = timedelta(hours=WINDOW_BEFORE)
    window_after = timedelta(hours=WINDOW_AFTER)

    fault_mapping = {}

    for _, fault_row in fault_data.iterrows():
        fault_time = fault_row['DateTime']
        fault_type = fault_row['Fault']

        mask = ((scada_data['DateTime'] >= fault_time - window_before) &
                (scada_data['DateTime'] <= fault_time + window_after))

        for idx in scada_data[mask].index:
            if idx in fault_mapping:
                fault_mapping[idx].append(fault_type)
            else:
                fault_mapping[idx] = [fault_type]

    for idx, fault_types in fault_mapping.items():
        labeled_data.at[idx, 'Fault'] = fault_types[0]

    fault_distribution = labeled_data['Fault'].value_counts()
    print("\n标记后的故障分布:")
    print(fault_distribution)

    return labeled_data, fault_distribution
