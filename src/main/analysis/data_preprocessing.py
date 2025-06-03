"""
Author: <WU Xinyan>
数据预处理模块
"""
import pandas as pd
import warnings
from datetime import timedelta
from utils import get_column_names
from config import SCADA_DATA_PATH, FAULT_DATA_PATH, WINDOW_BEFORE, WINDOW_AFTER
from logging_utils import get_logger

warnings.filterwarnings('ignore')

def load_data(logger=None):
    if logger is None:
        logger = get_logger('data_loader')

    logger.info("正在加载SCADA数据和故障数据...")

    try:
        logger.debug(f"尝试从 {SCADA_DATA_PATH} 加载SCADA数据")
        scada_data = pd.read_csv(SCADA_DATA_PATH)
        logger.info(f"SCADA数据加载成功: {scada_data.shape[0]}行, {scada_data.shape[1]}列")
    except FileNotFoundError as e:
        logger.error(f"SCADA数据加载失败: {str(e)}")
        raise

    try:
        logger.debug(f"尝试从 {FAULT_DATA_PATH} 加载故障数据")
        fault_data = pd.read_csv(FAULT_DATA_PATH)
        logger.info(f"故障数据加载成功: {fault_data.shape[0]}行, {fault_data.shape[1]}列")
    except FileNotFoundError as e:
        logger.error(f"故障数据加载失败: {str(e)}")
        raise

    logger.info(f"SCADA数据行数: {scada_data.shape[0]}, 列数: {scada_data.shape[1]}")
    logger.info(f"故障数据行数: {fault_data.shape[0]}, 列数: {fault_data.shape[1]}")

    return scada_data, fault_data


def preprocess_data(scada_data, fault_data, logger=None):
    if logger is None:
        logger = get_logger('preprocessor')

    logger.info("开始数据预处理...")

    scada_data = scada_data.copy()
    fault_data = fault_data.copy()

    if 'DateTime' not in scada_data.columns:
        logger.warning("SCADA数据中未找到DateTime列，尝试创建...")
        try:
            scada_data['DateTime'] = pd.to_datetime(scada_data.iloc[:, 0], format='%m/%d/%Y %H:%M', errors='coerce')
            logger.info("成功从第一列创建DateTime列")
        except Exception as e:
            logger.warning(f"从第一列创建DateTime列失败: {str(e)}")
            try:
                scada_data['DateTime'] = pd.to_datetime(scada_data.iloc[:, 0], errors='coerce')
                logger.info("成功使用通用格式从第一列创建DateTime列")
            except Exception as e:
                logger.error(f"无法创建DateTime列: {str(e)}")
                logger.warning("创建替代时间戳...")
                # 创建一个假的时间序列作为替代
                scada_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(scada_data), freq='10min')
                logger.info("已创建替代时间序列")

            # 确认DateTime列不为空
        if scada_data['DateTime'].isna().all():
            logger.warning("所有日期时间解析失败。创建替代时间戳...")
            scada_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(scada_data), freq='10min')
            logger.info("已创建替代时间序列")

        if not pd.api.types.is_datetime64_any_dtype(scada_data['DateTime']):
            logger.info("将SCADA数据中的DateTime列转换为datetime类型...")
            scada_data['DateTime'] = pd.to_datetime(scada_data['DateTime'], errors='coerce')

        if 'DateTime' not in fault_data.columns and 'datetime' in fault_data.columns:
            fault_data = fault_data.rename(columns={'datetime': 'DateTime'})
            logger.info("故障数据中的'datetime'列已重命名为'DateTime'")

    # 处理故障数据中的时间
    if 'DateTime' not in fault_data.columns:
        logger.warning("故障数据中未找到DateTime列，尝试创建...")
        try:
            fault_data['DateTime'] = pd.to_datetime(fault_data.iloc[:, 0], errors='coerce')
            logger.info("成功从第一列创建DateTime列")
        except Exception as e:
            logger.warning(f"从第一列创建DateTime列失败: {str(e)}")
            if 'Time' in fault_data.columns:
                try:
                    fault_data['DateTime'] = pd.to_datetime(fault_data['Time'], unit='s')
                    logger.info("使用Time列创建了DateTime列")
                except Exception as e:
                    logger.error(f"无法从Time列创建DateTime: {str(e)}")
            else:
                logger.error("无法为故障数据创建DateTime列。这可能导致后续问题。")
                fault_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(fault_data), freq='1D')
                logger.info("已创建替代时间序列")

    if not pd.api.types.is_datetime64_any_dtype(fault_data['DateTime']):
        logger.info("将故障数据中的DateTime列转换为datetime类型...")
        fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'], errors='coerce')

    logger.info(f"SCADA数据DateTime列类型: {scada_data['DateTime'].dtype}")
    logger.info(f"故障数据DateTime列类型: {fault_data['DateTime'].dtype}")

    column_names = get_column_names()
    datetime_col_exists = 'DateTime' in scada_data.columns

    if scada_data.shape[1] == len(column_names) - (1 if datetime_col_exists else 0):
        new_columns = column_names.copy()
        if datetime_col_exists:
            new_columns.remove('DateTime')
            current_cols = list(scada_data.columns)
            datetime_idx = current_cols.index('DateTime')
            remaining_cols = current_cols[:datetime_idx] + current_cols[datetime_idx + 1:]
            final_columns = {}
            for old_col, new_col in zip(remaining_cols, new_columns):
                final_columns[old_col] = new_col
            scada_data.rename(columns=final_columns, inplace=True)
        else:
            scada_data.columns = new_columns
    else:
        # 列数不匹配，生成通用列名但保留DateTime
        logger.warning(f"列名数量不匹配。生成通用列名...")
        if datetime_col_exists:
            current_cols = list(scada_data.columns)
            datetime_idx = current_cols.index('DateTime')
            generic_columns = ['col_' + str(i) for i in range(scada_data.shape[1])]
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

    common_columns = ['Time', 'WindSpeed', 'Power', 'Temperature', 'Vibration']
    for common_col in common_columns:
        matching_cols = [col for col in scada_data.columns if common_col.lower() in col.lower()]
        if matching_cols and common_col not in scada_data.columns:
            scada_data.rename(columns={matching_cols[0]: common_col}, inplace=True)

    # 数据清洗
    scada_missing = scada_data.isnull().sum().sum()
    fault_missing = fault_data.isnull().sum().sum()

    logger.info(f"SCADA数据中的缺失值数量: {scada_missing}")
    logger.info(f"故障数据中的缺失值数量: {fault_missing}")

    # 处理缺失值
    if 'DateTime' in scada_data.columns:
        scada_data = scada_data.dropna(subset=['DateTime'])
    if 'DateTime' in fault_data.columns:
        fault_data = fault_data.dropna(subset=['DateTime'])

    # 检查并处理重复值
    scada_duplicates = scada_data.duplicated().sum()
    fault_duplicates = fault_data.duplicated().sum()

    logger.info(f"SCADA数据中的重复行数量: {scada_duplicates}")
    logger.info(f"故障数据中的重复行数量: {fault_duplicates}")

    # 删除重复行
    if scada_duplicates > 0:
        scada_data.drop_duplicates(inplace=True)
    if fault_duplicates > 0:
        fault_data.drop_duplicates(inplace=True)

    if 'DateTime' not in scada_data.columns:
        logger.warning("错误: SCADA数据中缺少DateTime列。请检查数据格式或预处理步骤。")
        scada_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(scada_data), freq='10min')

    if 'DateTime' not in fault_data.columns:
        logger.warning("错误: 故障数据中缺少DateTime列。请检查数据格式或预处理步骤。")
        fault_data['DateTime'] = pd.date_range(start='2014-01-01', periods=len(fault_data), freq='1D')

    # 匹配故障时间与SCADA记录
    try:
        scada_data.sort_values('DateTime', inplace=True)
        fault_data.sort_values('DateTime', inplace=True)

        scada_min_time = scada_data['DateTime'].min()
        scada_max_time = scada_data['DateTime'].max()

        logger.info(f"SCADA数据时间范围: {scada_min_time} 至 {scada_max_time}")

        if not pd.api.types.is_datetime64_any_dtype(fault_data['DateTime']):
            fault_data['DateTime'] = pd.to_datetime(fault_data['DateTime'], errors='coerce')

        valid_faults = fault_data[(fault_data['DateTime'] >= scada_min_time) &
                                  (fault_data['DateTime'] <= scada_max_time)]

        logger.info(f"在SCADA数据范围内的故障数量: {len(valid_faults)} (总数: {len(fault_data)})")

    except KeyError as e:
        logger.warning(f"排序或筛选错误: {e}。检查DataFrame的列...")
        logger.info("SCADA数据列:", scada_data.columns.tolist())
        logger.info("故障数据列:", fault_data.columns.tolist())
        # 为避免程序崩溃，使用所有故障数据继续
        valid_faults = fault_data.copy()

    # 创建带故障标签的数据
    labeled_data = scada_data.copy()
    labeled_data['Fault'] = 'NoFault'

    window_before = timedelta(hours=WINDOW_BEFORE)
    window_after = timedelta(hours=WINDOW_AFTER)

    fault_mapping = {}

    for _, fault_row in fault_data.iterrows():
        fault_time = fault_row['DateTime']
        fault_type = fault_row['Fault'] if 'Fault' in fault_row else 'Unknown'

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
    logger.info("\n标记后的故障分布:")
    print(fault_distribution)

    return labeled_data, fault_distribution