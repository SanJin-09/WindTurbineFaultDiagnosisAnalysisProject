"""
Author: <WU Xinyan>
config包含所有配置参数
谨慎更改!!
"""
import os

CONFIG_FILE_PATH = os.path.abspath(__file__)
# 获取 src 目录路径
SRC_DIR = os.path.dirname(CONFIG_FILE_PATH)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
# 构建数据目录路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 文件路径使用绝对路径
SCADA_DATA_PATH = os.path.join(DATA_DIR, 'scada_data.csv')
FAULT_DATA_PATH = os.path.join(DATA_DIR, 'fault_data.csv')

# 数据预处理参数
WINDOW_BEFORE = 2  # 故障前时间窗口（hour）
WINDOW_AFTER = 0.17  # 故障后时间窗口（hour）

# 特征工程参数
WINDOW_SIZE = 5
IMPORTANCE_THRESHOLD = 0.01

# 模型训练参数
TEST_SIZE = 0.3
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
GB_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
MLP_HIDDEN_LAYERS = (100,)
MLP_MAX_ITER = 1000

# 路径
MODEL_PATH = '../report/best_turbine_fault_model.pkl'
SCALER_PATH = '../report/feature_scaler.pkl'
SELECTOR_PATH = '../report/feature_selector.pkl'
ENCODER_PATH = '../report/label_encoder.pkl'
