"""
Author: <WU Xinyan>
config包含所有配置参数
谨慎更改!!
"""
# 文件路径
SCADA_DATA_PATH = 'scada_data.csv'
FAULT_DATA_PATH = 'fault_data.csv'

# 数据预处理参数
WINDOW_BEFORE = 2  # 故障前时间窗口（hour）
WINDOW_AFTER = 0.17  # 故障后时间窗口（hour）

# 特征工程参数
WINDOW_SIZE = 5  # 滑动窗口大小
IMPORTANCE_THRESHOLD = 0.01  # 阈值

# 模型训练参数
TEST_SIZE = 0.3
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
GB_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
MLP_HIDDEN_LAYERS = (100,)
MLP_MAX_ITER = 1000

# 路径
MODEL_PATH = 'best_turbine_fault_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
SELECTOR_PATH = 'feature_selector.pkl'
ENCODER_PATH = 'label_encoder.pkl'
