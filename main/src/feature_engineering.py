"""
Author: <WU Xinyan>
特征工程模块
极化，差分，筛选特征
特殊情况下会有bug(?)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from config import WINDOW_SIZE, IMPORTANCE_THRESHOLD


def extract_features(labeled_data):
    print("\n正在进行特征工程...")

    labeled_data.sort_values('DateTime', inplace=True)

    numeric_columns = labeled_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if 'Time' in numeric_columns:
        numeric_columns.remove('Time')
    features = pd.DataFrame()

    for col in numeric_columns:
        features[col] = labeled_data[col]

    for col in numeric_columns:
        features[f'{col}_mean'] = labeled_data[col].rolling(window=WINDOW_SIZE, min_periods=1).mean()

        features[f'{col}_std'] = labeled_data[col].rolling(window=WINDOW_SIZE, min_periods=1).std()

        features[f'{col}_min'] = labeled_data[col].rolling(window=WINDOW_SIZE, min_periods=1).min()

        features[f'{col}_max'] = labeled_data[col].rolling(window=WINDOW_SIZE, min_periods=1).max()
        features[f'{col}_median'] = labeled_data[col].rolling(window=WINDOW_SIZE, min_periods=1).median()

        # 依据速率对特征进行差分
        features[f'{col}_diff'] = labeled_data[col].diff()

        features[f'{col}_diff_pct'] = features[f'{col}_diff'] / (features[col].shift(1) + 1e-10) * 100

    if 'DateTime' in labeled_data.columns:
        features['hour'] = labeled_data['DateTime'].dt.hour
        features['day'] = labeled_data['DateTime'].dt.day
        features['month'] = labeled_data['DateTime'].dt.month
        features['weekday'] = labeled_data['DateTime'].dt.weekday

    features['Fault'] = labeled_data['Fault']

    # 删除空值
    features.dropna(inplace=True)

    print(f"创建的特征数量: {features.shape[1] - 1}")
    print(f"特征数据行数: {features.shape[0]}")

    return features


def prepare_features(features):
    le = LabelEncoder()
    y = le.fit_transform(features['Fault'])
    x = features.drop('Fault', axis=1)

    print(f"分类标签编码:")
    for i, label in enumerate(le.classes_):
        print(f"{i}: {label}")

    return x, y, le


def select_features(x_train, y_train, x_test):
    print("\n正在进行特征选择...")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 随机森林
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(x_train_scaled, y_train)

    importance = selector.feature_importances_
    indices = np.argsort(importance)[::-1]

    # 打印
    print("最重要的10个特征:")
    important_features = []
    for i in range(min(10, len(indices))):
        feat_name = x_train.columns[indices[i]]
        important_features.append(feat_name)
        print(f"{feat_name}: {importance[indices[i]]:.4f}")

    sfm = SelectFromModel(selector, threshold=IMPORTANCE_THRESHOLD)
    sfm.fit(x_train_scaled, y_train)

    x_train_selected = sfm.transform(x_train_scaled)
    x_test_selected = sfm.transform(x_test_scaled)

    print(f"选择的特征数量: {x_train_selected.shape[1]}")

    return x_train_selected, x_test_selected, scaler, sfm, important_features
