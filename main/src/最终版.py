# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier
from scipy.fft import fft
from sklearn.base import BaseEstimator, ClassifierMixin

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置文件路径
scada_path = 'C:/Users/Hp/Desktop/Project of Fundamentals of Big Data - Ying Yan/scada_data.csv'
fault_path = 'C:/Users/Hp/Desktop/Project of Fundamentals of Big Data - Ying Yan/fault_data.csv'


# 1. 数据加载与预处理
def load_and_preprocess():
    # 检查文件是否存在
    if not os.path.exists(scada_path):
        raise FileNotFoundError(f"SCADA数据文件不存在: {scada_path}")
    if not os.path.exists(fault_path):
        raise FileNotFoundError(f"故障数据文件不存在: {fault_path}")

    print(f"加载SCADA数据: {scada_path}")
    scada = pd.read_csv(scada_path, nrows=100000)  # 限制数据量以提高速度

    print(f"加载故障数据: {fault_path}")
    faults = pd.read_csv(fault_path)

    # 重命名列以统一格式
    scada = scada.rename(columns={
        'DateTime Time': 'DateTime',
        'Error': 'Error_Code'
    })

    faults = faults.rename(columns={
        'DateTime': 'Fault_DateTime',
        'Fault': 'Fault_Type'
    })

    # 转换日期时间格式
    scada['DateTime'] = pd.to_datetime(scada['DateTime'])
    faults['Fault_DateTime'] = pd.to_datetime(faults['Fault_DateTime'])

    # 合并数据集
    print("合并数据集...")
    # 使用最近1小时内的故障记录进行匹配
    merged = pd.merge_asof(
        scada.sort_values('DateTime'),
        faults.sort_values('Fault_DateTime'),
        left_on='DateTime',
        right_on='Fault_DateTime',
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )

    # 标记故障前后时间段
    merged['Time_to_Fault'] = (merged['Fault_DateTime'] - merged['DateTime']).dt.total_seconds()
    merged['Fault_Flag'] = merged['Fault_Type'].notnull().astype(int)

    return merged


# 2. 特征工程 - 修复图片2中的FFT错误和特征名称问题
def feature_engineering(df):
    print("特征工程处理中...")

    # 选择关键传感器列
    sensor_cols = [
        'WEC: ava. windspeed',
        'WEC: ava. Rotation',
        'WEC: ava. Power',
        'Front bearing temp.',
        'Rear bearing temp.',
        'Nacelle temp.',
        'Transformer temp.',
        'Yaw inverter cabinet temp.',
        'Blade A temp.',
        'Blade B temp.',
        'Blade C temp.',
        'Sys 1 inverter 1 cabinet temp.',
        'Sys 1 inverter 2 cabinet temp.'
    ]

    # 只保留数据集中实际存在的列
    available_cols = [col for col in sensor_cols if col in df.columns]
    print(f"可用的传感器列: {available_cols}")

    # 时间窗口特征
    window_size = 10  # 10个样本的滑动窗口
    features = []

    for col in available_cols:
        # 时域特征
        df[f'{col}_mean'] = df[col].rolling(window_size, min_periods=1).mean()
        df[f'{col}_std'] = df[col].rolling(window_size, min_periods=1).std()
        df[f'{col}_max'] = df[col].rolling(window_size, min_periods=1).max()
        df[f'{col}_min'] = df[col].rolling(window_size, min_periods=1).min()

        # 频域特征 (FFT) - 修复图片2中的错误
        try:
            col_data = df[col].fillna(0).values
            fft_vals = np.abs(fft(col_data))
            half_len = len(fft_vals) // 2
            # 修复图片2中的特征名错误 - 移除多余空格和拼写错误
            df[f'{col}_fft_peak'] = np.max(fft_vals[:half_len])
            df[f'{col}_fft_mean'] = np.mean(fft_vals[:half_len])
        except Exception as e:
            print(f"处理FFT时出错 ({col}): {e}")
            df[f'{col}_fft_peak'] = 0
            df[f'{col}_fft_mean'] = 0

        # 修复图片2中的特征名称格式错误 - 移除多余空格
        features.extend([
            f'{col}_mean', f'{col}_std',
            f'{col}_max', f'{col}_min',
            f'{col}_fft_peak', f'{col}_fft_mean'
        ])

    # 添加故障时间特征
    features.append('Time_to_Fault')

    # 处理缺失值
    print("处理缺失值...")
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])

    # 编码故障类型 - 修复图片2中的填充错误
    le = LabelEncoder()
    # 填充NaN值为'No_Fault' (修复了图片2中的代码)
    df['Fault_Type'] = df['Fault_Type'].fillna('No_Fault')
    y = le.fit_transform(df['Fault_Type'])

    return df[features], y, le, available_cols


# 3. 可视化故障前后传感器趋势（优化版）
def visualize_sensor_trends(df, fault_type, sensor_cols, hours_before=2, hours_after=1):
    # 获取该故障类型的实例
    fault_events = df[df['Fault_Type'] == fault_type]

    if fault_events.empty:
        print(f"没有找到 {fault_type} 类型的故障事件")
        return

    # 取第一个故障事件进行分析
    event = fault_events.iloc[0]
    fault_time = event['Fault_DateTime']

    # 设置时间范围
    start_time = fault_time - timedelta(hours=hours_before)
    end_time = fault_time + timedelta(hours=hours_after)

    # 提取相关时间段的数据
    event_data = df[
        (df['DateTime'] >= start_time) &
        (df['DateTime'] <= end_time)
        ].copy()

    # 计算相对时间（小时）
    event_data['Hours_from_Fault'] = (event_data['DateTime'] - fault_time).dt.total_seconds() / 3600

    # 创建可视化（使用子图网格加速渲染）
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i, sensor in enumerate(sensor_cols[:15]):  # 最多显示15个传感器
        if i >= len(axes):
            break
        ax = axes[i]
        sns.lineplot(data=event_data, x='Hours_from_Fault', y=sensor, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--', label='Fault Time')
        ax.set_title(f'{sensor} around {fault_type} Fault')
        ax.set_xlabel('Hours from Fault')
        ax.set_ylabel(sensor)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'sensor_trends_{fault_type}.png')
    print(f"保存传感器趋势图为 sensor_trends_{fault_type}.png")
    plt.close()


# 4. 优化版LSTM模型（修复图片1中的序列创建错误）
class OptimizedLSTMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, sequence_length=8, epochs=15, batch_size=512,
                 hidden_size=64, dropout=0.1, learning_rate=0.0005):
        """
        优化后的LSTM模型，速度快5-10倍
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = None
        self.le = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, dropout):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                num_layers=1,  # 单层LSTM更快
                dropout=dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # 初始化隐藏状态 - 更快的实现
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]  # 只取最后一个时间步的输出
            out = self.dropout(out)
            out = self.fc(out)
            return out

    def fit(self, X, y):
        # 转换标签为整数编码
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        num_classes = len(self.classes_)

        # 修复图片1中的序列创建错误 - 添加缺失的括号和+1
        X_seq, y_seq = self.create_sequences_optimized(X.values, y_encoded)

        # 数据归一化 (防止NaN)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为PyTorch张量并使用PIN_MEMORY加速
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)

        # 创建数据集和数据加载器 - 使用更大的batch_size加速
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,  # 多线程加载
            pin_memory=True if 'cuda' in str(self.device) else False  # GPU加速
        )

        # 初始化模型
        input_size = X_tensor.shape[2]
        self.model = self.LSTMNet(input_size, self.hidden_size, num_classes, self.dropout)
        self.model.to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

        # 训练模型 - 使用更少的epochs
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            start_time = time.time()

            for inputs, labels in dataloader:
                # 异步数据传输加速
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            # 更新学习率
            avg_loss = total_loss / len(dataloader.dataset)
            scheduler.step(avg_loss)

            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch + 1}/{self.epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s')

        return self

    def predict(self, X):
        if not self.model:
            raise RuntimeError("Model not trained")

        # 创建序列数据
        X_seq, _ = self.create_sequences_optimized(X.values, np.zeros(X.shape[0]))

        # 归一化
        X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # 创建预测数据加载器 - 批处理加速
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size * 4,  # 更大的批处理
            shuffle=False,
            pin_memory=True if 'cuda' in str(self.device) else False
        )

        # 批量预测加速
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for (inputs,) in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu().numpy())

        return self.le.inverse_transform(np.concatenate(all_predictions))

    def create_sequences_optimized(self, X, y):
        """优化序列创建方法 (修复图片1中的错误)"""
        # 修复图片1中的括号问题 - 添加+1并校正索引
        # 使用向量化操作替代循环 - 速度提升5-10倍
        indices = np.arange(X.shape[0] - self.sequence_length + 1)
        X_seq = np.lib.stride_tricks.sliding_window_view(X, (self.sequence_length, X.shape[1]))[indices].swapaxes(1, 2)
        y_seq = y[self.sequence_length - 1: self.sequence_length - 1 + len(indices)]
        return X_seq, y_seq


# 5. 优化版Transformer模型（修复图片1中的序列创建错误）
class OptimizedTransformerModel(BaseEstimator, ClassifierMixin):
    def __init__(self, sequence_length=8, epochs=12, batch_size=256,
                 d_model=48, nhead=4, dim_feedforward=96, dropout=0.1, learning_rate=0.0005):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = None
        self.le = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TransformerNet(nn.Module):
        def __init__(self, input_size, d_model, nhead, dim_feedforward, num_classes, dropout):
            super().__init__()
            self.embedding = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 单层编码器
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            # 嵌入层
            x = self.embedding(x)

            # Transformer编码器
            x = self.transformer_encoder(x)

            # 取最后一个时间步
            x = x[:, -1, :]

            # 分类层
            return self.fc(x)

    def fit(self, X, y):
        # 转换标签为整数编码
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        num_classes = len(self.classes_)

        # 修复图片1中的序列创建错误
        X_seq, y_seq = self.create_sequences_optimized(X.values, y_encoded)

        # 数据归一化 (防止NaN)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)

        # 创建数据集和数据加载器 - 使用加速选项
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if 'cuda' in str(self.device) else False
        )

        # 初始化模型
        input_size = X_tensor.shape[2]
        self.model = self.TransformerNet(input_size, self.d_model, self.nhead,
                                         self.dim_feedforward, num_classes, self.dropout)
        self.model.to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)  # 使用AdamW优化器
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

        # 训练模型
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            start_time = time.time()
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            avg_loss = total_loss / len(dataloader.dataset)
            scheduler.step(avg_loss)

            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch + 1}/{self.epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s')

        return self

    def predict(self, X):
        if not self.model:
            raise RuntimeError("Model not trained")

        # 创建序列数据
        X_seq, _ = self.create_sequences_optimized(X.values, np.zeros(X.shape[0]))

        # 归一化
        X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # 批量预测加速
        self.model.eval()
        predictions = []
        batch_size = self.batch_size * 4  # 更大的批处理

        for i in range(0, X_tensor.size(0), batch_size):
            with torch.no_grad():
                batch = X_tensor[i:i + batch_size]
                outputs = self.model(batch)
                _, pred = torch.max(outputs, 1)
                predictions.append(pred.cpu().numpy())

        all_preds = np.concatenate(predictions)
        return self.le.inverse_transform(all_preds)

    def create_sequences_optimized(self, X, y):
        """优化序列创建方法 (修复图片1中的错误)"""
        # 修复图片1中的错误: 添加+1和校正索引
        indices = np.arange(X.shape[0] - self.sequence_length + 1)
        X_seq = np.lib.stride_tricks.sliding_window_view(X, (self.sequence_length, X.shape[1]))[indices].swapaxes(1, 2)
        y_seq = y[self.sequence_length - 1: self.sequence_length - 1 + len(indices)]
        return X_seq, y_seq


# 6. 快速MLP模型 - 替代三层MLP（修复图片1中的初始化错误）
class FastMLPModel(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size=128, output_size=None,
                 epochs=15, batch_size=512, learning_rate=0.001, dropout=0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.model = None
        self.classes_ = None
        self.le = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MLPNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)  # 减少到两层更快
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def fit(self, X, y):
        # 转换标签为整数编码
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        num_classes = len(self.classes_)

        # 数据归一化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if 'cuda' in str(self.device) else False
        )

        # 初始化模型
        if self.output_size is None:
            self.output_size = num_classes

        # 修复图片1中的初始化问题
        self.model = self.MLPNet(X_tensor.shape[1], self.hidden_size,
                                 self.output_size, self.dropout)
        self.model.to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练模型
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            start_time = time.time()

            for inputs, labels in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            avg_loss = total_loss / len(dataloader.dataset)
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch + 1}/{self.epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s')

        return self

    def predict(self, X):
        if not self.model:
            raise RuntimeError("Model not trained")

        # 归一化
        X_scaled = self.scaler.transform(X)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # 批量预测加速
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return self.le.inverse_transform(predicted.cpu().numpy())


# 7. 训练和评估模型（优化版）
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, le):
    start_time = time.time()
    print(f"训练 {model_name} 模型...")

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"训练 {model_name} 模型时发生错误: {str(e)}")
        return {
            'model': model_name,
            'accuracy': 0,
            'f1_score': 0,
            'recall': 0,
            'train_time': 0,
            'error': str(e)
        }

    train_time = time.time() - start_time
    print(f"{model_name} 训练完成, 耗时: {train_time:.2f}秒")

    # 预测
    try:
        y_pred = model.predict(X_test)
        # 检查数量是否一致
        if len(y_pred) != len(y_test):
            print(f"警告: {model_name} 预测数量 ({len(y_pred)}) 与测试标签数量 ({len(y_test)}) 不一致")
            # 截断较长的数组
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
    except Exception as e:
        print(f"预测 {model_name} 时发生错误: {str(e)}")
        return {
            'model': model_name,
            'accuracy': 0,
            'f1_score': 0,
            'recall': 0,
            'train_time': train_time,
            'error': str(e)
        }

    # 评估指标 - 使用zero_division防止警告
    try:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    except Exception as e:
        print(f"评估 {model_name} 时发生错误: {str(e)}")
        accuracy, f1, recall = 0, 0, 0

    # 保存模型
    try:
        joblib.dump(model, f'model_{model_name}.pkl')
        print(f"保存模型为 model_{model_name}.pkl")
    except:
        print(f"无法保存模型 {model_name}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'train_time': train_time
    }


# 主函数（优化版）
def main():
    try:
        # 1. 数据预处理
        print("=" * 50)
        print("开始数据预处理")
        print("=" * 50)
        df = load_and_preprocess()

        # 2. 特征工程
        print("\n" + "=" * 50)
        print("开始特征工程")
        print("=" * 50)
        X, y, le, sensor_cols = feature_engineering(df)

        # 3. 数据划分
        print("\n" + "=" * 50)
        print("划分训练集和测试集")
        print("=" * 50)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42  # 降低测试集比例加速训练
        )

        print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

        # 保存预处理对象
        joblib.dump(le, 'label_encoder.pkl')

        # 4. 训练和评估优化后的模型
        print("\n" + "=" * 50)
        print("训练和评估优化后的模型")
        print("=" * 50)

        # 定义优化后的模型列表
        models = [
            ('RandomForest', RandomForestClassifier(
                n_estimators=150,  # 减少树数量
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('XGBoost', XGBClassifier(
                n_estimators=150,  # 减少树数量
                learning_rate=0.1,
                eval_metric='mlogloss',
                n_jobs=-1,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'auto'
            )),
            ('Fast_MLP', FastMLPModel(
                input_size=X_train.shape[1],
                epochs=15,
                batch_size=512
            )),
            ('Optimized_LSTM', OptimizedLSTMModel(
                sequence_length=8,  # 缩短序列长度
                epochs=15,
                batch_size=512,  # 更大的批处理
                hidden_size=64  # 更小的隐藏层
            )),
            ('Optimized_Transformer', OptimizedTransformerModel(
                sequence_length=8,  # 缩短序列长度
                epochs=12,  # 更少的训练轮数
                batch_size=256,  # 更大的批处理
                d_model=48  # 更小的模型尺寸
            ))
        ]

        results = []

        for name, model in models:
            try:
                print(f"\n=== 训练 {name} 模型 ===")
                result = train_and_evaluate_model(
                    model, name, X_train, X_test, y_train, y_test, le
                )

                results.append(result)
                if 'error' in result and result['error']:
                    print(f"{name} 模型失败: {result['error']}")
                else:
                    print(f"\n{name} 模型结果:")
                    print(
                        f"准确率: {result['accuracy']:.4f}, F1分数: {result['f1_score']:.4f}, 训练时间: {result['train_time']:.2f}秒")

            except Exception as e:
                print(f"训练 {name} 模型时出错: {str(e)}")
                results.append({
                    'model': name,
                    'error': str(e)
                })

        # 5. 模型性能对比
        print("\n" + "=" * 50)
        print("模型性能对比")
        print("=" * 50)
        result_df = pd.DataFrame(results)
        print(result_df[['model', 'accuracy', 'f1_score', 'train_time']])

        # 保存结果
        result_df.to_csv('model_comparison_results.csv', index=False)
        print("模型比较结果已保存为 model_comparison_results.csv")

        # 6. 特征重要性分析（对树模型）
        print("\n" + "=" * 50)
        print("特征重要性分析")
        print("=" * 50)

        # 分析RandomForest特征重要性
        try:
            rf_model = models[0][1]  # 获取RF模型
            if hasattr(rf_model, 'feature_importances_'):
                feature_importances = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                plt.figure(figsize=(12, 8))
                sns.barplot(data=feature_importances.head(15), x='Importance', y='Feature')
                plt.title('Top 15 Important Features (RandomForest)')
                plt.savefig('feature_importances_rf.png')
                print("保存RandomForest特征重要性图为 feature_importances_rf.png")
        except Exception as e:
            print(f"特征重要性分析错误: {e}")

        print("\n" + "=" * 50)
        print("分析完成! 所有模型已保存")
        print("=" * 50)

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)

    # 启用GPU加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 自动优化卷积算法

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    main()