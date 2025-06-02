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
    scada = pd.read_csv(scada_path)

    print(f"加载故障数据: {fault_path}")
    faults = pd.read_csv(fault_path)

    # 数据清洗
    print("数据清洗中...")

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

    # 保存合并后的数据用于调试
    merged.to_csv('merged_data.csv', index=False)
    print("合并后的数据已保存为 merged_data.csv")

    return merged


# 2. 特征工程 - 修复特征名称格式问题
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
        # 时域特征 - 确保没有前导空格
        df[f'{col}_mean'] = df[col].rolling(window_size, min_periods=1).mean()
        df[f'{col}_std'] = df[col].rolling(window_size, min_periods=1).std()
        df[f'{col}_max'] = df[col].rolling(window_size, min_periods=1).max()
        df[f'{col}_min'] = df[col].rolling(window_size, min_periods=1).min()

        # 频域特征 (FFT) - 修复特征名称格式
        try:
            # 填充缺失值后进行FFT
            col_data = df[col].fillna(0).values
            fft_vals = np.abs(fft(col_data))
            # 修复：使用正确长度的切片
            half_len = len(fft_vals) // 2
            # 使用正确的特征名称格式 - 无前导空格
            df[f'{col}_fft_peak'] = np.max(fft_vals[:half_len])
            df[f'{col}_fft_mean'] = np.mean(fft_vals[:half_len])
        except Exception as e:
            print(f"处理FFT时出错 ({col}): {e}")
            df[f'{col}_fft_peak'] = 0
            df[f'{col}_fft_mean'] = 0

        # 特征名称格式化 - 确保没有前导空格
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

    # 编码故障类型 - 修复填充错误
    le = LabelEncoder()
    # 填充NaN值为'No_Fault'
    df['Fault_Type'] = df['Fault_Type'].fillna('No_Fault')
    y = le.fit_transform(df['Fault_Type'])

    return df[features], y, le, available_cols


# 3. 优化并修复后的LSTM模型
class FixedLSTMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, sequence_length=8, epochs=15, batch_size=512,
                 hidden_size=64, dropout=0.1, learning_rate=0.0005):
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

    class FixedLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, dropout):
            super().__init__()
            self.hidden_size = hidden_size

            # 确保单层LSTM不会有dropout警告
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                num_layers=1  # 明确指定单层
            )
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]  # 只取最后一个时间步的输出
            out = self.dropout_layer(out)
            return self.fc(out)

    def fit(self, X, y):
        # 转换标签为整数编码
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        num_classes = len(self.classes_)

        # 使用修复的序列创建方法
        X_seq, y_seq = self.create_sequences_fixed(X.values, y_encoded)

        # 数据归一化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)

        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,  # 多线程加载
            pin_memory=True if 'cuda' in str(self.device) else False
        )

        # 初始化模型
        input_size = X_tensor.shape[2]
        self.model = self.FixedLSTM(input_size, self.hidden_size, num_classes, self.dropout)
        self.model.to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        # 训练模型
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            start_time = time.time()

            for inputs, labels in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                outputs = self.model(input)
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

        # 使用修复的序列创建方法
        X_seq, _ = self.create_sequences_fixed(X.values, np.zeros(X.shape[0]))

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
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.cpu().numpy())

        all_preds = np.concatenate(predictions)
        return self.le.inverse_transform(all_preds)

    def create_sequences_fixed(self, X, y):
        """完全修复的序列创建方法"""
        # 确保序列创建范围正确
        num_samples = X.shape[0] - self.sequence_length + 1
        if num_samples <= 0:
            # 如果数据不足，自动调整序列长度
            self.sequence_length = max(1, X.shape[0] // 2)
            num_samples = max(1, X.shape[0] - self.sequence_length + 1)
            print(f"警告：序列长度自动调整为 {self.sequence_length} 以适应数据")

        # 使用向量化操作替代循环
        indices = np.arange(num_samples)
        X_seq = np.array([X[i:i + self.sequence_length] for i in indices])

        # 获取对应的标签 (确保数量匹配)
        if len(y) > self.sequence_length:
            y_seq = y[self.sequence_length - 1: self.sequence_length - 1 + len(indices)]
        else:
            # 处理数据不足的情况
            y_seq = y[-num_samples:] if len(y) > 0 else np.zeros(num_samples)

        return X_seq, y_seq


# 4. 优化并修复后的Transformer模型
class FixedTransformerModel(BaseEstimator, ClassifierMixin):
    def __init__(self, sequence_length=8, epochs=12, batch_size=256,
                 d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, learning_rate=0.0005):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.drop极简 = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = None
        self.le = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

    class FixedTransformer(nn.Module):
        def __init__(self, input_size, d_model, nhead, dim_feedforward, num_classes, dropout):
            super().__init__()
            # 嵌入层
            self.embedding = nn.Linear(input_size, d_model)

            # 简化Transformer结构
            encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 单层编码器

            # 分类层
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

        # 使用修复的序列创建方法
        X_seq, y_seq = self.create_sequences_fixed(X.values, y_encoded)

        # 数据归一化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)

        # 维度检查
        if X_tensor.dim() != 3:
            raise ValueError(f"输入应为3维张量，但得到 {X_tensor.dim()} 维")

        # 创建数据加载器
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
        self.model = self.FixedTransformer(
            input_size,
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            num_classes,
            self.dropout
        )
        self.model.to(self.device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

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

        # 使用修复的序列创建方法
        X_seq, _ = self.create_sequences_fixed(X.values, np.zeros(X.shape[0]))

        # 归一化
        X_scaled = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[2]))
        X_scaled = X_scaled.reshape(X_seq.shape)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # 批量预测
        self.model.eval()
        predictions = []
        batch_size = self.batch_size * 4  # 更大的批处理

        for i in range(0, X_tensor.size(0), batch_size):
            with torch.no_grad():
                batch = X_tensor[i:i + batch_size]
                outputs = self.model(batch)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.cpu().numpy())

        all_preds = np.concatenate(predictions)
        return self.le.inverse_transform(all_preds)

    def create_sequences_fixed(self, X, y):
        """完全修复的序列创建方法"""
        # 确保序列创建范围正确
        num_samples = X.shape[0] - self.sequence_length + 1
        if num_samples <= 0:
            # 如果数据不足，自动调整序列长度
            self.sequence_length = max(1, X.shape[0] // 2)
            num_samples = max(1, X.shape[0] - self.sequence_length + 1)
            print(f"警告：序列长度自动调整为 {self.sequence_length} 以适应数据")

        # 使用向量化操作替代循环
        indices = np.arange(num_samples)
        X_seq = np.array([X[i:i + self.sequence_length] for i in indices])

        # 获取对应的标签 (确保数量匹配)
        if len(y) > self.sequence_length:
            y_seq = y[self.sequence_length - 1: self.sequence_length - 1 + num_samples]
        else:
            # 处理数据不足的情况
            y_seq = y[-num_samples:] if len(y) > 0 else np.zeros(num_samples)

        return X_seq, y_seq


# 5. 模型训练和评估函数
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, le):
    start_time = time.time()
    print(f"训练 {model_name} 模型...")

    try:
        # 训练模型
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        if len(y_pred) != len(y_test):
            # 截断较长的数组
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # 保存模型
        joblib.dump(model, f'model_{model_name}.pkl')

        return {
            'model': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall,
            'train_time': train_time
        }

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


# 画图函数
def plot_fault_distribution(df):
    """绘制故障类型分布图"""
    plt.figure(figsize=(12, 8))
    fault_counts = df['Fault_Type'].value_counts()
    sns.barplot(x=fault_counts.index, y=fault_counts.values, palette='viridis')
    plt.title('Fault Type Distribution')
    plt.xlabel('Fault Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fault_type_distribution.png')
    print("保存故障类型分布图为 fault_type_distribution.png")


def plot_correlation_heatmap(X):
    """绘制特征相关性热力图"""
    plt.figure(figsize=(18, 14))
    corr = X.corr()
    # 只显示相关性绝对值较高的部分
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # 只显示相关性绝对值大于0.5的单元格
    corr_abs = corr.abs()
    high_corr_mask = corr_abs > 0.5
    corr_masked = corr.where(high_corr_mask & ~mask, np.nan)

    sns.heatmap(corr_masked, cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size": 8},
                cbar_kws={"shrink": .75}, mask=corr_masked.isnull())
    plt.title('Feature Correlation Heatmap (|correlation| > 0.5)')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png')
    print("保存特征相关性热力图为 feature_correlation_heatmap.png")


def plot_model_performance(results):
    """绘制模型性能对比图"""
    if not results:
        return

    # 创建数据框架
    metrics = ['accuracy', 'f1_score', 'recall']
    df = pd.DataFrame(results)

    # 设置位置
    bar_width = 0.25
    r = np.arange(len(df))

    # 创建柱状图
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics):
        plt.bar(r + i * bar_width, df[metric], width=bar_width, label=metric.replace('_', ' ').title())

    # 添加文本标签
    for i, row in df.iterrows():
        for j, metric in enumerate(metrics):
            plt.text(i + j * bar_width, row[metric] + 0.02, f'{row[metric]:.3f}',
                     ha='center', va='bottom', fontsize=9)

    # 装饰图表
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(r + bar_width, df['model'], fontsize=10)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    print("保存模型性能对比图为 model_performance_comparison.png")


def plot_training_time(results):
    """绘制模型训练时间对比图"""
    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values('train_time', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='train_time', palette='mako')
    plt.title('Model Training Time Comparison (seconds)')
    plt.ylabel('Training Time (s)')
    plt.xlabel('Model')

    # 添加数值标签
    for index, row in enumerate(df.itertuples()):
        plt.text(index, row.train_time + 10, f'{row.train_time:.1f}s',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('model_training_time_comparison.png')
    print("保存模型训练时间对比图为 model_training_time_comparison.png")


def plot_confusion_matrices(results, y_test, le):
    """为每个模型绘制混淆矩阵"""
    classes = le.classes_

    for result in results:
        if 'predictions' in result and result['predictions']:
            y_pred = result['predictions']
            model_name = result['model']

            # 确保长度一致
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test_subset = y_test[:min_len]

            # 绘制混淆矩阵
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test_subset, y_pred)

            # 标准化混淆矩阵
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # 绘制热力图
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                        xticklabels=classes, yticklabels=classes,
                        vmin=0, vmax=1)
            plt.title(f'Normalized Confusion Matrix - {model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{model_name}.png')
            print(f"保存{model_name}模型的混淆矩阵为 confusion_matrix_{model_name}.png")


# 主函数
def main():
    try:
        # 1. 数据预处理
        print("=" * 50)
        print("开始数据预处理")
        print("=" * 50)
        df = load_and_preprocess()

        # 添加：绘制故障类型分布图
        plot_fault_distribution(df)

        # 2. 特征工程
        print("\n" + "=" * 50)
        print("开始特征工程")
        print("=" * 50)
        X, y, le, sensor_cols = feature_engineering(df)

        # 添加：绘制特征相关性热力图
        plot_correlation_heatmap(X)

        # 3. 特征筛选
        print("\n应用特征筛选...")
        # 选择最重要的特征
        important_features = [
            'WEC: ava. windspeed_mean',
            'Front bearing temp._mean',
            'Rear bearing temp._mean',
            'Nacelle temp._mean',
            'Transformer temp._mean',
            'Time_to_Fault'
        ]
        # 只保留实际存在的特征
        available_features = [f for f in important_features if f in X.columns]
        if available_features:
            X = X[available_features]
            print(f"使用{len(available_features)}个重要特征")
        else:
            print("警告：所有重要特征都不存在，使用全部特征")

        # 4. 数据划分
        print("\n" + "=" * 50)
        print("划分训练集和测试集")
        print("=" * 50)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
        print(f"测试集标签分布:\n{pd.Series(y_test).value_counts()}")

        # 保存预处理对象
        joblib.dump(le, 'label_encoder.pkl')

        # 5. 训练和评估模型
        print("\n" + "=" * 50)
        print("训练和评估模型")
        print("=" * 50)

        # 定义模型列表 - 使用修复后的模型
        models = [
            ('RandomForest', RandomForestClassifier(
                n_estimators=150, max_depth=15, n_jobs=-1, random_state=42
            )),
            ('XGBoost', XGBClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.1, n_jobs=-1,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'auto'
            )),
            ('Fixed_LSTM', FixedLSTMModel(
                sequence_length=8, epochs=15, batch_size=512, hidden_size=64
            )),
            ('Fixed_Transformer', FixedTransformerModel(
                sequence_length=8, epochs=12, batch_size=256, d_model=64
            ))
        ]

        results = []
        all_predictions = {}

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
                    all_predictions[name] = result['predictions']

            except Exception as e:
                print(f"训练 {name} 模型时出错: {str(e)}")
                results.append({
                    'model': name,
                    'error': str(e)
                })

        # 6. 模型性能对比
        print("\n" + "=" * 50)
        print("模型性能对比")
        print("=" * 50)
        result_df = pd.DataFrame(results)
        print(result_df[['model', 'accuracy', 'f1_score', 'train_time']])

        # 保存结果
        result_df.to_csv('model_comparison_results.csv', index=False)
        print("模型比较结果已保存为 model_comparison_results.csv")

        # 添加：绘制模型性能对比图
        plot_model_performance(results)

        # 添加：绘制模型训练时间对比图
        plot_training_time(results)

        # 添加：为每个模型绘制混淆矩阵
        plot_confusion_matrices(results, y_test, le)

        # 7. 特征重要性分析
        print("\n" + "=" * 50)
        print("特征重要性分析")
        print("=" * 50)

        # 分析RandomForest特征重要性
        try:
            rf_model = models[0][1]
            if hasattr(rf_model, 'feature_importances_'):
                feature_importances = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                plt.figure(figsize=(12, 8))
                sns.barplot(data=feature_importances.head(15), x='Importance', y='Feature')
                plt.title('Top 15 Important Features (RandomForest)')
                plt.tight_layout()
                plt.savefig('feature_importances_rf.png')
                print("保存RandomForest特征重要性图为 feature_importances_rf.png")

                # 添加：特征重要性圆环图
                plt.figure(figsize=(10, 10))
                top_features = feature_importances.head(10)
                # 设置环形参数
                plt.pie(top_features['Importance'],
                        labels=top_features['Feature'],
                        autopct='%1.1f%%',
                        startangle=140,
                        wedgeprops=dict(width=0.4),
                        pctdistance=0.85)
                plt.title('Top 10 Feature Importance (RandomForest)', pad=20)
                plt.savefig('feature_importance_donut.png')
                print("保存特征重要性圆环图为 feature_importance_donut.png")
        except Exception as e:
            print(f"特征重要性分析错误: {e}")

        print("\n" + "=" * 50)
        print("分析完成! 所有模型和可视化已保存")
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

