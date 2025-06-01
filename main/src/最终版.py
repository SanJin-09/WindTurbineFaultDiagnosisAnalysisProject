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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.fft import fft
from sklearn.base import BaseEstimator, ClassifierMixin

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

    # 检查SCADA数据列名
    print("SCADA数据列名:", scada.columns.tolist())

    # 检查故障数据列名
    print("故障数据列名:", faults.columns.tolist())

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


# 2. 特征工程
def feature_engineering(df):
    print("特征工程处理中...")

    # 选择关键传感器列 - 基于实际数据列名
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

        # 频域特征 (FFT)
        try:
            # 填充缺失值后进行FFT
            col_data = df[col].fillna(0).values
            fft_vals = np.abs(fft(col_data))
            df[f'{col}_fft_peak'] = np.max(fft_vals[:len(fft_vals) // 2])
            df[f'{col}_fft_mean'] = np.mean(fft_vals[:len(fft_vals) // 2])
        except Exception as e:
            print(f"处理FFT时出错 ({col}): {e}")
            df[f'{col}_fft_peak'] = 0
            df[f'{col}_fft_mean'] = 0

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

    # 编码故障类型
    le = LabelEncoder()
    # 填充NaN值为'No_Fault'
    df['Fault_Type'] = df['Fault_Type'].fillna('No_Fault')
    y = le.fit_transform(df['Fault_Type'])

    return df[features], y, le, available_cols


# 3. 可视化故障前后传感器趋势
def visualize_sensor_trends(df, fault_type, sensor_cols, hours_before=2, hours_after=1):
    """
    可视化特定故障类型前后的传感器趋势

    参数:
    df -- 包含故障数据的数据框
    fault_type -- 要分析的故障类型 (如 'GF', 'MF')
    sensor_cols -- 要可视化的传感器列
    hours_before -- 故障前多少小时的数据
    hours_after -- 故障后多少小时的数据
    """
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

    # 创建可视化
    plt.figure(figsize=(15, 10))
    num_plots = len(sensor_cols)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    for i, sensor in enumerate(sensor_cols, 1):
        plt.subplot(rows, cols, i)
        sns.lineplot(data=event_data, x='Hours_from_Fault', y=sensor)
        plt.axvline(x=0, color='r', linestyle='--', label='Fault Time')
        plt.title(f'{sensor} around {fault_type} Fault')
        plt.xlabel('Hours from Fault')
        plt.ylabel(sensor)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'sensor_trends_{fault_type}.png')
    print(f"保存传感器趋势图为 sensor_trends_{fault_type}.png")
    plt.close()


# 4. LSTM模型
class LSTMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, sequence_length=10, epochs=20, batch_size=64, units=64, dropout=0.2):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.model = None
        self.classes_ = None
        self.le = None

    def fit(self, X, y):
        # 转换标签为one-hot编码
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        y_categorical = to_categorical(y_encoded)

        # 创建序列数据
        X_seq, y_seq = self.create_sequences(X.values, y_categorical)

        # 构建LSTM模型
        self.model = Sequential()
        self.model.add(LSTM(self.units, input_shape=(self.sequence_length, X.shape[1]),
                            return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(self.units))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(y_categorical.shape[1], activation='softmax'))

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X):
        # 创建序列数据
        X_seq, _ = self.create_sequences(X.values, np.zeros((X.shape[0], len(self.classes_))))

        # 预测
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        return self.le.inverse_transform(y_pred)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            # 注意：标签对应序列的最后时间步
            y_seq.append(y[i + self.sequence_length - 1])
        return np.array(X_seq), np.array(y_seq)


# 5. Transformer模型
class TransformerModel(BaseEstimator, ClassifierMixin):
    def __init__(self, sequence_length=10, epochs=15, batch_size=64, d_model=64, num_heads=4, ff_dim=128, dropout=0.1):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.model = None
        self.classes_ = None
        self.le = None

    def fit(self, X, y):
        # 转换标签为one-hot编码
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        y_categorical = to_categorical(y_encoded)
        num_classes = y_categorical.shape[1]

        # 创建序列数据
        X_seq, y_seq = self.create_sequences(X.values, y_categorical)

        # 构建Transformer模型
        inputs = Input(shape=(self.sequence_length, X.shape[1]))

        # 多头注意力机制
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model
        )(inputs, inputs)

        # 残差连接和层归一化
        norm_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

        # 前馈网络
        ff_output = Dense(self.ff_dim, activation="relu")(norm_output)
        ff_output = Dense(X.shape[1])(ff_output)

        # 残差连接和层归一化
        norm_output2 = LayerNormalization(epsilon=1e-6)(norm_output + ff_output)

        # 全局平均池化（取序列最后一步）
        pooled = Dense(128, activation="relu")(norm_output2[:, -1, :])
        pooled = Dropout(self.dropout)(pooled)
        outputs = Dense(num_classes, activation="softmax")(pooled)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X):
        # 创建序列数据
        X_seq, _ = self.create_sequences(X.values, np.zeros((X.shape[0], len(self.classes_))))

        # 预测
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        return self.le.inverse_transform(y_pred)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            # 注意：标签对应序列的最后时间步
            y_seq.append(y[i + self.sequence_length - 1])
        return np.array(X_seq), np.array(y_seq)


# 6. 训练和评估模型
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, le):
    """
    训练和评估单个模型

    参数:
    model -- 模型实例
    model_name -- 模型名称
    X_train, X_test, y_train, y_test -- 训练和测试数据
    le -- 标签编码器

    返回:
    dict -- 包含模型性能指标和训练时间的字典
    """
    start_time = time.time()

    # 训练模型
    print(f"训练 {model_name} 模型...")
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # 分类报告
    class_names = le.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    print(f"保存混淆矩阵为 confusion_matrix_{model_name}.png")

    # 特征重要性（如果可用）
    if hasattr(model, 'feature_importances_'):
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        plt.figure(figsize=(12, 8))
        feat_importances.nlargest(15).plot(kind='barh')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png')
        plt.close()
        print(f"保存特征重要性图为 feature_importance_{model_name}.png")

    # 保存模型
    joblib.dump(model, f'model_{model_name}.pkl')
    print(f"保存模型为 model_{model_name}.pkl")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'train_time': train_time,
        'classification_report': report
    }


# 7. 模型比较可视化
def visualize_model_comparison(results):
    # 创建比较数据框
    df = pd.DataFrame(results)

    # 性能指标比较
    metrics = ['accuracy', 'f1_score', 'recall']
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(data=df, x='model', y=metric)
        plt.title(f'Model {metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.close()
    print("保存模型性能对比图为 model_performance_comparison.png")

    # 训练时间比较
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='train_time')
    plt.title('Model Training Time Comparison (seconds)')
    plt.xticks(rotation=45)
    plt.ylabel('Training Time (s)')
    plt.tight_layout()
    plt.savefig('model_training_time_comparison.png')
    plt.close()
    print("保存训练时间对比图为 model_training_time_comparison.png")

    # 保存详细结果
    df.to_csv('model_comparison_results.csv', index=False)
    print("模型比较结果已保存为 model_comparison_results.csv")


# 主函数
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

        # 3. 可视化故障前后的传感器趋势
        print("\n" + "=" * 50)
        print("可视化故障前后的传感器趋势")
        print("=" * 50)

        # 获取所有故障类型（排除'No_Fault'）
        fault_types = [ft for ft in le.classes_ if ft != 'No_Fault']

        if not fault_types:
            print("未找到故障数据，跳过传感器趋势可视化")
        else:
            # 可视化每种故障类型的传感器趋势
            for fault_type in fault_types[:min(3, len(fault_types))]:  # 最多可视化3种故障类型
                print(f"可视化 {fault_type} 故障前后的传感器趋势...")
                visualize_sensor_trends(df, fault_type, sensor_cols[:6])  # 可视化前6个传感器

        # 4. 数据划分
        print("\n" + "=" * 50)
        print("划分训练集和测试集")
        print("=" * 50)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # 保存预处理对象
        joblib.dump(le, 'label_encoder.pkl')

        # 5. 训练和评估多个模型
        print("\n" + "=" * 50)
        print("训练和评估多个模型")
        print("=" * 50)

        # 定义模型列表 (根据图片修正部分)
        models = [
            ('RandomForest', RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
            )),
            ('SVM', SVC(
                kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True
            )),
            ('NeuralNetwork', MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu',
                solver='adam', max_iter=500, random_state=42
            )),
            ('XGBoost', XGBClassifier(
                n_estimators=200, learning_rate=0.1,
                eval_metric='mlogloss'
            )),
            ('LSTM', LSTMModel(sequence_length=10, epochs=25)),
            ('Transformer', TransformerModel(sequence_length=10, epochs=20))
        ]

        results = []

        for name, model in models:
            try:
                print(f"\n=== 训练 {name} 模型 ===")
                result = train_and_evaluate_model(
                    model, name, X_train, X_test, y_train, y_test, le
                )

                results.append(result)
                print(f"\n{name} 模型结果:")
                print(f"准确率: {result['accuracy']:.4f}, F1分数: {result['f1_score']:.4f}")
                print(f"训练时间: {result['train_time']:.2f}秒")

            except Exception as e:
                print(f"训练 {name} 模型时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 6. 模型比较可视化
        print("\n" + "=" * 50)
        print("模型比较可视化")
        print("=" * 50)
        visualize_model_comparison(results)

        print("\n" + "=" * 50)
        print("分析完成! 所有模型和可视化图表已保存")
        print("=" * 50)
        print("结果摘要：")
        for result in results:
            print(
                f"- {result['model']}: 准确率={result['accuracy']:.4f}, F1分数={result['f1_score']:.4f}, 训练时间={result['train_time']:.2f}秒")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()