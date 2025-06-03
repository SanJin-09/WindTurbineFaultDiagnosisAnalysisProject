"""
Author: <SUN Runze>
数据与模型训练成果可视化
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.main.analysis.data_preprocessing import preprocess_data, load_data
from src.main.analysis.model_training import train_models, split_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logger = logging.getLogger('visualization')

def plot_fault_distribution(fault_distribution, title="故障分布"):
    """绘制故障类型分布图"""
    plt.figure(figsize=(12, 6))
    ax = fault_distribution.plot(kind='bar', color='skyblue')
    plt.title(title, fontsize=15)
    plt.xlabel('故障类型')
    plt.ylabel('样本数量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上方添加数据标签
    for i, v in enumerate(fault_distribution):
        ax.text(i, v + 0.5 * max(fault_distribution) / 20, str(v), ha='center', fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/fault_distribution.png')
    plt.show()
    logger.info("已保存故障分布图")

def plot_feature_distribution(labeled_data, features):
    """绘制关键特征分布直方图"""
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(labeled_data[feature], kde=True, bins=30)
        plt.title(f'{feature}分布', fontsize=12)
        plt.xlabel(feature)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/feature_distribution.png')
    plt.show()
    logger.info("已保存特征分布图")

def plot_model_accuracies(accuracies):
    """绘制各模型准确率比较图"""
    models = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    bars = plt.bar(models, accuracy_values, color=colors)

    plt.title('模型准确率对比', fontsize=15)
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.ylim(0.8, 1.0)

    # 在柱子上方添加准确率标签
    for bar, value in zip(bars, accuracy_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('images/model_accuracies.png')
    plt.show()
    logger.info("已保存模型准确率对比图")

def plot_confusion_matrix(y_test, y_pred, classes, best_model_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title(f'{best_model_name}混淆矩阵', fontsize=15)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.show()
    logger.info("已保存混淆矩阵图")

def plot_classification_comparison(labeled_data, y_test, y_pred, features, timesteps=50):
    """绘制真实值与预测值的对比（部分时间序列）"""
    # 从测试集中提取时间序列数据
    time_indices = np.arange(len(y_test))[:timesteps]
    true_labels = y_test[:timesteps]
    pred_labels = y_pred[:timesteps]

    # 绘制时间序列对比
    plt.figure(figsize=(15, 8))

    # 绘制部分特征的变化
    plt.subplot(2, 1, 1)
    for feature in features[:3]:  # 只展示前三个特征
        plt.plot(time_indices, labeled_data[feature].values[:timesteps], label=feature)
    plt.title('部分特征随时间变化', fontsize=12)
    plt.xlabel('时间步')
    plt.ylabel('特征值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制真实标签和预测标签
    plt.subplot(2, 1, 2)
    plt.plot(time_indices, true_labels, 'o-', label='真实故障')
    plt.plot(time_indices, pred_labels, 's-', label='预测故障')
    plt.title('真实故障与预测故障对比', fontsize=12)
    plt.xlabel('时间步')
    plt.ylabel('故障类型')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/classification_comparison.png')
    plt.show()
    logger.info("已保存分类结果对比图")

def plot_feature_correlation(labeled_data):
    """绘制特征相关性热力图"""
    plt.figure(figsize=(12, 10))
    corr = labeled_data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('特征相关性热力图', fontsize=15)
    plt.tight_layout()
    plt.savefig('images/feature_correlation.png')
    plt.show()
    logger.info("已保存特征相关性热力图")

def plot_time_series_features(labeled_data, features, num_samples=1000):
    """绘制部分时间序列特征图"""
    plt.figure(figsize=(15, 8))

    # 确保DateTime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(labeled_data['DateTime']):
        labeled_data['DateTime'] = pd.to_datetime(labeled_data['DateTime'], errors='coerce')

    sampled_data = labeled_data.iloc[:num_samples]

    for feature in features:
        plt.plot(sampled_data['DateTime'], sampled_data[feature], label=feature, alpha=0.7)

    # 标记故障点
    fault_points = sampled_data[sampled_data['Fault'] != 'NoFault']
    for _, row in fault_points.iterrows():
        plt.axvline(x=row['DateTime'], color='red', linestyle='--', alpha=0.7)

    plt.title('时间序列特征变化及故障点', fontsize=15)
    plt.xlabel('时间')
    plt.ylabel('特征值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/time_series_features.png')
    plt.show()
    logger.info("已保存时间序列特征图")

def plot_feature_importance(model, feature_names, model_name):
    """绘制特征重要性图"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title(f"{model_name}特征重要性", fontsize=15)
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'images/{model_name}_feature_importance.png')
        plt.show()
        logger.info(f"已保存{model_name}特征重要性图")
    else:
        logger.warning(f"{model_name}模型没有feature_importances_属性，无法绘制特征重要性图")

# 主程序
def graphics_drawing():
    # 加载并处理数据
    scada_data, fault_data = load_data(logger=logger)
    labeled_data, fault_distribution = preprocess_data(scada_data, fault_data, logger=logger)

    # 确保DateTime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(labeled_data['DateTime']):
        labeled_data['DateTime'] = pd.to_datetime(labeled_data['DateTime'], errors='coerce')

    # 准备特征和标签 - 只选择数值列
    numeric_cols = labeled_data.select_dtypes(include=np.number).columns
    features = [col for col in numeric_cols if col != 'Fault']
    x = labeled_data[features]
    y = labeled_data['Fault']

    # 分割数据集
    x_train, x_test, y_train, y_test = split_data(x, y)

    # 训练模型
    best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = train_models(x_train, y_train, x_test, y_test)

    # 获取故障类别
    classes = np.unique(y)

    # 绘制所有图表
    plot_fault_distribution(fault_distribution)
    plot_feature_distribution(labeled_data, features[:5])
    plot_model_accuracies(accuracies)
    plot_confusion_matrix(y_test, y_pred, classes, best_model_name)
    plot_classification_comparison(labeled_data, y_test, y_pred, features[:3])
    plot_feature_correlation(labeled_data)
    plot_time_series_features(labeled_data, features[:3])

    # 绘制特征重要性图
    plot_feature_importance(best_model, features, best_model_name)

    logger.info("所有可视化图表已生成并保存")