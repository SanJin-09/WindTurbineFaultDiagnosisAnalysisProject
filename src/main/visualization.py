import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# PyTorch相关导入
from sklearn.metrics import confusion_matrix

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

def plot_correlation_heatmap(x):
    """绘制特征相关性热力图"""
    plt.figure(figsize=(18, 14))
    corr = x.corr()
    # 只显示相关性绝对值较高的部分
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # 只显示相关性绝对值大于0.5的单元格
    corr_abs = corr.abs()
    high_corr_mask = corr_abs > 0.5
    corr_masked = corr.where(high_corr_mask & ~mask, np.nan)

    sns.heatmap(corr_masked, cmap='cool_warm', annot=True, fmt=".2f", annot_kws={"size": 8},
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

def graphics_drawing(results, y_test, le, models, x):
    try:
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
                    'Feature': x.columns,
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