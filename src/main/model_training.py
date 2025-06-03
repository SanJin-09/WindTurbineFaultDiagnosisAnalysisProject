"""
Author: <WU Xinyan>
模型训练和评估模块
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from config import (TEST_SIZE, RANDOM_STATE, RF_N_ESTIMATORS, GB_N_ESTIMATORS, MLP_HIDDEN_LAYERS, MLP_MAX_ITER)

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")
    return x_train, x_test, y_train, y_test


def train_models(x_train, y_train, x_test, y_test):
    print("\n正在训练和评估模型...")
    accuracies = {}

    # 1. 随机森林
    print("\n训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林准确率: {rf_acc:.4f}")
    accuracies['Random Forest'] = rf_acc

    # 2. 梯度提升树
    print("\n训练梯度提升树模型...")
    gb_model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        random_state=RANDOM_STATE
    )
    gb_model.fit(x_train, y_train)
    y_pred_gb = gb_model.predict(x_test)
    gb_acc = accuracy_score(y_test, y_pred_gb)
    print(f"梯度提升树准确率: {gb_acc:.4f}")
    accuracies['Gradient Boosting'] = gb_acc

    # 3. 神经网络
    print("\n训练神经网络模型...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE
    )
    mlp_model.fit(x_train, y_train)
    y_pred_mlp = mlp_model.predict(x_test)
    mlp_acc = accuracy_score(y_test, y_pred_mlp)
    print(f"神经网络准确率: {mlp_acc:.4f}")
    accuracies['Neural Network'] = mlp_acc

    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]

    if best_model_name == 'Random Forest':
        best_model = rf_model
        y_pred_best = y_pred_rf
    elif best_model_name == 'Gradient Boosting':
        best_model = gb_model
        y_pred_best = y_pred_gb
    else:
        best_model = mlp_model
        y_pred_best = y_pred_mlp

    print(f"\n最佳模型: {best_model_name} (准确率: {best_accuracy:.4f})")

    # 报告输出（不稳定）
    print("\n最佳模型的分类报告:")
    class_report = classification_report(y_test, y_pred_best)
    print(class_report)

    return best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred_best
