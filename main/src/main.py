import warnings

warnings.filterwarnings('ignore')

from data_preprocessing import load_data, preprocess_data
from feature_engineering import extract_features, prepare_features, select_features
from model_training import split_data, train_models
from utils import save_model, print_summary


def main():
    print("=== 风力涡轮机故障诊断系统 ===\n")

    # 1. 加载数据
    scada_data, fault_data = load_data()

    # 2. 数据预处理
    labeled_data, fault_distribution = preprocess_data(scada_data, fault_data)

    # 3. 特征工程
    features = extract_features(labeled_data)

    # 4. 准备特征和标签
    x, y, le = prepare_features(features)

    # 5. 分割数据
    x_train, x_test, y_train, y_test = split_data(x, y)

    # 6. 特征选择
    x_train_selected, x_test_selected, scaler, selector, important_features = select_features(x_train, y_train, x_test)

    # 7. 训练和评估模型
    best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = train_models(
        x_train_selected, y_train, x_test_selected, y_test
    )

    # 8. 保存模型
    save_model(best_model, scaler, selector, le)

    # 9. 打印摘要
    print_summary(
        fault_distribution,
        features.shape,
        x_train_selected.shape[1],
        important_features,
        best_model_name,
        best_accuracy,
        accuracies
    )


if __name__ == "__main__":
    main()
