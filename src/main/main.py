import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

from src.main.analysis.config import LOG_LEVEL, LOG_DIR, LOG_TO_FILE
from src.main.analysis.logging_utils import setup_logger
from src.main.analysis.data_preprocessing import load_data, preprocess_data
from src.main.analysis.feature_engineering import extract_features, prepare_features, select_features
from src.main.analysis.model_training import split_data, train_models
from utils import save_model, print_summary


def main():

    logger = setup_logger('turbine_fault', LOG_LEVEL, LOG_TO_FILE, LOG_DIR)

    start_time = time.time()
    logger.info("=== 风力涡轮机故障诊断系统启动 ===")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. 加载数据
        logger.info("正在加载数据...")
        scada_data, fault_data = load_data()

        # 2. 数据预处理
        logger.info("正在进行数据预处理...")
        labeled_data, fault_distribution = preprocess_data(scada_data, fault_data)

        # 3. 特征工程
        logger.info("正在进行特征工程...")
        features = extract_features(labeled_data)

        # 4. 准备特征和标签
        logger.info("正在准备特征和标签...")
        x, y, le = prepare_features(features)

        # 5. 分割数据
        logger.info("正在分割训练集和测试集...")
        x_train, x_test, y_train, y_test = split_data(x, y)

        # 6. 特征选择
        logger.info("正在进行特征选择...")
        x_train_selected, x_test_selected, scaler, selector, important_features = select_features(
            x_train, y_train, x_test
        )

        # 7. 训练和评估模型
        logger.info("正在训练模型...")
        best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = train_models(
            x_train_selected, y_train, x_test_selected, y_test
        )

        # 8. 保存模型
        logger.info("正在保存模型...")
        save_model(best_model, scaler, selector, le)

        # 9. 打印摘要
        logger.info("生成结果摘要...")
        print_summary(
            fault_distribution,
            features.shape,
            x_train_selected.shape[1],
            important_features,
            best_model_name,
            best_accuracy,
            accuracies,
        )

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"处理完成! 总用时: {total_time:.2f}秒")
        logger.info("=== 风力涡轮机故障诊断系统结束 ===")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"错误: {str(e)}")
        logger.info("=== 风力涡轮机故障诊断系统异常终止 ===")
        raise


if __name__ == "__main__":
    main()
