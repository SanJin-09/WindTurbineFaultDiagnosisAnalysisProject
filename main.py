import warnings
import time
import sys
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.main.analysis.config import LOG_LEVEL, LOG_DIR, LOG_TO_FILE
from src.main.analysis.logging_utils import setup_logger
from src.main.analysis.data_preprocessing import load_data, preprocess_data
from src.main.analysis.feature_engineering import extract_features, prepare_features, select_features
from src.main.analysis.model_training import split_data, train_models
from src.main.analysis.plot import graphics_drawing
from src.main.analysis.dashboard import DataDashboard
from src.main.utils import save_model, print_summary

warnings.filterwarnings('ignore')

def run_analysis():
    logger = setup_logger('turbine_fault', LOG_LEVEL, LOG_TO_FILE, LOG_DIR)

    start_time = time.time()
    logger.info("=== 风力涡轮机故障诊断分析开始 ===")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. 加载数据
        logger.info("正在加载数据...")
        scada_data, fault_data = load_data(logger)

        # 2. 数据预处理
        logger.info("正在进行数据预处理...")
        labeled_data, fault_distribution = preprocess_data(scada_data, fault_data, logger)

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

        # 9. 图形绘制
        logger.info("正在绘制数据图像...")
        graphics_drawing(labeled_data, fault_distribution, best_model, best_model_name, accuracies, y_test, y_pred)

        # 10. 打印摘要
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
        logger.info(f"分析完成! 总用时: {total_time:.2f}秒")
        logger.info("=== 风力涡轮机故障诊断分析结束 ===")

        return True

    except Exception as e:
        logger.error(f"分析过程出错: {str(e)}", exc_info=True)
        print(f"分析错误: {str(e)}")
        logger.info("=== 风力涡轮机故障诊断分析异常终止 ===")
        return False


def start_dashboard():
    """启动图形界面"""
    print("正在启动风力涡轮机故障诊断系统...")

    try:
        # 创建仪表盘实例
        dashboard = DataDashboard()
        print("仪表盘启动成功！")

        # 运行仪表盘
        dashboard.run()
        return True

    except Exception as e:
        print(f"仪表盘启动失败: {e}")
        return False


def main():
    """主程序入口"""
    print("风力涡轮机故障诊断系统")
    print("=" * 50)
    print("请选择运行模式:")
    print("1. 启动图形界面 ")
    print("2. 运行完整分析流程")
    print("3. 先运行分析，后启动界面")
    print("4. 退出")

    try:
        choice = input("请输入选择 (1-4): ").strip()

        if choice == '1':
            start_dashboard()

        elif choice == '2':
            success = run_analysis()
            if success:
                print("分析完成！结果已保存。")
            else:
                print("分析失败！")
            input("按回车键退出...")

        elif choice == '3':
            # 先分析后启动GUI
            print("正在运行分析...")
            success = run_analysis()

            if success:
                print("分析完成！正在启动图形界面...")
                start_dashboard()
            else:
                print("分析失败，无法启动界面。")
                input("按回车键退出...")

        elif choice == '4':
            print("退出程序。")
            return

        else:
            print("无效选择，启动默认图形界面...")
            start_dashboard()

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"程序执行出现未预期的错误: {e}")
        input("按回车键退出...")


if __name__ == "__main__":
    main()
