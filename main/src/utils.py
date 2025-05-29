"""
Author: <WU Xinyan>
utils包含所有工具函数
后续可拓展
"""
import joblib


def get_column_names():
    column_names = [
        'DateTime', 'Time', 'Status', 'WindSpeed', 'WindSpeedAvg', 'WindSpeedMin',
        'WindDirection', 'WindDirectionAvg', 'WindDirectionMin', 'Power', 'PowerMax',
        'PowerMin', 'Nacelle', 'Revolution', 'RevolutionMax', 'RevolutionMin',
        'TempAmbient', 'TempGenerator', 'TempGearbox', 'TempMainBearing',
        'Vibration1', 'Vibration2', 'Vibration3', 'Vibration4',
        'State1', 'State2', 'State3', 'State4', 'State5', 'State6', 'State7', 'State8',
        'State9', 'State10', 'State11', 'State12', 'State13', 'State14', 'State15', 'State16',
        'Param1', 'Param2', 'Param3', 'Param4', 'Param5', 'Param6', 'Param7', 'Param8',
        'Param9', 'Param10', 'Param11', 'Param12', 'Param13', 'Param14', 'Param15', 'Param16',
        'Param17', 'Param18', 'Param19', 'Param20', 'Param21', 'Param22', 'Param23', 'Param24',
        'Param25', 'Param26', 'Param27', 'Param28', 'Param29', 'Param30', 'TurbineType', 'Pitch', 'Efficiency'
    ]
    return column_names


def save_model(model, scaler, selector, encoder):
    from config import MODEL_PATH, SCALER_PATH, SELECTOR_PATH, ENCODER_PATH

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selector, SELECTOR_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print(f"模型已保存到 {MODEL_PATH}")
    print(f"缩放器已保存到 {SCALER_PATH}")
    print(f"特征选择器已保存到 {SELECTOR_PATH}")
    print(f"标签编码器已保存到 {ENCODER_PATH}")


def load_model():
    from config import MODEL_PATH, SCALER_PATH, SELECTOR_PATH, ENCODER_PATH

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, selector, encoder


def print_summary(fault_counts, features_shape, selected_features, important_features,
                  best_model_name, best_accuracy, accuracies):
    """
    下面是一个摘要
    """
    print(f"""
========================= 风力涡轮机故障诊断结果 =========================

1. 数据概述:
   - 主要故障类型: {', '.join(fault_counts.index[:3].tolist())}

2. 特征工程:
   - 创建特征总数: {features_shape[1] - 1}
   - 选择特征数量: {selected_features}
   - 最重要特征: {', '.join(important_features[:3])}

3. 模型性能:
   - 最佳模型: {best_model_name}
   - 准确率: {best_accuracy:.4f}
   - 各模型对比:
     * 随机森林: {accuracies['Random Forest']:.4f}
     * 梯度提升树: {accuracies['Gradient Boosting']:.4f}
     * XGBoost: {accuracies['XGBoost']:.4f}
     * 神经网络: {accuracies['Neural Network']:.4f}

4. 结论与建议:
   - 本模型能够有效识别不同类型的风机故障
   - 最重要的指标是{important_features[0]}，监控该指标有助于提前预警
   - 基于{best_model_name}的预测模型可以实时部署用于风机健康监控

=======================================================================
""")