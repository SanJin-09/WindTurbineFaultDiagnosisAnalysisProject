import os
import sys
import pandas as pd
import numpy as np
import queue
import threading
import time
import traceback
import logging
import tempfile
import shutil
import zipfile
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ==========================
# 配置部分 - 只自动化数据路径
# ==========================
# 请根据您的实际数据路径设置以下变量
SCADA_DATA_PATH = r"D:\WindTurbineFaultDiagnosisAnalysisProject\assets\scada_data.csv"  # SCADA数据路径
FAULT_DATA_PATH = r"D:\WindTurbineFaultDiagnosisAnalysisProject\assets\fault_data.csv"  # 故障数据路径
# ==========================

# 确保项目根目录在系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
try:
    from src.main.analysis.data_preprocessing import load_data, preprocess_data
    from src.main.analysis.model_training import train_models
    from src.main.analysis.plot import graphics_drawing
    import src.main.analysis.config as config
except ImportError:
    # 如果从不同目录运行，尝试直接导入
    from data_preprocessing import load_data, preprocess_data
    from model_training import train_models
    from plot import graphics_drawing
    import config


class FaultDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("风力涡轮机故障诊断系统")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f5f9ff')

        # 初始化状态
        self.processed_data = None
        self.model_results = None
        self.analysis_started = False
        self.analysis_complete = False
        self.log_output = ""
        self.error_occurred = False
        self.error_message = ""
        self.scada_path = SCADA_DATA_PATH  # 使用预设的SCADA路径
        self.fault_path = FAULT_DATA_PATH  # 使用预设的故障路径
        self.config = config
        self.log_queue = queue.Queue()  # 初始化日志队列

        # 创建临时目录
        self.temp_dir = self.create_temp_dir()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # 创建UI
        self.create_widgets()

        # 日志捕获
        self.setup_logging()

        # 在UI中显示预设路径
        self.scada_path_var.set(SCADA_DATA_PATH)
        self.fault_path_var.set(FAULT_DATA_PATH)

        # 开始日志队列处理
        self.process_log_queue()

    def create_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.log_queue.put(f"创建临时目录: {temp_dir}")
        return temp_dir

    def delete_temp_dir(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.log_queue.put(f"删除临时目录: {self.temp_dir}")

    def setup_logging(self):
        self.log_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        self.log_handler.setStream(self)  # 设置自定义流

        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.log_handler)

    def log(self, message):
        """线程安全的日志记录方法"""
        self.log_queue.put(message)

    def process_log_queue(self):
        """处理日志队列中的消息"""
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                self.log_output += message + "\n"
                self.update_log_display()
        except queue.Empty:
            pass

        # 每200毫秒检查一次队列
        self.root.after(200, self.process_log_queue)

    def write(self, message):
        """实现文件类接口用于日志捕获"""
        if message.strip():  # 避免空消息
            self.log_queue.put(message.strip())

    def flush(self):
        pass

    def update_log_display(self):
        if hasattr(self, 'log_text'):
            self.log_text.configure(state='normal')
            self.log_text.delete('1.0', tk.END)
            self.log_text.insert(tk.END, self.log_output)
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)  # 滚动到末尾

    def download_results(self):
        """下载分析结果为ZIP文件"""
        if not self.analysis_complete or not self.processed_data or not self.model_results:
            return

        # 打开文件对话框让用户选择保存位置
        zip_path = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".zip",
            filetypes=(("ZIP压缩文件", "*.zip"), ("所有文件", "*.*"))
        )

        if not zip_path:
            return

        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # 添加所有图像
                if os.path.exists(self.images_dir):
                    for file in os.listdir(self.images_dir):
                        if file.endswith('.png'):
                            file_path = os.path.join(self.images_dir, file)
                            zipf.write(file_path, os.path.join("images", file))
                else:
                    self.log("警告: 图表目录不存在")

                # 添加处理后的数据
                if self.processed_data:
                    labeled_data, _ = self.processed_data
                    data_path = os.path.join(self.temp_dir, "processed_data.csv")
                    labeled_data.to_csv(data_path, index=False)
                    zipf.write(data_path, "processed_data.csv")
                else:
                    self.log("警告: 处理后的数据不可用")

                # 添加日志文件
                log_path = os.path.join(self.temp_dir, "analysis_log.txt")
                with open(log_path, "w") as log_file:
                    log_file.write(self.log_output)
                zipf.write(log_path, "analysis_log.txt")

                # 添加参数配置
                config_path = os.path.join(self.temp_dir, "config_summary.txt")
                with open(config_path, "w") as config_file:
                    config_file.write(f"SCADA数据路径: {self.scada_path}\n")
                    config_file.write(f"故障数据路径: {self.fault_path}\n")
                    config_file.write(f"故障前时间窗口: {self.config.WINDOW_BEFORE}小时\n")
                    config_file.write(f"故障后时间窗口: {self.config.WINDOW_AFTER}小时\n")
                    config_file.write(f"测试集比例: {self.config.TEST_SIZE}\n")
                    config_file.write(f"随机森林树数量: {self.config.RF_N_ESTIMATORS}\n")
                    config_file.write(f"梯度提升树数量: {self.config.GB_N_ESTIMATORS}\n")
                zipf.write(config_path, "config_summary.txt")

            messagebox.showinfo("下载完成", f"分析结果已保存到:\n{zip_path}")
            self.log(f"分析结果已导出到: {zip_path}")

        except Exception as e:
            messagebox.showerror("导出错误", f"导出结果时发生错误: {str(e)}")
            self.log(f"⚠️ 导出结果时发生错误: {str(e)}")

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 标题
        title_label = ttk.Label(main_frame, text="风力涡轮机故障诊断系统",
                                font=("Helvetica", 24, "bold"), foreground="#1e3c72")
        title_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(main_frame, text="使用预设数据路径进行故障诊断分析",
                                   font=("Helvetica", 14), foreground="#2a5298")
        subtitle_label.pack(pady=(0, 20))

        # 创建分割布局
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # 左侧面板 - 控制面板
        control_frame = ttk.LabelFrame(paned_window, text="数据上传和参数设置")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        paned_window.add(control_frame, weight=30)

        # 右侧面板 - 结果展示
        result_frame = ttk.Frame(paned_window)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        paned_window.add(result_frame, weight=70)

        # 填充控制面板
        self.create_control_panel(control_frame)

        # 填充结果面板
        self.create_result_panel(result_frame)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_control_panel(self, parent):
        # 文件路径显示
        path_frame = ttk.LabelFrame(parent, text="数据路径")
        path_frame.pack(fill=tk.X, padx=10, pady=10)

        # SCADA数据路径
        scada_frame = ttk.Frame(path_frame)
        scada_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(scada_frame, text="SCADA数据路径:").pack(side=tk.LEFT, padx=(0, 5))
        self.scada_path_var = tk.StringVar()
        scada_entry = ttk.Entry(scada_frame, textvariable=self.scada_path_var, state='readonly', width=50)
        scada_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 故障数据路径
        fault_frame = ttk.Frame(path_frame)
        fault_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        ttk.Label(fault_frame, text="故障数据路径:").pack(side=tk.LEFT, padx=(0, 5))
        self.fault_path_var = tk.StringVar()
        fault_entry = ttk.Entry(fault_frame, textvariable=self.fault_path_var, state='readonly', width=50)
        fault_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 参数设置部分
        params_frame = ttk.LabelFrame(parent, text="分析参数")
        params_frame.pack(fill=tk.X, padx=10, pady=10)

        # 故障前时间窗口
        window_before_frame = ttk.Frame(params_frame)
        window_before_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(window_before_frame, text="故障前时间窗口(小时):").pack(side=tk.LEFT)
        self.window_before_var = tk.DoubleVar(value=self.config.WINDOW_BEFORE)
        ttk.Scale(window_before_frame, from_=0.1, to=24.0, variable=self.window_before_var,
                  command=lambda v: self.update_var_display(self.window_before_var, v)).pack(side=tk.LEFT, padx=5,
                                                                                             fill=tk.X, expand=True)
        self.window_before_display = ttk.Label(window_before_frame, text=str(self.config.WINDOW_BEFORE))
        self.window_before_display.pack(side=tk.LEFT)

        # 故障后时间窗口
        window_after_frame = ttk.Frame(params_frame)
        window_after_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(window_after_frame, text="故障后时间窗口(小时):").pack(side=tk.LEFT)
        self.window_after_var = tk.DoubleVar(value=self.config.WINDOW_AFTER)
        ttk.Scale(window_after_frame, from_=0.1, to=6.0, variable=self.window_after_var,
                  command=lambda v: self.update_var_display(self.window_after_var, v)).pack(side=tk.LEFT, padx=5,
                                                                                            fill=tk.X, expand=True)
        self.window_after_display = ttk.Label(window_after_frame, text=str(self.config.WINDOW_AFTER))
        self.window_after_display.pack(side=tk.LEFT)

        # 测试集比例
        test_size_frame = ttk.Frame(params_frame)
        test_size_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(test_size_frame, text="测试集比例:").pack(side=tk.LEFT)
        self.test_size_var = tk.DoubleVar(value=self.config.TEST_SIZE)
        ttk.Scale(test_size_frame, from_=0.1, to=0.5, variable=self.test_size_var,
                  command=lambda v: self.update_var_display(self.test_size_var, v)).pack(side=tk.LEFT, padx=5,
                                                                                         fill=tk.X, expand=True)
        self.test_size_display = ttk.Label(test_size_frame, text=str(self.config.TEST_SIZE))
        self.test_size_display.pack(side=tk.LEFT)

        # 随机森林树数量
        rf_frame = ttk.Frame(params_frame)
        rf_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(rf_frame, text="随机森林树数量:").pack(side=tk.LEFT)
        self.rf_var = tk.IntVar(value=self.config.RF_N_ESTIMATORS)
        ttk.Scale(rf_frame, from_=10, to=500, variable=self.rf_var,
                  command=lambda v: self.update_var_display(self.rf_var, v)).pack(side=tk.LEFT, padx=5, fill=tk.X,
                                                                                  expand=True)
        self.rf_display = ttk.Label(rf_frame, text=str(self.config.RF_N_ESTIMATORS))
        self.rf_display.pack(side=tk.LEFT)

        # 梯度提升树数量
        gb_frame = ttk.Frame(params_frame)
        gb_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        ttk.Label(gb_frame, text="梯度提升树数量:").pack(side=tk.LEFT)
        self.gb_var = tk.IntVar(value=self.config.GB_N_ESTIMATORS)
        ttk.Scale(gb_frame, from_=10, to=500, variable=self.gb_var,
                  command=lambda v: self.update_var_display(self.gb_var, v)).pack(side=tk.LEFT, padx=5, fill=tk.X,
                                                                                  expand=True)
        self.gb_display = ttk.Label(gb_frame, text=str(self.config.GB_N_ESTIMATORS))
        self.gb_display.pack(side=tk.LEFT)

        # 开始分析按钮
        start_frame = ttk.Frame(parent)
        start_frame.pack(fill=tk.X, padx=10, pady=10)
        self.start_button = ttk.Button(start_frame, text="开始分析", command=self.start_analysis)
        self.start_button.pack(fill=tk.X)

        # 下载结果按钮
        download_frame = ttk.Frame(parent)
        download_frame.pack(fill=tk.X, padx=10, pady=10)
        self.download_button = ttk.Button(download_frame, text="下载分析结果", command=self.download_results,
                                          state=tk.DISABLED)
        self.download_button.pack(fill=tk.X)

        # 日志显示
        log_frame = ttk.LabelFrame(parent, text="日志输出")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.configure(state='disabled')

    def create_result_panel(self, parent):
        # 创建笔记本样式的结果面板
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 结果摘要标签页
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="结果摘要")

        # 数据卡片
        self.data_frame = ttk.LabelFrame(self.summary_tab, text="数据摘要")
        self.data_frame.pack(fill=tk.X, padx=10, pady=10)

        # 模型性能卡片
        self.model_frame = ttk.LabelFrame(self.summary_tab, text="模型性能")
        self.model_frame.pack(fill=tk.X, padx=10, pady=10)

        # 故障分布卡片
        self.fault_frame = ttk.LabelFrame(self.summary_tab, text="故障分布")
        self.fault_frame.pack(fill=tk.X, padx=10, pady=10)

        # 可视化标签页
        self.visualization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_tab, text="可视化")

        # 创建画布用于显示图表
        self.canvas_frame = ttk.Frame(self.visualization_tab)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 图表选择
        self.plot_selector_frame = ttk.Frame(self.visualization_tab)
        self.plot_selector_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.plot_selector_frame, text="选择图表:").pack(side=tk.LEFT)
        self.plot_var = tk.StringVar()
        plots = ["故障分布", "特征分布", "模型准确率对比", "特征重要性",
                 "特征相关性", "时间序列特征", "混淆矩阵", "特征随时间变化"]
        self.plot_selector = ttk.Combobox(self.plot_selector_frame, textvariable=self.plot_var, values=plots)
        self.plot_selector.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.plot_selector.current(0)

        # 修复：确保方法存在后再绑定事件
        if hasattr(self, 'show_selected_plot'):
            self.plot_selector.bind('<<ComboboxSelected>>', self.show_selected_plot)
        else:
            # 设置一个占位函数
            def placeholder(event):
                messagebox.showwarning("功能未就绪", "图表选择功能尚未准备就绪")

            self.plot_selector.bind('<<ComboboxSelected>>', placeholder)

        ttk.Button(self.plot_selector_frame, text="显示", command=self.show_selected_plot).pack(side=tk.LEFT)

    def update_var_display(self, var, value):
        """更新滑块值显示"""
        if var == self.window_before_var:
            self.window_before_display.config(text=f"{float(value):.2f}")
        elif var == self.window_after_var:
            self.window_after_display.config(text=f"{float(value):.2f}")
        elif var == self.test_size_var:
            self.test_size_display.config(text=f"{float(value):.2f}")
        elif var == self.rf_var:
            self.rf_display.config(text=str(int(float(value))))
        elif var == self.gb_var:
            self.gb_display.config(text=str(int(float(value))))

    def start_analysis(self):
        # 更新配置参数
        self.config.WINDOW_BEFORE = self.window_before_var.get()
        self.config.WINDOW_AFTER = self.window_after_var.get()
        self.config.TEST_SIZE = self.test_size_var.get()
        self.config.RF_N_ESTIMATORS = self.rf_var.get()
        self.config.GB_N_ESTIMATORS = self.gb_var.get()
        self.config.SCADA_DATA_PATH = self.scada_path
        self.config.FAULT_DATA_PATH = self.fault_path

        # 禁用开始按钮
        self.start_button.config(state=tk.DISABLED)

        # 重置状态
        self.processed_data = None
        self.model_results = None
        self.analysis_started = True
        self.analysis_complete = False
        self.error_occurred = False
        self.error_message = ""
        self.log_output = ""
        self.update_log_display()
        self.status_var.set("分析中...")
        self.log("分析启动中...")

        # 创建分析线程
        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()

        # 开始进度监控
        self.monitor_analysis()

    def run_analysis(self):
        try:
            # 步骤1: 加载数据
            self.log("步骤 1/4: 加载数据...")

            scada_data, fault_data = load_data()

            self.log(f"SCADA数据路径: {self.scada_path}")
            self.log(f"故障数据路径: {self.fault_path}")
            self.log(f"加载了 {len(scada_data)} 条SCADA记录和 {len(fault_data)} 条故障记录")
            self.log("数据加载成功!")

            # 步骤2: 数据预处理
            self.log("\n步骤 2/4: 数据预处理...")

            labeled_data, fault_distribution = preprocess_data(scada_data, fault_data)
            self.processed_data = (labeled_data, fault_distribution)

            self.log(f"预处理后的数据记录数: {len(labeled_data)}")
            self.log(f"检测到的故障类型数: {len(fault_distribution)}")
            self.log("数据预处理成功!")

            # 步骤3: 模型训练
            self.log("\n步骤 3/4: 模型训练...")

            # 准备特征和标签
            numeric_cols = labeled_data.select_dtypes(include=np.number).columns
            features = [col for col in numeric_cols if col != 'Fault']
            X = labeled_data[features]
            y = labeled_data['Fault']

            # 拆分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=42
            )

            # 调用训练函数
            best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = train_models(X_train, y_train,
                                                                                                  X_test, y_test)
            self.model_results = (best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred)

            self.log("\n模型训练成功!")

            # 修复的日志输出 - 不尝试获取keys()
            if hasattr(accuracies, 'keys'):
                model_names = ", ".join(str(name) for name in accuracies.keys())
                self.log(f"使用的模型: {model_names}")
            else:
                self.log("使用的模型信息不可用")

            self.log(f"最佳模型: {best_model_name}, 准确率: {best_accuracy:.4f}")

            # 步骤4: 可视化
            self.log("\n步骤 4/4: 生成可视化图表...")

            # 创建子线程进行可视化
            self.log("启动可视化线程...")
            graphics_thread = threading.Thread(
                target=self.run_visualization,
                args=(labeled_data, fault_distribution, best_model, best_model_name, accuracies, y_test, y_pred),
                daemon=True
            )
            graphics_thread.start()

            # 等待可视化完成
            graphics_thread.join(timeout=600)  # 设置10分钟超时
            if graphics_thread.is_alive():
                self.log("警告: 可视化任务超时")

            self.analysis_complete = True
            self.log("分析完成!")

        except Exception as e:
            self.error_occurred = True
            self.error_message = str(e)
            self.log(f"\n⚠️ 错误: {str(e)}")
            self.log(traceback.format_exc())

    def run_visualization(self, labeled_data, fault_distribution, best_model, best_model_name, accuracies, y_test,
                          y_pred):
        """在单独线程中运行可视化以避免阻塞主线程"""
        try:
            self.log("开始生成可视化图表...")
            self.log(f"图表将保存到: {self.images_dir}")

            # 确保保存目录存在
            os.makedirs(self.images_dir, exist_ok=True)

            # 保存当前工作目录
            original_dir = os.getcwd()

            # 切换到临时目录，因为plot.py可能只保存到当前目录
            os.chdir(self.temp_dir)

            # 调用可视化函数
            graphics_drawing(
                labeled_data,
                fault_distribution,
                best_model,
                best_model_name,
                accuracies,
                y_test,
                y_pred
            )

            # 切换回原始目录
            os.chdir(original_dir)

            # 检查生成的图表文件
            if os.path.exists(self.images_dir):
                image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
                self.log(f"生成图表: {', '.join(image_files)}")
            else:
                self.log("警告: 图表目录不存在")

            self.log("可视化完成!")

        except Exception as e:
            self.log(f"可视化过程中发生错误: {str(e)}")
            self.log(traceback.format_exc())

    def monitor_analysis(self):
        if self.analysis_complete or self.error_occurred:
            self.analysis_started = False
            self.start_button.config(state=tk.NORMAL)

            if self.analysis_complete:
                self.status_var.set("分析完成!")
                self.download_button.config(state=tk.NORMAL)
                self.update_summary()
                self.show_plot("故障分布")  # 默认显示第一个图表
            elif self.error_occurred:
                self.status_var.set(f"分析失败: {self.error_message}")
        else:
            # 继续监控
            self.root.after(500, self.monitor_analysis)

    def update_summary(self):
        if not self.processed_data or not self.model_results:
            return

        # 解包模型结果
        try:
            labeled_data, fault_distribution = self.processed_data
            best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = self.model_results

            # 清除旧的摘要信息
            for frame in [self.data_frame, self.model_frame, self.fault_frame]:
                for widget in frame.winfo_children():
                    widget.destroy()

            # 数据摘要
            ttk.Label(self.data_frame, text=f"总样本数: {len(labeled_data)}").pack(anchor=tk.W, padx=10, pady=5)
            ttk.Label(self.data_frame, text=f"特征数量: {len(labeled_data.columns) - 2}").pack(anchor=tk.W, padx=10,
                                                                                               pady=5)
            ttk.Label(self.data_frame, text=f"故障类型数: {len(fault_distribution)}").pack(anchor=tk.W, padx=10, pady=5)

            # 模型性能
            ttk.Label(self.model_frame, text=f"最佳模型: {best_model_name}").pack(anchor=tk.W, padx=10, pady=5)
            ttk.Label(self.model_frame, text=f"准确率: {best_accuracy:.4f}").pack(anchor=tk.W, padx=10, pady=5)
            ttk.Label(self.model_frame, text=f"测试集大小: {len(y_test)}").pack(anchor=tk.W, padx=10, pady=5)

            # 显示所有模型准确率
            if hasattr(accuracies, 'items'):
                for model_name, accuracy in accuracies.items():
                    ttk.Label(self.model_frame, text=f"{model_name}准确率: {accuracy:.4f}").pack(anchor=tk.W, padx=10,
                                                                                                 pady=2)
            else:
                ttk.Label(self.model_frame, text="无准确率数据").pack(anchor=tk.W, padx=10, pady=2)

            # 故障分布
            for fault, count in fault_distribution.items():
                ttk.Label(self.fault_frame, text=f"{fault}: {count}").pack(anchor=tk.W, padx=10, pady=2)

        except Exception as e:
            self.log(f"更新摘要时出错: {str(e)}")

    def show_plot(self, plot_name):
        if not self.analysis_complete or not self.processed_data or not self.model_results:
            return

        # 清除当前图表
        self.fig.clear()

        try:
            # 获取模型名称（用于特征重要性文件名）
            if hasattr(self, 'model_results') and len(self.model_results) > 1:
                best_model_name = self.model_results[1].replace(" ", "_").lower()
            else:
                best_model_name = "unknown_model"

            # 映射图表名称到实际文件名
            plot_mapping = {
                "故障分布": "fault_distribution.png",
                "特征分布": "feature_distribution.png",
                "模型准确率对比": "model_accuracies.png",
                "特征重要性": f"{best_model_name}_feature_importance.png",
                "特征相关性": "feature_correlation.png",
                "时间序列特征": "time_series_features.png",
                "混淆矩阵": "confusion_matrix.png",
                "特征随时间变化": "classification_comparison.png"
            }

            filename = plot_mapping.get(plot_name)
            if not filename:
                raise ValueError(f"未知的图表名称: {plot_name}")

            img_path = os.path.join(self.images_dir, filename)
            self.log(f"尝试加载图表文件: {img_path}")

            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax = self.fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(plot_name)
                self.log(f"显示图表: {plot_name}")
                self.canvas.draw()
            else:
                # 尝试查找替换文件名（处理不同拼写）
                alt_filenames = [
                    f"feature_importance_{best_model_name}.png",
                    f"feature_importance.png",
                    f"{plot_name.lower().replace(' ', '_')}.png"
                ]

                # 查找存在的文件
                found = False
                for alt in alt_filenames:
                    alt_path = os.path.join(self.images_dir, alt)
                    if os.path.exists(alt_path):
                        img = Image.open(alt_path)
                        ax = self.fig.add_subplot(111)
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(plot_name)
                        self.log(f"找到备用图表文件: {alt_path}")
                        self.canvas.draw()
                        found = True
                        break

                if not found:
                    # 找不到任何匹配文件
                    ax = self.fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"未找到 {plot_name} 图表文件\n路径: {img_path}\n\n尝试了以下文件:\n" +
                            "\n".join([alt for alt in alt_filenames]),
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=10)
                    self.log(f"⚠️ 未找到图表文件: {img_path}")
                    self.log(f"尝试了以下备选文件名: {alt_filenames}")
                    self.log(f"实际目录内容: {os.listdir(self.images_dir)}")
                    self.canvas.draw()

        except Exception as e:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"显示图表时出错: {str(e)}\n{traceback.format_exc()}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10)
            self.log(f"显示图表时出错: {str(e)}")
            self.log(traceback.format_exc())
            self.canvas.draw()

    def show_selected_plot(self, event=None):
        """处理图表选择事件"""
        if hasattr(self, 'plot_var'):
            plot_name = self.plot_var.get()
            if plot_name:
                self.show_plot(plot_name)
        else:
            messagebox.showwarning("功能未就绪", "图表选择功能尚未准备就绪")

    def on_closing(self):
        """关闭应用程序时的清理工作"""
        self.delete_temp_dir()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaultDiagnosisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()