"""
Author: <WU Xinyan & Assistant>
风力涡轮机故障诊断交互式数据仪表盘
模块化设计，集成到主程序中
"""
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox, scrolledtext

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from src.main.analysis.config import *
from src.main.analysis.data_preprocessing import load_data, preprocess_data
from src.main.analysis.feature_engineering import extract_features, prepare_features, select_features

from src.main.analysis.logging_utils import get_logger
from src.main.analysis.model_training import split_data, train_models
from src.main.analysis.plot import graphics_drawing


class DataDashboard:
    """风力涡轮机故障诊断数据仪表盘"""

    def __init__(self, root=None):
        """初始化仪表盘"""
        if root is None:
            self.root = tk.Tk()
            self.standalone = True
        else:
            self.root = root
            self.standalone = False

        self.setup_window()
        self.init_variables()
        self.setup_logger()
        self.create_ui()

        # 自动加载数据（如果文件存在）
        self.auto_load_data()

    def setup_window(self):
        """设置窗口属性"""
        self.root.title("风力涡轮机故障诊断系统 - 数据仪表盘")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)

    def init_variables(self):
        """初始化变量"""
        self.logger = None
        self.log_queue = queue.Queue()

        self.raw_data = None
        self.processed_data = None
        self.features_data = None
        self.model_results = None
        self.analysis_complete = False

        self.current_plot = None
        self.plots_available = []

        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

    def setup_logger(self):
        """设置日志系统"""
        self.logger = get_logger('dashboard')
        self.logger.info("仪表盘已启动")

    def create_ui(self):
        """创建用户界面"""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.create_header()
        self.create_main_content()
        self.create_status_bar()

        self.process_log_queue()

    def create_header(self):
        """创建头部区域"""
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(
            header_frame,
            text="风力涡轮机故障诊断系统",
            font=('Microsoft YaHei', 20, 'bold')
        )
        title_label.pack(pady=10)

        control_frame = ttk.Frame(header_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            control_frame,
            text="加载数据",
            command=self.load_data_async,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="开始分析",
            command=self.start_analysis,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="刷新图表",
            command=self.refresh_plots
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="导出结果",
            command=self.export_results
        ).pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

        self.progress_label = ttk.Label(control_frame, text="就绪")
        self.progress_label.pack(side=tk.RIGHT, padx=5)

    def create_main_content(self):
        """创建主要内容区域"""
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.create_left_panel()

        self.create_right_panel()

    def create_left_panel(self):
        """创建左侧控制面板"""
        left_frame = ttk.Frame(self.paned_window)
        left_frame.pack(fill=tk.BOTH, expand=True)
        self.paned_window.add(left_frame, weight=25)

        self.create_data_overview(left_frame)

        self.create_parameter_control(left_frame)

        self.create_log_display(left_frame)

    def create_right_panel(self):
        """创建右侧显示面板"""
        right_frame = ttk.Frame(self.paned_window)
        right_frame.pack(fill=tk.BOTH, expand=True)
        self.paned_window.add(right_frame, weight=75)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.create_data_tab()

        self.create_visualization_tab()

        self.create_results_tab()

    def create_data_overview(self, parent):
        """创建数据概览区域"""
        overview_frame = ttk.LabelFrame(parent, text="数据概览")
        overview_frame.pack(fill=tk.X, padx=5, pady=5)

        self.data_info = scrolledtext.ScrolledText(
            overview_frame,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.data_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.update_data_overview("等待数据加载...")

    def create_parameter_control(self, parent):
        """创建参数控制区域"""
        param_frame = ttk.LabelFrame(parent, text="分析参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(param_frame, text=f"故障前时间窗口: {WINDOW_BEFORE}小时").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(param_frame, text=f"故障后时间窗口: {WINDOW_AFTER}小时").pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(param_frame, text=f"测试集比例: {TEST_SIZE}").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(param_frame, text=f"随机森林估算器: {RF_N_ESTIMATORS}").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(param_frame, text=f"梯度提升估算器: {GB_N_ESTIMATORS}").pack(anchor=tk.W, padx=5, pady=2)

    def create_log_display(self, parent):
        """创建日志显示区域"""
        log_frame = ttk.LabelFrame(parent, text="系统日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_display = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_data_tab(self):
        """创建数据选项卡"""
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="数据表格")

        table_frame = ttk.Frame(self.data_tab)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        selector_frame = ttk.Frame(table_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(selector_frame, text="选择数据表:").pack(side=tk.LEFT, padx=5)

        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.table_var,
            values=["原始数据", "预处理数据", "特征数据"],
            state="readonly"
        )
        self.table_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_select)

        self.create_data_table(table_frame)

    def create_data_table(self, parent):
        """创建数据表格"""
        columns = ('索引', '列1', '列2', '列3', '列4', '列5')

        self.data_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)

        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor=tk.W)

        tree_scroll_v = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.data_tree.yview)
        tree_scroll_h = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.data_tree.xview)

        self.data_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)

        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)

    def create_visualization_tab(self):
        """创建可视化选项卡"""
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="数据可视化")

        control_frame = ttk.Frame(self.viz_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="选择图表:").pack(side=tk.LEFT, padx=5)

        self.plot_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(
            control_frame,
            textvariable=self.plot_var,
            values=[
                "故障分布图", "特征分布图", "模型准确率对比",
                "混淆矩阵", "特征重要性", "时间序列特征"
            ],
            state="readonly"
        )
        self.plot_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.plot_combo.bind('<<ComboboxSelected>>', self.on_plot_select)

        ttk.Button(
            control_frame,
            text="显示图表",
            command=self.show_selected_plot
        ).pack(side=tk.RIGHT, padx=5)

        self.create_plot_area()

    def create_plot_area(self):
        """创建图表显示区域"""
        plot_frame = ttk.Frame(self.viz_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def create_results_tab(self):
        """创建结果选项卡"""
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="分析结果")

        results_frame = ttk.Frame(self.results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        perf_frame = ttk.LabelFrame(results_frame, text="模型性能")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)

        self.performance_info = scrolledtext.ScrolledText(
            perf_frame,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.performance_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        feat_frame = ttk.LabelFrame(results_frame, text="重要特征")
        feat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.features_info = scrolledtext.ScrolledText(
            feat_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.features_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ttk.Frame(self.main_container)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_var = tk.StringVar(value="系统就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)

        self.time_var = tk.StringVar()
        time_label = ttk.Label(status_frame, textvariable=self.time_var)
        time_label.pack(side=tk.RIGHT, padx=5)

        self.update_time()

    def update_time(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(current_time)
        self.root.after(1000, self.update_time)

    def auto_load_data(self):
        """自动加载数据"""
        if os.path.exists(SCADA_DATA_PATH) and os.path.exists(FAULT_DATA_PATH):
            self.log_message("发现数据文件，自动加载中...")
            self.root.after(1000, self.load_data_async)
        else:
            self.log_message("未找到数据文件，请手动加载数据")

    def load_data_async(self):
        """异步加载数据"""
        self.update_status("正在加载数据...")
        self.progress_var.set(10)

        def load_worker():
            try:
                self.log_message("开始加载数据...")
                scada_data, fault_data = load_data(self.logger)

                self.raw_data = {
                    'scada': scada_data,
                    'fault': fault_data
                }

                self.root.after(0, lambda: self.on_data_loaded(True, "数据加载成功"))

            except Exception as e:
                self.root.after(0, lambda: self.on_data_loaded(False, f"数据加载失败: {str(e)}"))

        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()

    def on_data_loaded(self, success, message):
        """数据加载完成回调"""
        if success:
            self.progress_var.set(30)
            self.update_status("数据加载完成")
            self.log_message(message)

            scada_info = f"SCADA数据: {self.raw_data['scada'].shape[0]}行 x {self.raw_data['scada'].shape[1]}列"
            fault_info = f"故障数据: {self.raw_data['fault'].shape[0]}行 x {self.raw_data['fault'].shape[1]}列"

            overview_text = f"{scada_info}\n{fault_info}\n\n数据加载时间: {datetime.now().strftime('%H:%M:%S')}"
            self.update_data_overview(overview_text)

            self.update_data_table()

        else:
            self.progress_var.set(0)
            self.update_status("数据加载失败")
            self.log_message(message)
            messagebox.showerror("错误", message)

    def start_analysis(self):
        """开始完整分析"""
        if self.raw_data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        self.update_status("正在进行数据分析...")
        self.progress_var.set(40)

        def analysis_worker():
            try:
                self.log_message("开始数据预处理...")
                labeled_data, fault_distribution = preprocess_data(
                    self.raw_data['scada'],
                    self.raw_data['fault'],
                    self.logger
                )

                self.root.after(0, lambda: self.progress_var.set(50))

                self.log_message("开始特征工程...")
                features = extract_features(labeled_data)
                x, y, le = prepare_features(features)

                self.root.after(0, lambda: self.progress_var.set(60))

                self.log_message("分割数据集...")
                x_train, x_test, y_train, y_test = split_data(x, y)

                x_train_selected, x_test_selected, scaler, selector, important_features = select_features(
                    x_train, y_train, x_test
                )

                self.root.after(0, lambda: self.progress_var.set(80))

                self.log_message("开始模型训练...")
                best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = train_models(
                    x_train_selected, y_train, x_test_selected, y_test
                )

                self.root.after(0, lambda: self.progress_var.set(90))

                self.log_message("生成可视化图表...")
                graphics_drawing(
                    labeled_data, fault_distribution, best_model,
                    best_model_name, accuracies, y_test, y_pred
                )

                self.processed_data = labeled_data
                self.features_data = features
                self.model_results = {
                    'best_model': best_model,
                    'best_model_name': best_model_name,
                    'best_accuracy': best_accuracy,
                    'accuracies': accuracies,
                    'important_features': important_features,
                    'fault_distribution': fault_distribution,
                    'y_test': y_test,
                    'y_pred': y_pred
                }

                self.analysis_complete = True
                self.root.after(0, lambda: self.on_analysis_complete(True, "分析完成"))

            except Exception as e:
                self.root.after(0, lambda: self.on_analysis_complete(False, f"分析失败: {str(e)}"))

        thread = threading.Thread(target=analysis_worker, daemon=True)
        thread.start()

    def on_analysis_complete(self, success, message):
        """分析完成回调"""
        if success:
            self.progress_var.set(100)
            self.update_status("分析完成")
            self.log_message(message)

            self.update_results_display()

            self.plot_combo.current(0)
            self.show_selected_plot()

            messagebox.showinfo("成功", "数据分析完成！")

        else:
            self.progress_var.set(0)
            self.update_status("分析失败")
            self.log_message(message)
            messagebox.showerror("错误", message)

    def update_data_overview(self, text):
        """更新数据概览"""
        self.data_info.config(state=tk.NORMAL)
        self.data_info.delete(1.0, tk.END)
        self.data_info.insert(tk.END, text)
        self.data_info.config(state=tk.DISABLED)

    def update_data_table(self):
        """更新数据表格"""
        self.table_combo.current(0)
        self.on_table_select(None)

    def update_results_display(self):
        """更新结果显示"""
        if not self.model_results:
            return

        perf_text = f"""最佳模型: {self.model_results['best_model_name']}
准确率: {self.model_results['best_accuracy']:.4f}

所有模型准确率:
"""
        for model, acc in self.model_results['accuracies'].items():
            perf_text += f"  {model}: {acc:.4f}\n"

        self.performance_info.config(state=tk.NORMAL)
        self.performance_info.delete(1.0, tk.END)
        self.performance_info.insert(tk.END, perf_text)
        self.performance_info.config(state=tk.DISABLED)

        feat_text = "重要特征排序:\n\n"
        for i, feat in enumerate(self.model_results['important_features'][:10], 1):
            feat_text += f"{i:2d}. {feat}\n"

        self.features_info.config(state=tk.NORMAL)
        self.features_info.delete(1.0, tk.END)
        self.features_info.insert(tk.END, feat_text)
        self.features_info.config(state=tk.DISABLED)

    def on_table_select(self, event):
        """表格选择事件"""
        selection = self.table_var.get()

        if selection == "原始数据" and self.raw_data:
            self.display_dataframe(self.raw_data['scada'])
        elif selection == "预处理数据" and self.processed_data is not None:
            self.display_dataframe(self.processed_data)
        elif selection == "特征数据" and self.features_data is not None:
            self.display_dataframe(self.features_data)

    def display_dataframe(self, df):
        """在表格中显示DataFrame"""

        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        if df is None or df.empty:
            return

        columns = ['索引'] + list(df.columns)
        self.data_tree['columns'] = columns

        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120, anchor=tk.W)

        for i, (index, row) in enumerate(df.head(100).iterrows()):
            values = [str(index)] + [str(value) for value in row.values]
            self.data_tree.insert('', tk.END, values=values)

    def on_plot_select(self, event):
        """图表选择事件"""
        self.show_selected_plot()

    def show_selected_plot(self):
        """显示选定的图表"""
        if not self.analysis_complete:
            messagebox.showwarning("警告", "请先完成数据分析")
            return

        plot_type = self.plot_var.get()

        self.fig.clear()

        try:
            if plot_type == "故障分布图":
                self.plot_fault_distribution()
            elif plot_type == "特征分布图":
                self.plot_feature_distribution()
            elif plot_type == "模型准确率对比":
                self.plot_model_accuracies()
            elif plot_type == "混淆矩阵":
                self.plot_confusion_matrix()
            elif plot_type == "特征重要性":
                self.plot_feature_importance()
            elif plot_type == "时间序列特征":
                self.plot_time_series()

            self.canvas.draw()

        except Exception as e:
            self.log_message(f"图表显示错误: {str(e)}")
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"图表显示错误:\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()

    def plot_fault_distribution(self):
        """绘制故障分布图"""
        ax = self.fig.add_subplot(111)
        fault_dist = self.model_results['fault_distribution']

        bars = ax.bar(range(len(fault_dist)), fault_dist.values, color='skyblue')
        ax.set_xlabel('故障类型')
        ax.set_ylabel('样本数量')
        ax.set_title('故障分布')
        ax.set_xticks(range(len(fault_dist)))
        ax.set_xticklabels(fault_dist.index, rotation=45)

        for bar, value in zip(bars, fault_dist.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    str(value), ha='center', va='bottom')

        self.fig.tight_layout()

    def plot_feature_distribution(self):
        """绘制特征分布图"""
        if self.processed_data is None:
            return

        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns[:6]

        for i, col in enumerate(numeric_cols):
            ax = self.fig.add_subplot(2, 3, i + 1)
            ax.hist(self.processed_data[col].dropna(), bins=30, alpha=0.7, color='lightblue')
            ax.set_title(f'{col}分布')
            ax.set_xlabel(col)
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def plot_model_accuracies(self):
        """绘制模型准确率对比"""
        ax = self.fig.add_subplot(111)
        accuracies = self.model_results['accuracies']

        models = list(accuracies.keys())
        values = list(accuracies.values())

        bars = ax.bar(models, values, color=['#FF9999', '#66B2FF', '#99FF99'])
        ax.set_ylabel('准确率')
        ax.set_title('模型准确率对比')
        ax.set_ylim(0, 1)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        self.fig.tight_layout()

    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        ax = self.fig.add_subplot(111)

        y_test = self.model_results['y_test']
        y_pred = self.model_results['y_pred']

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel('真实标签')
        ax.set_xlabel('预测标签')
        ax.set_title('混淆矩阵')

        self.fig.tight_layout()

    def plot_feature_importance(self):
        """绘制特征重要性"""
        ax = self.fig.add_subplot(111)

        model = self.model_results['best_model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = self.model_results['important_features'][:10]  # 前10个重要特征

            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances[:len(features)], color='lightcoral')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('重要性')
            ax.set_title('特征重要性')
        else:
            ax.text(0.5, 0.5, '当前模型不支持特征重要性显示',
                    ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()

    def plot_time_series(self):
        """绘制时间序列特征"""
        ax = self.fig.add_subplot(111)

        if self.processed_data is None or 'DateTime' not in self.processed_data.columns:
            ax.text(0.5, 0.5, '无时间序列数据',
                    ha='center', va='center', transform=ax.transAxes)
            return

        data_sample = self.processed_data.head(1000)
        numeric_cols = data_sample.select_dtypes(include=[np.number]).columns[:3]

        for col in numeric_cols:
            ax.plot(data_sample['DateTime'], data_sample[col], label=col, alpha=0.7)

        ax.set_xlabel('时间')
        ax.set_ylabel('特征值')
        ax.set_title('时间序列特征变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def refresh_plots(self):
        """刷新图表"""
        if self.analysis_complete:
            self.show_selected_plot()
            self.log_message("图表已刷新")
        else:
            messagebox.showwarning("警告", "请先完成数据分析")

    def export_results(self):
        """导出结果"""
        if not self.analysis_complete:
            messagebox.showwarning("警告", "请先完成数据分析")
            return

        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
            )

            if filename:
                if filename.endswith('.csv'):
                    self.processed_data.to_csv(filename, index=False)
                elif filename.endswith('.xlsx'):
                    self.processed_data.to_excel(filename, index=False)

                self.log_message(f"结果已导出到: {filename}")
                messagebox.showinfo("成功", f"结果已导出到:\n{filename}")

        except Exception as e:
            self.log_message(f"导出失败: {str(e)}")
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)

    def log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        if hasattr(self, 'log_display'):
            self.log_display.config(state=tk.NORMAL)
            self.log_display.insert(tk.END, formatted_message + '\n')
            self.log_display.see(tk.END)
            self.log_display.config(state=tk.DISABLED)

        if hasattr(self, 'logger'):
            self.logger.info(message)

        self.log_queue.put(formatted_message)

    def process_log_queue(self):
        """处理日志队列"""
        try:
            while not self.log_queue.empty():
                self.log_queue.get_nowait()
        except:
            pass

        self.root.after(100, self.process_log_queue)

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        self.log_message("仪表盘已关闭")

    def run(self):
        """运行仪表盘"""
        if self.standalone:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()

    def on_closing(self):
        """关闭事件处理"""
        self.cleanup()
        if self.standalone:
            self.root.destroy()


def create_dashboard(master=None):
    """创建并返回仪表盘实例"""
    return DataDashboard(master)


def start_dashboard_standalone():
    """独立启动仪表盘"""
    dashboard = DataDashboard()
    dashboard.run()

if __name__ == "__main__":
    start_dashboard_standalone()
