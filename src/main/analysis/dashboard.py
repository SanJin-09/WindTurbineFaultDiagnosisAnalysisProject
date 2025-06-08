"""
Author: <WU Xinyan & SUN Runze>
风力涡轮机故障诊断交互式数据仪表盘
"""
import queue
import shutil
import tempfile
import threading
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import ttk, scrolledtext

import numpy as np

# 导入时添加异常处理
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，部分图表功能可能受限")

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib导入失败")

try:
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.main.analysis.config import *
from src.main.analysis.data_preprocessing import load_data, preprocess_data
from src.main.analysis.feature_engineering import extract_features, prepare_features, select_features
from src.main.analysis.logging_utils import get_logger
from src.main.analysis.model_training import split_data, train_models


class DataDashboard:
    """风力涡轮机故障诊断数据仪表盘 - 修复版本"""

    def __init__(self, root=None):
        """初始化仪表盘"""
        print("开始初始化仪表盘...")

        if root is None:
            self.root = tk.Tk()
            self.standalone = True
        else:
            self.root = root
            self.standalone = False

        # 添加异常处理
        try:
            self.setup_window()
            self.init_variables()
            self.setup_logger()

            # 分步创建UI，避免一次性创建过多组件
            self.create_ui_step_by_step()

            print("仪表盘初始化完成")

        except Exception as e:
            print(f"仪表盘初始化失败: {e}")
            traceback.print_exc()
            self.show_init_error(str(e))

    def show_init_error(self, error_msg):
        """显示初始化错误"""
        error_label = tk.Label(
            self.root,
            text=f"初始化失败:\n{error_msg}\n\n请检查依赖是否正确安装",
            font=('Arial', 12),
            fg='red',
            justify=tk.LEFT
        )
        error_label.pack(expand=True, padx=20, pady=20)

    def setup_window(self):
        """设置窗口属性"""
        self.root.title("风力涡轮机故障诊断系统 - 数据仪表盘")
        self.root.geometry("1400x900")  # 减小初始窗口大小
        self.root.minsize(1000, 600)

        # 设置窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def init_variables(self):
        """初始化变量"""
        print("初始化变量...")

        self.logger = None
        self.log_queue = queue.Queue()
        self.ui_update_queue = queue.Queue()

        # 数据相关
        self.raw_data = None
        self.processed_data = None
        self.features_data = None
        self.model_results = None
        self.analysis_complete = False
        self.data_loading = False
        self.analysis_running = False

        # UI状态
        self.current_plot = None
        self.plots_available = []
        self.ui_created = False

        # 临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

        print("变量初始化完成")

    def setup_logger(self):
        """设置日志系统"""
        try:
            self.logger = get_logger('dashboard')
            self.logger.info("仪表盘已启动")
        except Exception as e:
            print(f"日志系统初始化失败: {e}")
            # 创建一个简单的日志替代
            self.logger = None

    def create_ui_step_by_step(self):
        """分步创建UI"""
        print("开始创建UI...")

        # 第1步：创建基础框架
        self.create_basic_framework()
        self.root.update_idletasks()

        # 第2步：创建头部
        self.create_header()
        self.root.update_idletasks()

        # 第3步：创建主内容区域
        self.create_main_content()
        self.root.update_idletasks()

        # 第4步：创建状态栏
        self.create_status_bar()
        self.root.update_idletasks()

        # 第5步：启动后台任务
        self.start_background_tasks()

        self.ui_created = True
        print("UI创建完成")

    def create_basic_framework(self):
        """创建基础框架"""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_header(self):
        """创建头部区域"""
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # 标题
        title_label = ttk.Label(
            header_frame,
            text="风力涡轮机故障诊断系统",
            font=('Arial', 18, 'bold')  # 使用更通用的字体
        )
        title_label.pack(pady=10)

        # 控制按钮
        self.create_control_buttons(header_frame)

    def create_control_buttons(self, parent):
        """创建控制按钮"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)

        # 按钮
        buttons_info = [
            ("加载数据", self.safe_load_data),
            ("开始分析", self.safe_start_analysis),
            ("刷新图表", self.refresh_plots),
            ("导出结果", self.export_results),
            ("清理重置", self.reset_dashboard)
        ]

        for text, command in buttons_info:
            btn = ttk.Button(control_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)

        # 进度条
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(side=tk.RIGHT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.TOP, padx=5)

        self.progress_label = ttk.Label(progress_frame, text="就绪")
        self.progress_label.pack(side=tk.TOP, padx=5)

    def create_main_content(self):
        """创建主要内容区域"""
        # 使用简单的Frame而不是PanedWindow（减少复杂性）
        content_frame = ttk.Frame(self.main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 创建Notebook
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建各个选项卡
        self.create_data_tab()
        self.create_visualization_tab()
        self.create_log_tab()

    def create_data_tab(self):
        """创建数据选项卡"""
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="数据概览")

        # 数据信息显示区域
        info_frame = ttk.LabelFrame(self.data_tab, text="数据信息")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.data_info = scrolledtext.ScrolledText(
            info_frame,
            height=15,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.data_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 初始显示
        self.safe_update_data_info("等待数据加载...")

        # 参数显示
        self.create_parameter_display()

    def create_parameter_display(self):
        """创建参数显示区域"""
        param_frame = ttk.LabelFrame(self.data_tab, text="当前参数设置")
        param_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        param_text = f"""
分析参数配置:
• 故障前时间窗口: {WINDOW_BEFORE} 小时
• 故障后时间窗口: {WINDOW_AFTER} 小时  
• 测试集比例: {TEST_SIZE}
• 随机森林估算器: {RF_N_ESTIMATORS}
• 梯度提升估算器: {GB_N_ESTIMATORS}
• 特征选择阈值: {IMPORTANCE_THRESHOLD}
"""

        param_label = ttk.Label(param_frame, text=param_text, justify=tk.LEFT)
        param_label.pack(anchor=tk.W, padx=10, pady=5)

    def create_visualization_tab(self):
        """创建可视化选项卡"""
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="数据可视化")

        if not HAS_MATPLOTLIB:
            error_label = ttk.Label(
                self.viz_tab,
                text="matplotlib未正确安装，可视化功能不可用",
                font=('Arial', 12)
            )
            error_label.pack(expand=True)
            return

        # 图表控制区域
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

        ttk.Button(
            control_frame,
            text="显示图表",
            command=self.safe_show_plot
        ).pack(side=tk.RIGHT, padx=5)

        # 图表显示区域
        self.create_plot_area()

    def create_plot_area(self):
        """创建图表显示区域"""
        try:
            plot_frame = ttk.Frame(self.viz_tab)
            plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # matplotlib图形
            self.fig = Figure(figsize=(10, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 工具栏
            toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
            toolbar.update()

            # 显示初始图表
            self.show_welcome_plot()

        except Exception as e:
            print(f"图表区域创建失败: {e}")
            error_label = ttk.Label(plot_frame, text=f"图表功能初始化失败: {e}")
            error_label.pack(expand=True)

    def show_welcome_plot(self):
        """显示欢迎图表"""
        try:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                   '欢迎使用风力涡轮机故障诊断系统\n\n请先加载数据并完成分析\n然后选择要显示的图表',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
        except Exception as e:
            print(f"欢迎图表显示失败: {e}")

    def create_log_tab(self):
        """创建日志选项卡"""
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="系统日志")

        log_frame = ttk.LabelFrame(self.log_tab, text="实时日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_display = scrolledtext.ScrolledText(
            log_frame,
            height=25,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加清除日志按钮
        button_frame = ttk.Frame(log_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Button(button_frame, text="清除日志", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="保存日志", command=self.save_log).pack(side=tk.LEFT, padx=5)

    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ttk.Frame(self.main_container)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_var = tk.StringVar(value="系统就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)

        # 时间显示
        self.time_var = tk.StringVar()
        time_label = ttk.Label(status_frame, textvariable=self.time_var)
        time_label.pack(side=tk.RIGHT, padx=5)

    def start_background_tasks(self):
        """启动后台任务"""
        # 启动时间更新
        self.update_time()

        # 启动日志处理
        self.process_log_queue()

        # 启动UI更新队列处理
        self.process_ui_update_queue()

        # 延迟启动自动数据加载
        self.root.after(2000, self.auto_load_data)

    def update_time(self):
        """更新时间显示"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_var.set(current_time)
        except Exception as e:
            print(f"时间更新失败: {e}")

        # 继续更新
        self.root.after(1000, self.update_time)

    def process_log_queue(self):
        """处理日志队列"""
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                self.display_log_message(message)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"日志处理错误: {e}")

        # 继续处理
        self.root.after(200, self.process_log_queue)

    def process_ui_update_queue(self):
        """处理UI更新队列"""
        try:
            while not self.ui_update_queue.empty():
                update_func = self.ui_update_queue.get_nowait()
                if callable(update_func):
                    update_func()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"UI更新错误: {e}")

        # 继续处理
        self.root.after(100, self.process_ui_update_queue)

    def auto_load_data(self):
        """自动加载数据"""
        if not self.ui_created:
            return

        if os.path.exists(SCADA_DATA_PATH) and os.path.exists(FAULT_DATA_PATH):
            self.safe_log_message("发现数据文件，准备自动加载...")
            self.root.after(1000, self.safe_load_data)
        else:
            self.safe_log_message("未找到数据文件，请手动加载数据")

    def safe_load_data(self):
        """安全的数据加载"""
        if self.data_loading:
            self.safe_log_message("数据正在加载中，请稍候...")
            return

        if not self.ui_created:
            self.safe_log_message("UI未就绪，无法加载数据")
            return

        self.data_loading = True
        self.safe_update_status("正在加载数据...")
        self.progress_var.set(10)

        def load_worker():
            try:
                self.safe_log_message("开始加载数据...")
                scada_data, fault_data = load_data(self.logger)

                # 将结果放入队列，而不是直接更新UI
                result_data = {
                    'scada': scada_data,
                    'fault': fault_data
                }

                # 使用队列进行线程安全的UI更新
                self.ui_update_queue.put(
                    lambda: self.on_data_loaded_safe(True, "数据加载成功", result_data)
                )

            except Exception as e:
                error_msg = f"数据加载失败: {str(e)}"
                print(f"数据加载错误: {e}")
                traceback.print_exc()

                self.ui_update_queue.put(
                    lambda: self.on_data_loaded_safe(False, error_msg, None)
                )

        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()

    def on_data_loaded_safe(self, success, message, data):
        """安全的数据加载完成回调"""
        try:
            self.data_loading = False

            if success and data:
                self.raw_data = data
                self.progress_var.set(30)
                self.safe_update_status("数据加载完成")
                self.safe_log_message(message)

                # 更新数据概览
                scada_info = f"SCADA数据: {self.raw_data['scada'].shape[0]}行 x {self.raw_data['scada'].shape[1]}列"
                fault_info = f"故障数据: {self.raw_data['fault'].shape[0]}行 x {self.raw_data['fault'].shape[1]}列"

                # 数据预览（只显示前几行）
                scada_preview = self.raw_data['scada'].head().to_string()

                overview_text = f"""数据加载成功！

{scada_info}
{fault_info}

加载时间: {datetime.now().strftime('%H:%M:%S')}

SCADA数据预览:
{scada_preview}

数据文件路径:
• SCADA: {SCADA_DATA_PATH}
• 故障: {FAULT_DATA_PATH}
"""
                self.safe_update_data_info(overview_text)

            else:
                self.progress_var.set(0)
                self.safe_update_status("数据加载失败")
                self.safe_log_message(message)

                # 不要在这里显示错误对话框，可能会导致线程问题
                self.safe_update_data_info(f"数据加载失败:\n{message}")

        except Exception as e:
            print(f"数据加载回调错误: {e}")
            traceback.print_exc()

    def safe_start_analysis(self):
        """安全的开始分析"""
        if self.analysis_running:
            self.safe_log_message("分析正在进行中，请稍候...")
            return

        if self.raw_data is None:
            self.safe_log_message("请先加载数据")
            return

        self.analysis_running = True
        self.safe_update_status("正在进行数据分析...")
        self.progress_var.set(40)

        def analysis_worker():
            try:
                # 数据预处理
                self.safe_log_message("开始数据预处理...")
                labeled_data, fault_distribution = preprocess_data(
                    self.raw_data['scada'],
                    self.raw_data['fault'],
                    self.logger
                )

                self.ui_update_queue.put(lambda: self.progress_var.set(50))

                # 特征工程
                self.safe_log_message("开始特征工程...")
                features = extract_features(labeled_data)
                x, y, le = prepare_features(features)

                self.ui_update_queue.put(lambda: self.progress_var.set(60))

                # 数据分割
                self.safe_log_message("分割数据集...")
                x_train, x_test, y_train, y_test = split_data(x, y)

                # 特征选择
                x_train_selected, x_test_selected, scaler, selector, important_features = select_features(
                    x_train, y_train, x_test
                )

                self.ui_update_queue.put(lambda: self.progress_var.set(80))

                # 模型训练
                self.safe_log_message("开始模型训练...")
                best_model, best_model_name, best_accuracy, accuracies, y_test, y_pred = train_models(
                    x_train_selected, y_train, x_test_selected, y_test
                )

                self.ui_update_queue.put(lambda: self.progress_var.set(90))

                # 保存结果
                results = {
                    'processed_data': labeled_data,
                    'features_data': features,
                    'model_results': {
                        'best_model': best_model,
                        'best_model_name': best_model_name,
                        'best_accuracy': best_accuracy,
                        'accuracies': accuracies,
                        'important_features': important_features,
                        'fault_distribution': fault_distribution,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                }

                self.ui_update_queue.put(
                    lambda: self.on_analysis_complete_safe(True, "分析完成", results)
                )

            except Exception as e:
                error_msg = f"分析失败: {str(e)}"
                print(f"分析错误: {e}")
                traceback.print_exc()

                self.ui_update_queue.put(
                    lambda: self.on_analysis_complete_safe(False, error_msg, None)
                )

        thread = threading.Thread(target=analysis_worker, daemon=True)
        thread.start()

    def on_analysis_complete_safe(self, success, message, results):
        """安全的分析完成回调"""
        try:
            self.analysis_running = False

            if success and results:
                self.processed_data = results['processed_data']
                self.features_data = results['features_data']
                self.model_results = results['model_results']
                self.analysis_complete = True

                self.progress_var.set(100)
                self.safe_update_status("分析完成")
                self.safe_log_message(message)

                # 更新结果显示
                self.update_results_info()

                # 自动显示第一个图表
                if HAS_MATPLOTLIB:
                    self.plot_combo.current(0)
                    self.safe_show_plot()

                self.safe_log_message("数据分析完成！可以查看可视化结果。")

            else:
                self.progress_var.set(0)
                self.safe_update_status("分析失败")
                self.safe_log_message(message)

        except Exception as e:
            print(f"分析完成回调错误: {e}")
            traceback.print_exc()

    def update_results_info(self):
        """更新结果信息显示"""
        if not self.model_results:
            return

        try:
            result_text = f"""分析结果摘要:

最佳模型: {self.model_results['best_model_name']}
准确率: {self.model_results['best_accuracy']:.4f}

所有模型准确率:
"""
            for model, acc in self.model_results['accuracies'].items():
                result_text += f"• {model}: {acc:.4f}\n"

            result_text += f"\n重要特征 (前10个):\n"
            for i, feat in enumerate(self.model_results['important_features'][:10], 1):
                result_text += f"{i:2d}. {feat}\n"

            result_text += f"\n故障分布:\n"
            for fault, count in self.model_results['fault_distribution'].items():
                result_text += f"• {fault}: {count}\n"

            # 更新数据信息显示
            current_info = self.data_info.get(1.0, tk.END)
            updated_info = current_info + "\n" + "="*50 + "\n" + result_text

            self.safe_update_data_info(updated_info)

        except Exception as e:
            print(f"结果信息更新失败: {e}")

    def safe_show_plot(self):
        """安全的显示图表"""
        if not self.analysis_complete:
            self.safe_log_message("请先完成数据分析")
            return

        if not HAS_MATPLOTLIB:
            self.safe_log_message("matplotlib不可用，无法显示图表")
            return

        plot_type = self.plot_var.get()
        if not plot_type:
            return

        try:
            # 清除当前图表
            self.fig.clear()

            if plot_type == "故障分布图":
                self.plot_fault_distribution()
            elif plot_type == "特征分布图":
                self.plot_feature_distribution()
            elif plot_type == "模型准确率对比":
                self.plot_model_accuracies()
            elif plot_type == "混淆矩阵" and HAS_SKLEARN:
                self.plot_confusion_matrix()
            elif plot_type == "特征重要性":
                self.plot_feature_importance()
            elif plot_type == "时间序列特征":
                self.plot_time_series()

            self.canvas.draw()
            self.safe_log_message(f"已显示图表: {plot_type}")

        except Exception as e:
            self.safe_log_message(f"图表显示错误: {str(e)}")
            # 显示错误信息
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"图表显示错误:\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()

    def plot_fault_distribution(self):
        """绘制故障分布图"""
        ax = self.fig.add_subplot(111)
        fault_dist = self.model_results['fault_distribution']

        bars = ax.bar(range(len(fault_dist)), fault_dist.values, color='skyblue', alpha=0.7)
        ax.set_xlabel('故障类型')
        ax.set_ylabel('样本数量')
        ax.set_title('故障分布')
        ax.set_xticks(range(len(fault_dist)))
        ax.set_xticklabels(fault_dist.index, rotation=45, ha='right')

        # 添加数值标签
        for bar, value in zip(bars, fault_dist.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   str(value), ha='center', va='bottom')

        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

    def plot_feature_distribution(self):
        """绘制特征分布图"""
        if self.processed_data is None:
            return

        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns[:6]

        for i, col in enumerate(numeric_cols):
            ax = self.fig.add_subplot(2, 3, i+1)
            data = self.processed_data[col].dropna()
            ax.hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black', linewidth=0.5)
            ax.set_title(f'{col}分布', fontsize=10)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel('频次', fontsize=8)
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def plot_model_accuracies(self):
        """绘制模型准确率对比"""
        ax = self.fig.add_subplot(111)
        accuracies = self.model_results['accuracies']

        models = list(accuracies.keys())
        values = list(accuracies.values())

        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'][:len(models)]
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_ylabel('准确率')
        ax.set_title('模型准确率对比')
        ax.set_ylim(0, 1.1)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        self.fig.tight_layout()

    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        if not HAS_SKLEARN:
            return

        ax = self.fig.add_subplot(111)

        y_test = self.model_results['y_test']
        y_pred = self.model_results['y_pred']

        cm = confusion_matrix(y_test, y_pred)

        # 手动绘制热力图（不依赖seaborn）
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # 添加文本注释
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('真实标签')
        ax.set_xlabel('预测标签')
        ax.set_title('混淆矩阵')
        self.fig.colorbar(im, ax=ax)
        self.fig.tight_layout()

    def plot_feature_importance(self):
        """绘制特征重要性"""
        ax = self.fig.add_subplot(111)

        model = self.model_results['best_model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = self.model_results['important_features'][:10]

            # 取对应的重要性值
            importance_values = importances[:len(features)]

            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importance_values, color='lightcoral', alpha=0.7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('重要性')
            ax.set_title('特征重要性排序')
            ax.grid(True, alpha=0.3, axis='x')

            # 添加数值标签
            for bar, value in zip(bars, importance_values):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                       f'{value:.3f}', ha='left', va='center', fontsize=8)
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

        # 取前1000个数据点
        data_sample = self.processed_data.head(1000)
        numeric_cols = data_sample.select_dtypes(include=[np.number]).columns[:3]

        colors = ['blue', 'red', 'green']
        for i, col in enumerate(numeric_cols):
            ax.plot(data_sample['DateTime'], data_sample[col],
                   label=col, alpha=0.7, color=colors[i % len(colors)])

        ax.set_xlabel('时间')
        ax.set_ylabel('特征值')
        ax.set_title('时间序列特征变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        self.fig.tight_layout()

    def refresh_plots(self):
        """刷新图表"""
        if self.analysis_complete and HAS_MATPLOTLIB:
            self.safe_show_plot()
            self.safe_log_message("图表已刷新")
        else:
            self.safe_log_message("请先完成数据分析或检查matplotlib安装")

    def export_results(self):
        """导出结果"""
        if not self.analysis_complete:
            self.safe_log_message("请先完成数据分析")
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

                self.safe_log_message(f"结果已导出到: {filename}")

        except Exception as e:
            self.safe_log_message(f"导出失败: {str(e)}")

    def reset_dashboard(self):
        """重置仪表盘"""
        try:
            # 重置数据
            self.raw_data = None
            self.processed_data = None
            self.features_data = None
            self.model_results = None
            self.analysis_complete = False
            self.data_loading = False
            self.analysis_running = False

            # 重置进度
            self.progress_var.set(0)
            self.safe_update_status("系统已重置")

            # 清空显示
            self.safe_update_data_info("系统已重置，等待重新加载数据...")

            if HAS_MATPLOTLIB:
                self.show_welcome_plot()

            self.safe_log_message("系统已重置")

        except Exception as e:
            self.safe_log_message(f"重置失败: {str(e)}")

    def clear_log(self):
        """清除日志"""
        try:
            self.log_display.config(state=tk.NORMAL)
            self.log_display.delete(1.0, tk.END)
            self.log_display.config(state=tk.DISABLED)
        except Exception as e:
            print(f"清除日志失败: {e}")

    def save_log(self):
        """保存日志"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )

            if filename:
                log_content = self.log_display.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                self.safe_log_message(f"日志已保存到: {filename}")

        except Exception as e:
            self.safe_log_message(f"保存日志失败: {str(e)}")

    # 安全的UI更新方法
    def safe_update_status(self, message):
        """安全更新状态栏"""
        try:
            self.status_var.set(message)
        except Exception as e:
            print(f"状态更新失败: {e}")

    def safe_log_message(self, message):
        """安全记录日志消息"""
        try:
            self.log_queue.put(message)
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(message)
        except Exception as e:
            print(f"日志记录失败: {e}")

    def display_log_message(self, message):
        """显示日志消息"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"

            self.log_display.config(state=tk.NORMAL)
            self.log_display.insert(tk.END, formatted_message + '\n')
            self.log_display.see(tk.END)
            self.log_display.config(state=tk.DISABLED)
        except Exception as e:
            print(f"日志显示失败: {e}")

    def safe_update_data_info(self, text):
        """安全更新数据信息"""
        try:
            self.data_info.config(state=tk.NORMAL)
            self.data_info.delete(1.0, tk.END)
            self.data_info.insert(tk.END, text)
            self.data_info.config(state=tk.DISABLED)
        except Exception as e:
            print(f"数据信息更新失败: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.safe_log_message("资源清理完成")
        except Exception as e:
            print(f"清理失败: {e}")

    def run(self):
        """运行仪表盘"""
        if self.standalone:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            print("启动仪表盘主循环...")
            try:
                self.root.mainloop()
            except Exception as e:
                print(f"主循环错误: {e}")
                traceback.print_exc()

    def on_closing(self):
        """关闭事件处理"""
        try:
            self.cleanup()
            if self.standalone:
                self.root.destroy()
        except Exception as e:
            print(f"关闭处理错误: {e}")


# 兼容性函数
def create_dashboard(master=None):
    """创建并返回仪表盘实例"""
    return DataDashboard(master)


def start_dashboard_standalone():
    """独立启动仪表盘"""
    print("启动风力涡轮机故障诊断系统仪表盘...")
    try:
        dashboard = DataDashboard()
        dashboard.run()
    except Exception as e:
        print(f"仪表盘启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    start_dashboard_standalone()
