import os.path as path
import cv2
import sys
import model as md  # 必须先于PyQt5导入
import numpy as np
from io import StringIO
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QListWidget,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QListWidgetItem,
    QSizePolicy,
)
from matplotlib import use


# ====================== 标准输出重定向 ======================
class StdoutRedirector(StringIO):
    """重定向sys.stdout，将输出通过信号发送到UI"""

    def __init__(self, log_signal):
        super().__init__()
        self.log_signal = log_signal

    def write(self, text):
        super().write(text)
        if text.strip():  # 过滤空行
            self.log_signal.emit(text.strip())

    def flush(self):
        pass


# ====================== 训练线程 ======================
class TrainThread(QThread):
    """模型训练子线程（避免UI卡死）"""

    log_signal = pyqtSignal(str)  # 训练日志信号
    finish_signal = pyqtSignal(md.model)  # 训练完成信号

    def __init__(self, data_path, epochs=100, batch=16):
        super().__init__()
        self.data_path = data_path
        self.epochs = epochs
        self.batch = batch

    def run(self):
        # 重定向标准输出到UI日志框
        original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.log_signal)

        try:
            # 执行模型训练
            self.log_signal.emit("开始训练模型...")
            detect_model = md.model.train(
                data=self.data_path, epochs=self.epochs, batch=self.batch
            )
            self.log_signal.emit("模型训练完成！")
            self.finish_signal.emit(detect_model)
        except Exception as e:
            self.log_signal.emit(f"训练出错: {str(e)}")
        finally:
            # 恢复标准输出
            sys.stdout = original_stdout


# ====================== 检测线程 ======================
class DetectThread(QThread):
    """检测子线程（视频/摄像头）"""

    frame_signal = pyqtSignal(np.ndarray)  # 检测帧信号
    car_info_signal = pyqtSignal(dict)  # 车辆信息信号
    finish_signal = pyqtSignal()  # 检测完成信号

    def __init__(
        self, detect_model, source, time=5, rate=3e-3, conf=0.25, iou=0.7, save=False
    ):
        super().__init__()
        self.detect_model = detect_model
        self.source = source  # 摄像头编号(int)或视频路径(str)
        self.time = time
        self.rate = rate
        self.conf = conf
        self.iou = iou
        self.save = save
        self.is_running = True  # 检测运行标志

    def stop(self):
        """停止检测"""
        self.is_running = False

    def custom_show(self, img):
        """自定义show函数：发送帧数据到UI，返回是否继续检测"""
        return self.frame_signal.emit(img) if self.is_running else 1

    def run(self):
        """执行检测逻辑"""
        try:
            # 处理检测源：数字字符串转摄像头编号，否则为视频路径
            source = int(self.source) if self.source.strip().isdigit() else self.source
            # 调用模型的track方法
            self.detect_model.track(
                source=source,
                time=self.time,
                rate=self.rate,
                show=self.custom_show,
                recv=self.car_info_signal.emit,
                conf=self.conf,
                iou=self.iou,
                save=self.save,
            )
        except Exception as e:
            self.car_info_signal.emit({"error": f"检测出错: {str(e)}"})
        finally:
            self.finish_signal.emit()


# ====================== 主窗口 ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("违停检测与车牌识别系统")
        self.resize(1200, 800)
        self.detect_model = None  # 当前加载的检测模型
        self.current_frame = None  # 当前检测帧
        self.init_ui()  # 初始化UI

    def init_ui(self):
        """初始化界面布局"""
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 水平分割器（左侧控制区，右侧显示区）
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------------- 左侧控制区 ----------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # 1. 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)
        self.model_path_edit = QLineEdit(placeholderText="模型文件路径(yaml/pt)")
        self.model_path_btn = QPushButton("选择模型文件")
        self.model_path_btn.clicked.connect(self.select_model_file)
        self.set_model_btn = QPushButton("加载模型")
        self.set_model_btn.clicked.connect(self.set_model)
        model_layout.addRow("模型文件：", self.model_path_edit)
        model_layout.addRow("", self.model_path_btn)
        model_layout.addRow("", self.set_model_btn)
        left_layout.addWidget(model_group)

        # 2. 模型训练组
        train_group = QGroupBox("模型训练")
        train_layout = QFormLayout(train_group)
        self.train_data_edit = QLineEdit(placeholderText="数据集yaml文件路径")
        self.train_data_btn = QPushButton("选择数据集文件")
        self.train_data_btn.clicked.connect(self.select_train_data)
        self.epochs_edit = QLineEdit("100", placeholderText="训练轮数")
        self.batch_edit = QLineEdit("16", placeholderText="批大小")
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_train)
        train_layout.addRow("数据集文件：", self.train_data_edit)
        train_layout.addRow("", self.train_data_btn)
        train_layout.addRow("训练轮数：", self.epochs_edit)
        train_layout.addRow("批大小：", self.batch_edit)
        train_layout.addRow("", self.train_btn)
        left_layout.addWidget(train_group)

        # 3. 检测控制组
        detect_group = QGroupBox("检测控制")
        detect_layout = QFormLayout(detect_group)
        self.detect_source_edit = QLineEdit("0", placeholderText="摄像头编号/视频路径")
        self.detect_source_btn = QPushButton("选择视频文件")
        self.detect_source_btn.clicked.connect(self.select_detect_source)
        self.detect_time_edit = QLineEdit("5", placeholderText="停留秒数阈值")
        self.detect_rate_edit = QLineEdit("0.003", placeholderText="位移归一化阈值")
        self.detect_conf_edit = QLineEdit("0.25", placeholderText="置信度阈值")
        self.detect_iou_edit = QLineEdit("0.7", placeholderText="交并比阈值")
        self.detect_save_btn = QPushButton("保存检测视频")
        self.detect_save_btn.setCheckable(True)
        self.start_detect_btn = QPushButton("开始检测")
        self.start_detect_btn.clicked.connect(self.start_detect)
        self.stop_detect_btn = QPushButton("停止检测")
        self.stop_detect_btn.clicked.connect(self.stop_detect)
        self.stop_detect_btn.setEnabled(False)
        detect_layout.addRow("检测源：", self.detect_source_edit)
        detect_layout.addRow("", self.detect_source_btn)
        detect_layout.addRow("停留秒数：", self.detect_time_edit)
        detect_layout.addRow("位移阈值：", self.detect_rate_edit)
        detect_layout.addRow("置信度：", self.detect_conf_edit)
        detect_layout.addRow("交并比：", self.detect_iou_edit)
        detect_layout.addRow("保存视频：", self.detect_save_btn)
        detect_layout.addRow("", self.start_detect_btn)
        detect_layout.addRow("", self.stop_detect_btn)
        left_layout.addWidget(detect_group)

        # 训练日志显示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("训练日志将显示在这里...")
        left_layout.addWidget(QLabel("训练日志："))
        left_layout.addWidget(self.log_text)

        # ---------------------- 右侧显示区 ----------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)

        # 检测帧显示标签（核心修改：添加最大尺寸+固定大小策略）
        self.frame_label = QLabel("检测画面")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("border: 1px solid #cccccc;")
        # ========== 设置最大尺寸（可根据需求调整，比如800x600） ==========
        self.frame_label.setMaximumSize(800, 600)
        # 可选：设置最小尺寸，避免控件过小
        self.frame_label.setMinimumSize(400, 300)
        # 设置大小策略：固定尺寸，不随父控件拉伸无限放大
        self.frame_label.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed  # 水平方向固定  # 垂直方向固定
        )
        right_layout.addWidget(self.frame_label)

        # 车辆信息列表
        self.car_info_list = QListWidget()
        self.car_info_list.setStyleSheet(
            """
            QListWidget { font-size: 12px; }
            QListWidget::item[role="illegal"] { color: red; font-weight: bold; }
            QListWidget::item[role="warning"] { color: orange; }
            QListWidget::item[role="normal"] { color: green; }
        """
        )
        right_layout.addWidget(
            QLabel("车辆状态（红色：违停 | 黄色：停靠中 | 绿色：正常）")
        )
        right_layout.addWidget(self.car_info_list)

        # 设置分割器初始比例
        splitter.setSizes([400, 800])

        # 帧更新定时器（30fps）
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame_display)
        self.frame_timer.setInterval(33)

    # ====================== 模型设置相关 ======================
    def select_model_file(self):
        """选择模型文件（yaml/pt）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.yaml *.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def set_model(self):
        """加载指定的模型文件"""
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.log_text.append("❌ 错误：请先选择模型文件！")
            return
        if not path.exists(model_path):
            self.log_text.append(f"❌ 错误：模型文件 {model_path} 不存在！")
            return

        try:
            # 初始化模型
            model_name = path.splitext(path.basename(model_path))[0]
            self.detect_model = md.model(model_name, model_path)
            self.log_text.append(f"✅ 成功加载模型：{model_path}")
        except Exception as e:
            self.log_text.append(f"❌ 加载模型失败：{str(e)}")

    # ====================== 模型训练相关 ======================
    def select_train_data(self):
        """选择训练数据集yaml文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集文件", "", "YAML文件 (*.yaml);;所有文件 (*.*)"
        )
        if file_path:
            self.train_data_edit.setText(file_path)

    def start_train(self):
        """启动模型训练线程"""
        data_path = self.train_data_edit.text().strip()
        if not data_path:
            self.log_text.append("❌ 错误：请先选择数据集文件！")
            return
        if not path.exists(data_path):
            self.log_text.append(f"❌ 错误：数据集文件 {data_path} 不存在！")
            return

        # 解析训练参数
        try:
            epochs = int(self.epochs_edit.text().strip())
            batch = int(self.batch_edit.text().strip())
            if epochs <= 0 or batch <= 0:
                raise ValueError
        except ValueError:
            self.log_text.append("❌ 错误：训练轮数/批大小必须为正整数！")
            return

        # 清空日志并启动训练线程
        self.log_text.clear()
        self.train_thread = TrainThread(data_path, epochs, batch)
        self.train_thread.log_signal.connect(self.append_log)
        self.train_thread.finish_signal.connect(self.on_train_finish)
        self.train_thread.start()
        self.train_btn.setEnabled(False)

    def append_log(self, text):
        """追加训练日志到文本框"""
        self.log_text.append(text)
        # 自动滚动到最后一行
        self.log_text.moveCursor(self.log_text.textCursor().End)

    def on_train_finish(self, detect_model):
        """训练完成回调"""
        self.detect_model = detect_model
        self.log_text.append("✅ 训练完成，模型已自动加载！")
        self.train_btn.setEnabled(True)

    # ====================== 检测相关 ======================
    def select_detect_source(self):
        """选择检测视频文件（摄像头直接输入编号）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)",
        )
        if file_path:
            self.detect_source_edit.setText(file_path)

    def start_detect(self):
        """启动检测线程"""
        if self.detect_model is None:
            self.log_text.append("❌ 错误：请先加载/训练模型！")
            return

        # 解析检测参数
        try:
            source = self.detect_source_edit.text().strip()
            time = float(self.detect_time_edit.text().strip())
            rate = float(self.detect_rate_edit.text().strip())
            conf = float(self.detect_conf_edit.text().strip())
            iou = float(self.detect_iou_edit.text().strip())
            save = self.detect_save_btn.isChecked()
        except ValueError:
            self.log_text.append("❌ 错误：检测参数格式错误（请输入合法数字）！")
            return

        # 启动检测线程
        self.detect_thread = DetectThread(
            self.detect_model, source, time, rate, conf, iou, save
        )
        self.detect_thread.frame_signal.connect(self.update_frame_data)
        self.detect_thread.car_info_signal.connect(self.update_car_info)
        self.detect_thread.finish_signal.connect(self.on_detect_finish)
        self.detect_thread.start()

        # 更新按钮状态
        self.start_detect_btn.setEnabled(False)
        self.stop_detect_btn.setEnabled(True)
        self.frame_timer.start()  # 启动帧更新定时器

    def stop_detect(self):
        """停止检测"""
        if hasattr(self, "detect_thread") and self.detect_thread.isRunning():
            self.detect_thread.stop()
            self.stop_detect_btn.setEnabled(False)
            self.log_text.append("⏹️ 正在停止检测...")

    def update_frame_data(self, frame):
        """更新当前检测帧数据"""
        self.current_frame = frame

    def update_frame_display(self):
        """将CV2帧转换为QPixmap并显示（限制最大尺寸）"""
        if self.current_frame is None:
            return

        # 转换BGR（CV2）到RGB（Qt）
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # 创建QImage
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # ========== 核心修改：基于QLabel的最大尺寸缩放图片 ==========
        # 获取QLabel的最大尺寸（800x600）
        max_size = self.frame_label.maximumSize()
        # 缩放图片：保持宽高比，且不超过最大尺寸
        pixmap = QPixmap.fromImage(q_image).scaled(
            max_size,  # 限制在最大尺寸内
            Qt.KeepAspectRatio,  # 保持宽高比
            Qt.SmoothTransformation,  # 平滑缩放
        )
        # 设置图片到QLabel（QLabel本身已限制最大尺寸，不会放大）
        self.frame_label.setPixmap(pixmap)

    def update_car_info(self, cars):
        """更新车辆信息列表（最终修复版，直接设置颜色）"""
        # 处理错误信息
        if "error" in cars:
            self.log_text.append(cars["error"])
            return

        # 清空列表并重新添加车辆信息
        self.car_info_list.clear()
        # 从UI输入框获取time_threshold
        time_threshold = float(self.detect_time_edit.text().strip())
        source = self.detect_source_edit.text().strip()
        fps = md.getinfo(int(source) if source.strip().isdigit() else source)[0]

        for car_id, car_info in cars.items():
            park_count = car_info.get("park", 0)
            plate = car_info.get("plate") or "unknown"
            car_type = car_info.get("cls", "未知类型")
            # 判断车辆状态
            if car_info["illegal"]:
                status = "illegal"
                status_text = "违停"
            elif park_count > 0:
                status = "warning"
                status_text = "停靠中"
            else:
                status = "normal"
                status_text = "正常"
            # 创建列表项
            item_text = f"ID:{car_id} | 车牌:{plate} | 类型:{car_type} | 停留秒数:{park_count/fps:.2f}/{time_threshold} | 状态:{status_text}"
            item = QListWidgetItem(item_text)
            if status == "illegal":
                # 违停：红色
                item.setForeground(QColor(255, 0, 0))  # RGB红色
                """
                item.setFont(
                    QFont(item.font().family(), item.font().pointSize(), QFont.Bold)
                )  # 加粗
                """
            elif status == "warning":
                # 停靠中：橙色
                item.setForeground(QColor(255, 165, 0))
            else:
                # 正常：绿色
                item.setForeground(QColor(0, 128, 0))

            self.car_info_list.addItem(item)

        # 刷新列表显示
        self.car_info_list.viewport().update()

    def on_detect_finish(self):
        """检测完成回调"""
        self.start_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(False)
        self.frame_timer.stop()
        self.current_frame = None
        self.frame_label.setText("检测完成")
        self.log_text.append("✅ 检测结束！")


if __name__ == "__main__":
    use("Agg")
    app = QApplication(sys.argv)
    app.setFont(app.font("SimHei"))  # 设置默认中文字体
    window = MainWindow()
    window.show()
    app.exec_()
