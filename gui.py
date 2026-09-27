import sys
import os
import subprocess
import re
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QLineEdit, QFileDialog, QGroupBox, 
                               QGridLayout, QTextEdit, QCheckBox, QSplitter)
from PySide6.QtCore import Qt

class KiguApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kigurumi 尺寸自动推算与验证系统")
        self.resize(1100, 700)
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_image = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # ================= 左侧面板 =================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        lbl_left = QLabel("第一步：立绘 AI 分析")
        lbl_left.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(lbl_left)
        
        self.btn_select = QPushButton("选择立绘图片...")
        self.btn_select.setMinimumHeight(40)
        self.btn_select.clicked.connect(self.select_image)
        left_layout.addWidget(self.btn_select)
        
        self.lbl_path = QLabel("未选择图片")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("color: gray;")
        left_layout.addWidget(self.lbl_path)
        
        group_override = QGroupBox("手动修正 (长发遮盖下巴时必填)")
        override_layout = QVBoxLayout(group_override)
        self.chin_y_entry = QLineEdit()
        self.chin_y_entry.setPlaceholderText("下巴 Y 坐标 (如 66)")
        override_layout.addWidget(self.chin_y_entry)
        
        self.chk_rembg = QCheckBox("使用 rembg AI 去背")
        self.chk_rembg.setChecked(True)
        override_layout.addWidget(self.chk_rembg)
        left_layout.addWidget(group_override)
        
        self.btn_analyze = QPushButton("开始分析立绘")
        self.btn_analyze.setMinimumHeight(40)
        self.btn_analyze.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_analyze.clicked.connect(self.run_analysis)
        left_layout.addWidget(self.btn_analyze)
        
        self.analysis_out = QTextEdit()
        self.analysis_out.setReadOnly(True)
        self.analysis_out.setStyleSheet("font-family: monospace; background-color: #f5f5f5; color: #333;")
        left_layout.addWidget(self.analysis_out)
        
        # ================= 右侧面板 =================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        lbl_right = QLabel("第二步：头壳物理测算")
        lbl_right.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(lbl_right)
        
        group_inputs = QGroupBox("物理参数设置")
        grid_layout = QGridLayout(group_inputs)
        
        self.inputs = {}
        fields = [
            ("height", "演员身高 (cm)", "190.0"),
            ("weight", "演员体重 (kg)", "84.8"),
            ("shoulder", "演员肩宽 (cm)", "45.0"),
            ("head_w", "演员头宽 (cm)", "18.0"),
            ("head_h", "演员头高 (cm)", "25.0"),
            ("ratio", "角色头身比 (N)", "6.11"),
            ("head_shoulder", "角色头肩比 (r)", "0.70"),
            ("wig", "假发厚度 (cm)", "2.5"),
        ]
        
        for i, (key, label, default) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            grid_layout.addWidget(QLabel(label), row, col)
            entry = QLineEdit(default)
            grid_layout.addWidget(entry, row, col + 1)
            self.inputs[key] = entry
            
        right_layout.addWidget(group_inputs)
        
        self.btn_calc = QPushButton("计算 3D 建模尺寸")
        self.btn_calc.setMinimumHeight(40)
        self.btn_calc.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_calc.clicked.connect(self.run_calc)
        right_layout.addWidget(self.btn_calc)
        
        self.calc_out = QTextEdit()
        self.calc_out.setReadOnly(True)
        self.calc_out.setStyleSheet("font-family: monospace; background-color: #f5f5f5; color: #333;")
        right_layout.addWidget(self.calc_out)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        # 设置左右初始比例 1:1
        splitter.setSizes([550, 550])
        
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.lbl_path.setText(file_path)
            self.current_image = file_path
            
    def run_analysis(self):
        if not self.current_image:
            self.analysis_out.append("⚠️ 请先在上方选择一张图片！")
            return
            
        self.analysis_out.clear()
        self.analysis_out.append("🤖 正在进行 AI 图像分析，请稍候...")
        QApplication.processEvents()
        
        cmd = [sys.executable, os.path.join(self.script_dir, "char_ratio_from_image.py"), self.current_image]
        
        chin_y = self.chin_y_entry.text().strip()
        if chin_y:
            cmd.extend(["--chin-y", chin_y])
            
        if not self.chk_rembg.isChecked():
            cmd.append("--no-rembg")
            
        env = dict(os.environ, PYTHONIOENCODING="utf-8")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    env=env, check=True)
            # stderr 里有人脸锚定失败等关键提示，一并展示，别让用户看不到
            text = result.stdout
            if result.stderr.strip():
                text += "\n" + result.stderr
            self.analysis_out.setPlainText(text)

            n_val = self._last_float_on_line(result.stdout, "N =")
            r_val = self._last_float_on_line(result.stdout, "r =")
            if n_val:
                self.inputs["ratio"].setText(n_val)
            if r_val:
                self.inputs["head_shoulder"].setText(r_val)
            if n_val or r_val:
                self.analysis_out.append("\n✨ 已将提取到的比例参数自动同步至右侧测算面板！\n(请注意辨别头宽比例是否受到长发/袖子干扰)")

        except subprocess.CalledProcessError as e:
            self.analysis_out.setPlainText(f"❌ 执行失败:\n{e.stderr}\n{e.stdout}")

    @staticmethod
    def _last_float_on_line(text, marker):
        """取含 marker 的那一行里最后一个数字。

        例：'头身比  N = 全身高 / 头高 = 656 / 80 = 8.20' → '8.20'。
        （旧正则 r"N = .*? = ([\\d\\.]+)" 会抓到第 3 个数 656，是错的。）
        """
        for line in text.splitlines():
            if marker in line:
                nums = re.findall(r"\d+(?:\.\d+)?", line)
                if nums:
                    return nums[-1]
        return None

    def run_calc(self):
        self.calc_out.clear()
        
        cmd = [sys.executable, os.path.join(self.script_dir, "kigurumi_head_calc.py")]
        
        arg_map = {
            "height": "--height",
            "weight": "--weight",
            "shoulder": "--shoulder-width",
            "head_w": "--head-width",
            "head_h": "--head-height",
            "ratio": "--ratio",
            "head_shoulder": "--head-shoulder",
            "wig": "--wig"
        }
        
        for key, flag in arg_map.items():
            val = self.inputs[key].text().strip()
            if val:
                cmd.extend([flag, val])
                
        cmd.extend(["--mount", "chin"])
        
        self.calc_out.append("📐 正在推算三维建模包围盒与物理穿戴边界...")
        QApplication.processEvents()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.calc_out.setPlainText(result.stdout)
        except subprocess.CalledProcessError as e:
            self.calc_out.setPlainText(f"❌ 执行失败:\n{e.stderr}\n{e.stdout}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = KiguApp()
    window.show()
    sys.exit(app.exec())
