from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QMainWindow, QPushButton, QFileDialog, QCheckBox, QSlider, QMessageBox
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
import sys
import deep_learning
import os


class SegmentationThread(QThread):
    finished = pyqtSignal() 

    def __init__(self, input_folder, output_folder, threshold, perf_check, diff_check, t2h24_check):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.threshold = threshold
        self.perf_check = perf_check
        self.diff_check = diff_check
        self.t2h24_check = t2h24_check

    def run(self):
        outputPath = os.path.join(self.output_folder, str(max([0] + [int(c) for c in os.listdir(self.output_folder) if c.isdigit()]) + 1)) + "/"
        os.makedirs(outputPath)
        print(outputPath)

        if self.perf_check and self.t2h24_check:
            os.system("python eval.py --input " + self.input_folder + " --output " + outputPath + " --diff -1 --perf 1 --threshold " + self.threshold + " --t2h24 1")

        if self.diff_check:
            deep_learning.deep_learning_segmentation(self.input_folder, outputPath, self.diff_check, False, False)

        if self.perf_check and self.t2h24_check == False:
            os.system("python eval.py --input " + self.input_folder + " --output " + outputPath + " --diff -1 --perf 1 --threshold " + self.threshold + " --t2h24 -1")

        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation")
        self.setGeometry(200, 200, 400, 220)

        self.input_button = QPushButton("Select Input Folder", self)
        self.input_button.setGeometry(10, 10, 200, 30)
        self.input_button.clicked.connect(self.choose_input_folder)

        self.input_lineedit = QLineEdit(self)
        self.input_lineedit.setGeometry(220, 10, 170, 30)
        self.input_lineedit.setReadOnly(True)
        self.input_lineedit.setPlaceholderText("Selected Input Folder:")
        self.input_lineedit.setStyleSheet("color: rgba(128, 128, 128, 0.5);")

        self.output_button = QPushButton("Select Output Folder", self)
        self.output_button.setGeometry(10, 50, 200, 30)
        self.output_button.clicked.connect(self.choose_output_folder)

        self.output_lineedit = QLineEdit(self)
        self.output_lineedit.setGeometry(220, 50, 170, 30)
        self.output_lineedit.setReadOnly(True)
        self.output_lineedit.setPlaceholderText("Selected Output Folder:")
        self.output_lineedit.setStyleSheet("color: rgba(128, 128, 128, 0.5);")

        self.execute_button = QPushButton("Execute", self)
        self.execute_button.setGeometry(220, 100, 80, 40)
        self.execute_button.clicked.connect(self.start_segmentation)

        self.checkbox_diff = QCheckBox("diff", self)
        self.checkbox_diff.setGeometry(70, 90, 60, 20)
        self.checkbox_diff.setChecked(True)  
        self.checkbox_diff.setEnabled(True)  
        self.checkbox_perf = QCheckBox("perf", self)
        self.checkbox_perf.setGeometry(70, 110, 60, 20)
        self.checkbox_perf.setChecked(True)  
        self.checkbox_perf.setEnabled(True)  
        self.checkbox_T2H24 = QCheckBox("T2H24", self)
        self.checkbox_T2H24.setGeometry(70, 130, 70, 20)
        self.checkbox_T2H24.setChecked(True)  
        self.checkbox_T2H24.setEnabled(True)

        self.slider_label = QLineEdit(self)
        self.slider_label.setGeometry(10, 190, 380, 20)
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider_label.returnPressed.connect(self.update_slider_from_label)

        self.threshold_slider = QSlider(self)
        self.threshold_slider.setGeometry(10, 160, 380, 20)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(75)
        self.threshold_slider.setOrientation(Qt.Horizontal)
        self.threshold_slider.setEnabled(True)
        self.threshold_slider.valueChanged.connect(self.update_slider_label)

        self.update_slider_label(self.threshold_slider.value())

        self.segmentation_thread = SegmentationThread("", "", "", False, False, False)
        self.segmentation_thread.finished.connect(self.show_end_window)

    def choose_input_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        input_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if input_folder_path:
            self.input_lineedit.setText(input_folder_path)

    def choose_output_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        output_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if output_folder_path:
            self.output_lineedit.setText(output_folder_path)

    def start_segmentation(self):
        input_folder = self.input_lineedit.text()
        output_folder = self.output_lineedit.text()
        threshold = self.slider_label.text()
        perf_check = self.checkbox_perf.isChecked()
        diff_check = self.checkbox_diff.isChecked()
        t2h24_check = self.checkbox_T2H24.isChecked()

        self.segmentation_thread.input_folder = input_folder
        self.segmentation_thread.output_folder = output_folder
        self.segmentation_thread.threshold = threshold
        self.segmentation_thread.perf_check = perf_check
        self.segmentation_thread.diff_check = diff_check
        self.segmentation_thread.t2h24_check = t2h24_check

        self.segmentation_thread.start()

    def update_slider_label(self, value):
        self.slider_label.setText(f"{value/100.0:.2f}")

    def update_slider_from_label(self):
        value = float(self.slider_label.text())
        self.threshold_slider.setValue(int(value * 100))

    def show_end_window(self):
        end_window = EndWindow()
        end_window.exec_()


class EndWindow(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation Complete")
        self.setText("Segmentation process has ended.")
        self.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
