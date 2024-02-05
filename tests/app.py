import sys
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPalette, QColor

from PyQt6.QtCore import QThread

from for_app import design
from for_app.audio_input_ui import AudioInput
import onnxruntime as ort

COLORS = [
    "green",
    "red",
    "blue",
    "yellow",
    "magenta",
    "black",
    "cyan",
    "darkMagenta",
    "darkRed",
    "darkgreen",
    "grey"
]

KNOWN_COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "background"
]

class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)


class MyApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.button_clicked)
        self.verticalLayout.addWidget(Color("white"))
        self.label.setText("Запись не начата")

        self.stream_running = False
    
    def init_ort_session(self):
        #-----onnx-----
        save_onnx = "final_weights/model.onnx"
        self.ort_session = ort.InferenceSession(save_onnx, providers=["CUDAExecutionProvider"])

    
    def init_audio_input(self):
        #-----Audio-----
        self.audio_recording_thread = QThread()
        # print("Mult")
        self.audio = AudioInput(
            sleep_time=int(self.lineEdit.text()),
            delay=int(self.lineEdit_2.text()),
            save=True, 
            ort_session=self.ort_session,
            print_answer=False
        )
        self.audio.moveToThread(self.audio_recording_thread)
        self.audio.signal.connect(self.update_texts)

        self.audio_recording_thread.started.connect(self.audio.record)        

    def update_texts(self, class_token):
        if self.stream_running:
            self.verticalLayout.itemAt(0).widget().setParent(None)
            self.verticalLayout.addWidget(Color(COLORS[class_token]))
            self.label.setText(KNOWN_COMMANDS[class_token])
    
    def button_clicked(self):
        if not self.stream_running:
            # print("Add")
            self.init_audio_input()
            print(f"Start recording for {self.lineEdit.text()} secods")
            self.audio_recording_thread.start()
            self.stream_running = True
            self.pushButton.setText("Остановить запись")
            self.pushButton.clicked.connect(self.button_update)
    
    def button_update(self):
        if self.stream_running:
            self.verticalLayout.itemAt(0).widget().setParent(None)
            self.verticalLayout.addWidget(Color("white"))
            self.label.setText("Запись не начата")
            # print("Here")
            self.audio_recording_thread.quit()
            # print("we")
            self.audio_recording_thread.wait(5)
            # print("go")
            self.stream_running = False
            self.pushButton.setText("Начать запись")
            self.pushButton.clicked.connect(self.button_clicked)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()

    window.init_ort_session()

    app.exec()

if __name__ == "__main__":
    main()