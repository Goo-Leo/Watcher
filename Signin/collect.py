import sys
import cv2
import os
import numpy as np
import asyncio

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap, QImage


class Signin(QtWidgets.QDialog):
    def __init__(self):
        super(Signin, self).__init__()
        self.setWindowTitle("Signin")
        self.setGeometry(100, 100, 300, 150)

        self.name_label = QLabel("Name:", self)
        self.name_label.move(20, 20)
        self.name_input = QLineEdit(self)
        self.name_input.move(80, 20)

        self.id_label = QLabel("ID:", self)
        self.id_label.move(20, 50)
        self.id_input = QLineEdit(self)
        self.id_input.move(80, 50)

        self.start_button = QPushButton("Sign in", self)
        self.start_button.move(80, 90)
        self.start_button.clicked.connect(self.start_recognition)

        self.show()

        def start_recognition(self):
            name = self.name_input.text()
            id_num = self.id_input.text()

            self.face_recognition_thread.name = name
            self.face_recognition_thread.id_num = id_num

            pass

        def update_frame(self, pixmap):
            self.video_label.setPixmap(pixmap)


class FaceRecognitionThread(QThread):
    frame_signal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()

        self.camera = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def run(self):
        while True:
            ret, frame = self.camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(p)
            self.frame_signal.emit(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Signin_dialog = Signin()
    sys.exit(app.exec_())
