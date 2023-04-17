import cv2
import numpy as np
import asyncio
import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout

class Signup(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sign up')
        self.setWindowIcon(QIcon('icon.png'))
        self.setFixedSize(400, 200)

        # Create input fields
        self.name_label = QLabel('Name:', self)
        self.name_input = QLineEdit(self)
        self.id_label = QLabel('ID:', self)
        self.id_input = QLineEdit(self)


        # Create button
        self.capture_button = QPushButton('Capture', self)
        self.capture_button.clicked.connect(self.start_capture)

        # Create camera frame
        self.camera_frame = QLabel(self)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.id_label)
        layout.addWidget(self.id_input)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.camera_frame)

        # self.name_label.setGeometry(50, 50, 100, 30)
        # self.name_input.setGeometry(150, 50, 200, 30)
        # self.id_label.setGeometry(50, 100, 100, 30)
        # self.id_input.setGeometry(150, 100, 200, 30)
        # self.capture_button.setGeometry(150, 150, 100, 30)

        # Set layout
        self.setLayout(layout)

        # Create face recognition thread
        self.face_recognition_thread = FaceRecognitionThread()

    def start_capture(self):
        self.camera_frame.setFixedSize(640, 480)
        self.face_recognition_thread.start()
        self.face_recognition_thread.image_data.connect(self.update_image)
        pass

    def update_image(self, pixmap):
        self.camera_frame.setPixmap(pixmap)


class FaceRecognitionThread(QtCore.QThread):
    image_data = QtCore.pyqtSignal(QPixmap)
    face_cascade = None

    def __init__(self):
        super(FaceRecognitionThread, self).__init__()

        # load classfier
        self.face_cascade = cv2.CascadeClassifier(
            '/home/orangepi/miniforge3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

    def run(self):
        cap = cv2.VideoCapture('/dev/video-camera0')

        while True:
            ret, cv_image = cap.read()

            if not ret:
                continue

            # find faces
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap(q_image)
            self.image_data.emit(pixmap)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    signup = Signup()
    signup.show()
    sys.exit(app.exec_())
