import asyncio

import cv2
import sys
import os
import mysql.connector
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
import encoder

mydb = mysql.connector.connect(
    host="localhost",
    user="boss",
    password="password",
    database="myman"
)
mycursor = mydb.cursor()


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

        self.name_label.setGeometry(50, 50, 100, 30)
        self.name_input.setGeometry(150, 50, 200, 30)
        self.id_label.setGeometry(50, 100, 100, 30)
        self.id_input.setGeometry(150, 100, 200, 30)
        self.capture_button.setGeometry(150, 150, 100, 30)

        # Set layout
        self.setLayout(layout)

        # Create face recognition thread
        self.face_recognition_thread = FaceRecognitionThread()
        # self.finished.

    def start_capture(self):
        self.setFixedSize(680, 640)
        self.camera_frame.setFixedSize(640, 480)
        namestr = self.name_input.text()
        idstr = self.id_input.text()

        if os.path.exists("./Faces/{}".format(namestr)) == 0:
            os.makedirs("./Faces/{}".format(namestr), mode=0o755)

        self.face_recognition_thread.set_name_id(namestr, idstr)
        self.face_recognition_thread.start()
        self.face_recognition_thread.image_data.connect(self.update_image)

        pass

    def update_image(self, pixmap):
        self.camera_frame.setPixmap(pixmap)

    def recollecting(self):
        self.setFixedSize(400, 200)


class FaceRecognitionThread(QtCore.QThread):

    # define a signal for frame update
    image_data = QtCore.pyqtSignal(QPixmap)
    face_cascade = None
    name = ""
    id = ""
    finished = pyqtSignal()

    def __init__(self):
        super(FaceRecognitionThread, self).__init__()
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.counter = QtCore.QTimer(self)
        self.count = 0
        self.stop_flag = False

    def set_name_id(self, name_input, id_input):
        self.name = name_input
        self.id = id_input
        query = "SELECT * FROM usr_info WHERE id = %s"
        mycursor.execute(query, (id_input,))
        result = mycursor.fetchall()

        if len(result) > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("数据已存在")
            msg.setInformativeText("ID为%s的数据已经存在，是否更新？" % id_input)
            msg.setWindowTitle("提示")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if msg.exec_() == QMessageBox.Yes:
                # 如果用户选择更新，则执行更新操作
                query = "UPDATE usr_info SET name = %s, status = %s WHERE id = %s"
                mycursor.execute(query, (name_input, 1, id_input))
                mydb.commit()
            else:
                self.stop()
        else:
            sqlcmd = "INSERT INTO usr_info(id,name,status) VALUES (%s,%s,1)"
            val = (id_input,name_input)
            mycursor.execute(sqlcmd, val)
            mydb.commit()
            print("[INFO]", mycursor.rowcount, "info inserted")

    def collect_trainning(self, roi):
        goal_dir = os.path.join("Faces/", self.name)
        cv2.imwrite(goal_dir + "/face_{}.png".format(self.count), roi)
        self.count += 1
        if self.count == 100:
            encoding = encoder.Face_Encoder()
            encoding.encode_images(self.name)
            print("[INFO] done")
            self.stop()

    def run(self):
        cap = cv2.VideoCapture('/dev/video-camera0')
        while not self.stop_flag:
            ret, cv_image = cap.read()

            if not ret:
                continue

            # find faces
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            faces = self.face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = cv_image[y:y + h, x:x + w]
                self.collect_trainning(roi)
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap(q_image)
            self.image_data.emit(pixmap)  # send signal

        cap.release()
        self.finished.emit()
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_flag = True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    signup = Signup()
    signup.show()
    sys.exit(app.exec_())
