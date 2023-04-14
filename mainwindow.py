import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import QMainWindow, QLCDNumber
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('ui/mainwindow.ui', self)  # load ui
        self.cap = cv2.VideoCapture('/dev/video-camera0')  # open camera

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(30)  # fresh per 30ms

    def update_frame(self):
        ret, frame = self.cap.read()  # get camera frame
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.blur(rgb_image,(10,10))  # change color space
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # convert to QImage
            self.label.setPixmap(QPixmap.fromImage(q_image))  # show image


    def update_time(self):
        datetime = QDateTime.currentDateTime()
        self.lcdNumber.setSegmentStyle(QLCDNumber.Flat)
        self.lcdNumber.display(datetime.toString("hh:mm:ss"))   # show time
