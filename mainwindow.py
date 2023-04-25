import sys
import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QMainWindow, QLCDNumber, QGraphicsBlurEffect, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('UI/mainwindow.ui', self)  # load UI
        self.show()
        self.cap = cv2.VideoCapture('/dev/video-camera0')  # open camera

        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(10)
        self.label.setGraphicsEffect(blur)

        self.pushButton.clicked.connect(self.loadMenu)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(30)  # fresh per 30ms

    def update_frame(self):
        ret, frame = self.cap.read()  # get camera frame
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # convert to QImage
            self.label.setPixmap(QPixmap.fromImage(q_image))  # show image

    def update_time(self):
        datetime = QDateTime.currentDateTime()
        self.lcdNumber.setSegmentStyle(QLCDNumber.Flat)
        self.lcdNumber.display(datetime.toString("hh:mm:ss"))  # show time

    def loadMenu(self):
        self.menu = Menu()
        self.menu.show()
        self.hide()


class Menu(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('UI/menu.ui', self)  # load menu.UI
        self.Home.clicked.connect(self.loadMainWindow)
        self.show()

    def loadMainWindow(self):
        self.mainwindow = MainWindow()
        self.mainwindow.show()
        self.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
