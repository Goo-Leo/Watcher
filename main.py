import sys
from PyQt5.QtWidgets import QApplication
import mainwindow
import login

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # window = mainwindow.MainWindow()
    # window.show()
    # sys.exit(app.exec_())
    Login = login.FaceRecognizer(encodings_path='/home/orangepi/PycharmProjects/Watcher/Signup/encodings.pickle')
    Login.recognize()
